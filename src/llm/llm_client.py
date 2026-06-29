from openai import OpenAI

from src.utils.config import Config


def _get_llm_clients():
    if not Config.OPENAI_API_KEYS:
        raise ValueError(
            "OPENAI_API_KEY/XAI_API_KEY/GROQ_API_KEY is missing in environment variables"
        )
    clients = []
    for key in Config.OPENAI_API_KEYS:
        kwargs = {"api_key": key}
        if Config.OPENAI_BASE_URL:
            kwargs["base_url"] = Config.OPENAI_BASE_URL
        clients.append(OpenAI(**kwargs))
    return clients


def _context_block(results):
    context_parts = []
    for rank, item in enumerate(results[: Config.GENERATION_CONTEXT_CHUNKS], start=1):
        snippet = str(item.get("payload", {}).get("text", ""))[: Config.GENERATION_CONTEXT_CHARS]
        source = item.get("file_name") or "unknown"
        context_parts.append(f"[{rank}] Source: {source}\n{snippet}")
    return "\n\n".join(context_parts)


def _fallback_answer(query, results):
    if not results:
        return "No relevant context found in retrieval results."

    lines = [f"Query: {query}", "Top evidence from retrieved chunks:"]
    top_sources = max(1, Config.GENERATION_FALLBACK_TOP_SOURCES)
    for rank, item in enumerate(results[:top_sources], start=1):
        source = item.get("file_name") or "unknown"
        company = item.get("company") or "unknown"
        snippet = str(item.get("snippet", "")).strip()
        lines.append(f"{rank}. [{company}] {source}: {snippet}")
    return "\n".join(lines)


def _build_prompt(query, results):
    prompt = Config.GENERATOR_SYSTEM_PROMPT
    chart_terms = [
        "plot",
        "graph",
        "chart",
        "line chart",
        "bar chart",
        "pie chart",
        "trend",
        "visualize",
        "visualise",
        "show graph",
        "show chart",
    ]
    table_terms = [
        "table",
        "tabular",
        "compare",
        "comparison",
        "vs",
        "versus",
    ]
    
    chart_instruction = ""
    table_instruction = ""
    query_lower = (query or "").lower()
    
    if any(term in query_lower for term in chart_terms):
        chart_instruction = (
            "\n\nThe user is asking for a chart or trend view. "
            "Do not format the answer as a table. "
            "CRITICAL: You MUST APPEND a strictly formatted JSON block (```json ... ```) at the very end of your response containing the chart data. "
            "The JSON must have keys: 'chart_type' (bar/line/pie), 'title' (string), 'labels' (array of strings), and 'values' (array of numbers). "
            "The numbers in the values array MUST be raw integers or floats without commas (e.g., use 47302 instead of 47,302). "
            "Do not use Mermaid."
        )
    elif any(term in query_lower for term in table_terms):
        table_instruction = (
            "\n\nThe user is asking for a comparison or table. "
            "You MUST format the detailed data as a Markdown Table. "
            "Leave a blank line before and after the table. "
            "Do not use standard bullet points for the data if a table is requested."
        )
        
    guardrail = (
        "\n\nGUARDRAIL: If the user asks for a chart, graph, comparison, or table, you MUST provide the requested format. If the data is not in the Context, explicitly state that you cannot provide the requested format due to missing data. "
        "You must ONLY answer based on the provided Context. "
        "If a text question is outside the scope, reply by stating that the query is outside the scope of the provided financial data. "
        "Then, add a section exactly named '### ❓ Suggested Questions' and suggest 1-3 alternative questions the user could ask based on the available Context, formatted as bullet points. "
        "CRITICAL: Do NOT provide the answers to the questions you suggest. Just list the suggested questions. "
        "CRITICAL: Do NOT duplicate your response or include meta tags like 'summary:'."
    )
    user_content = (
        f"Question:\n{query}\n\n"
        f"Context:\n{_context_block(results)}\n\n"
        f"{Config.GENERATOR_OUTPUT_INSTRUCTION}"
        f"{chart_instruction}"
        f"{table_instruction}"
        f"{guardrail}"
    )
    return prompt, user_content


def _answer_via_responses(client, model, prompt, user_content):
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return (response.output_text or "").strip()


def _answer_via_chat_completions(client, model, prompt, user_content):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )
    choice = response.choices[0] if response.choices else None
    message = getattr(choice, "message", None)
    content = getattr(message, "content", "")
    return (content or "").strip()


def generate_answer(query, results):
    if not Config.LLM_ENABLED:
        return {
            "answer": _fallback_answer(query, results),
            "generation_mode": "fallback_only",
        }

    if results:
        max_score = max([float(r.get("rerank_score") or r.get("score") or 0) for r in results])
        if max_score < 0.2:
            return {
                "answer": "No relevant context found in the database to answer your query. Please try rephrasing or asking something else.",
                "generation_mode": "guardrail_rejected"
            }

    try:
        clients = _get_llm_clients()
    except Exception as exc:
        return {
            "answer": _fallback_answer(query, results),
            "generation_mode": "fallback_only",
            "llm_error": str(exc),
        }

    prompt, user_content = _build_prompt(query, results)

    last_exc = None
    for client in clients:
        try:
            answer = _answer_via_responses(client, Config.LLM_MODEL, prompt, user_content)
            if answer:
                return {"answer": answer, "generation_mode": "llm_responses"}
        except Exception as exc:
            last_exc = exc
            pass

    for client in clients:
        try:
            answer = _answer_via_chat_completions(
                client,
                Config.LLM_MODEL,
                prompt,
                user_content,
            )
            if not answer:
                raise ValueError("Empty response from chat completions API")
            return {"answer": answer, "generation_mode": "llm_chat_completions"}
        except Exception as exc:
            last_exc = exc

    return {
        "answer": _fallback_answer(query, results),
        "generation_mode": "fallback_after_llm_error",
        "llm_error": str(last_exc),
    }


def generate_conversational_answer(query):
    """
    Generates a conversational response using the LLM with a specific AI persona prompt.
    """
    if not Config.LLM_ENABLED:
        return {
            "answer": "Hello, I am Alphalens AI! What can I help you with today?",
            "generation_mode": "conversational",
        }

    try:
        clients = _get_llm_clients()
    except Exception as exc:
        return {
            "answer": "Hello, I am Alphalens AI! What can I help you with today?",
            "generation_mode": "conversational",
        }

    prompt = (
        "You are Alphalens AI, a helpful and friendly financial AI assistant. "
        "Your job is to help users analyze financial reports, earnings calls, and revenue data for companies like Apple, Meta, and Amazon. "
        "Keep your answers short, friendly, and directly address the user's greeting or question about your capabilities. "
        "Do not use markdown formatting like 'Summary:' or 'Detailed Answer:' for conversational queries."
    )

    last_exc = None
    for client in clients:
        try:
            answer = _answer_via_chat_completions(
                client,
                Config.LLM_MODEL,
                prompt,
                query,
            )
            if answer:
                return {"answer": answer, "generation_mode": "conversational_llm"}
        except Exception as exc:
            last_exc = exc

    return {
        "answer": "Hello, I am Alphalens AI! What can I help you with today?",
        "generation_mode": "conversational",
    }
