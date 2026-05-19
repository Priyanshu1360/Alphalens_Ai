from openai import OpenAI

from src.config import Config


def _get_llm_client():
    if not Config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY/XAI_API_KEY/GROQ_API_KEY is missing in environment variables"
        )
    kwargs = {"api_key": Config.OPENAI_API_KEY}
    if Config.OPENAI_BASE_URL:
        kwargs["base_url"] = Config.OPENAI_BASE_URL
    return OpenAI(**kwargs)


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
    user_content = (
        f"Question:\n{query}\n\n"
        f"Context:\n{_context_block(results)}\n\n"
        f"{Config.GENERATOR_OUTPUT_INSTRUCTION}"
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

    try:
        client = _get_llm_client()
    except Exception as exc:
        return {
            "answer": _fallback_answer(query, results),
            "generation_mode": "fallback_only",
            "llm_error": str(exc),
        }

    prompt, user_content = _build_prompt(query, results)

    try:
        answer = _answer_via_responses(client, Config.LLM_MODEL, prompt, user_content)
        if answer:
            return {"answer": answer, "generation_mode": "llm_responses"}
    except Exception:
        pass

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
        return {
            "answer": _fallback_answer(query, results),
            "generation_mode": "fallback_after_llm_error",
            "llm_error": str(exc),
        }
