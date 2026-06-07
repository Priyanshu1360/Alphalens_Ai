"use client";

import React, { useState, useEffect, useRef } from "react";
import Sidebar from "@/components/chat/Sidebar";
import MessageBubble from "@/components/chat/MessageBubble";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send, Loader2, Menu } from "lucide-react";

type Message = {
  role: "user" | "assistant";
  content: string;
  metadata?: any;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  
  const [settings, setSettings] = useState({
    mode: "hybrid",
    limit: 6,
    prefetch_limit: 50,
    rerank: true,
    strategy: "auto"
  });

  const [threads, setThreads] = useState<string[]>([]);
  const [currentThread, setCurrentThread] = useState<string | null>(null);
  const [historyCache, setHistoryCache] = useState<any[]>([]);

  // Auto scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  // Fetch History
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch("http://localhost:8000/chat-history?limit=100");
        if (res.ok) {
          const data = await res.json();
          if (data.items) {
            setHistoryCache(data.items);
            const uniqueThreads = Array.from(new Set(data.items.map((item: any) => item.thread_id).filter(Boolean))) as string[];
            setThreads(uniqueThreads);
          }
        }
      } catch (err) {
        console.error("Failed to fetch history", err);
      }
    };
    fetchHistory();
  }, []);

  const handleSelectThread = (threadId: string) => {
    setCurrentThread(threadId);
    const threadItems = historyCache.filter((item: any) => item.thread_id === threadId);
    threadItems.sort((a: any, b: any) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
    
    const loadedMessages: Message[] = [];
    threadItems.forEach((item: any) => {
      loadedMessages.push({ role: "user", content: item.query });
      loadedMessages.push({ 
        role: "assistant", 
        content: item.answer, 
        metadata: {
          ...item.metadata,
          flow_chart: buildMermaid({ ...item.metadata, query: item.query })
        } 
      });
    });
    setMessages(loadedMessages);
  };

  const handleNewChat = () => {
    setMessages([]);
    setCurrentThread(null);
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;

    const query = input;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setLoading(true);

    try {
      const endpoint = settings.strategy === "auto" ? "/agent-query" : "/query";
      
      const payload: any = {
        query,
        mode: settings.mode,
        limit: settings.limit,
        prefetch_limit: settings.prefetch_limit,
        rerank: settings.rerank,
      };
      if (currentThread) {
        payload.thread_id = currentThread;
      }

      const res = await fetch(`http://localhost:8000${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error("Failed to fetch response");
      }

      const data = await res.json();
      
      const answer = data.answer || data.final_answer || "No response generated.";
      
      // Reconstruct metadata for UI
      const meta = {
        elapsed_ms: data.elapsed_ms,
        cache_hit: data.cache_hit,
        generation_mode: data.generation_mode || data.route_reason || "default",
        results: data.results || data.retrieved_docs || [],
        flow_chart: buildMermaid(data)
      };

      setMessages((prev) => [...prev, { role: "assistant", content: answer, metadata: meta }]);
    } catch (err: any) {
      setMessages((prev) => [...prev, { role: "assistant", content: `**Error:** ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const buildMermaid = (data: any) => {
    const safeQuery = (data.query || "").replace(/["\n]/g, " ");
    const cacheHit = data.cache_hit ? "Cache Hit" : "Cache Miss";
    let flow = `flowchart TD
      Q(["👤 User Query<br/>${safeQuery}"])
      S[("🗄️ Session Manager")]
      C["⚡ Cache Agent<br/>${cacheHit}"]
      A[/"✅ Final Answer"/]
      RAW["🛠️ Raw JSON"]
      `;
    
    if (data.cache_hit) {
      flow += `
      Q --> S
      S --> C
      C --> A
      A -.-> RAW`;
    } else {
      flow += `
      R["🔍 Retrieval Agent<br/>Docs: ${(data.results || data.retrieved_docs || []).length}"]
      G["🧠 Generation Agent"]
      Q --> S
      S --> C
      C --> R
      R --> G
      G --> A
      A -.-> RAW`;
    }

    flow += `
      classDef userNode fill:#2d1b2e,stroke:#ff8a8a,stroke-width:2px,color:#ffe3e3,rx:10,ry:10;
      classDef systemNode fill:#1a2c4a,stroke:#8ab4ff,stroke-width:2px,color:#d7e3ff,rx:10,ry:10;
      classDef finalNode fill:#1a4a2c,stroke:#8affab,stroke-width:2px,color:#e3ffe8,rx:10,ry:10;
      class Q userNode;
      class S,C systemNode;
      class A finalNode;
    `;
    return flow;
  };

  return (
    <div className="flex h-screen w-full bg-background text-foreground overflow-hidden">
      {/* Sidebar */}
      {isSidebarOpen && (
        <Sidebar 
          settings={settings} 
          setSettings={setSettings} 
          onNewChat={handleNewChat}
          threads={threads}
          currentThread={currentThread}
          onSelectThread={handleSelectThread}
          onSuggestionClick={(q: string) => {
            setInput(q);
            setTimeout(() => {
              const form = document.getElementById("chat-form");
              if (form) form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
            }, 100);
          }}
          closeSidebar={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-full relative">
        {/* Header */}
        <div className="flex items-center justify-between p-3 border-b bg-background/95 backdrop-blur z-10">
          <Button variant="ghost" size="icon" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
            <Menu size={20} />
          </Button>
          <h2 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
            Alphalens AI
          </h2>
          <div className="w-10"></div> {/* Spacer for perfect centering */}
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 pb-32">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center opacity-50">
              <h1 className="text-4xl font-bold mb-4">How can I help you?</h1>
              <p>Try asking about financial data, Apple's gross margin, etc.</p>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto flex flex-col">
              {messages.map((m, idx) => (
                <MessageBubble key={idx} role={m.role} content={m.content} metadata={m.metadata} />
              ))}
              {loading && (
                <div className="flex items-center gap-3 text-muted-foreground mt-4 ml-2">
                  <Loader2 className="animate-spin" size={20} />
                  <span className="text-sm">Thinking...</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input Box */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-background via-background to-transparent pt-10 pb-6 px-4">
          <div className="max-w-4xl mx-auto">
            <form id="chat-form" onSubmit={handleSubmit} className="relative bg-muted/30 border border-border rounded-xl shadow-lg flex items-end">
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit();
                  }
                }}
                placeholder="Ask about financial insights..."
                className="min-h-[60px] w-full resize-none bg-transparent border-0 focus-visible:ring-0 px-4 py-4 text-base"
                rows={1}
              />
              <div className="p-3">
                <Button 
                  type="submit" 
                  disabled={!input.trim() || loading} 
                  size="icon"
                  className="rounded-full h-10 w-10 transition-transform active:scale-95"
                >
                  <Send size={18} />
                </Button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
