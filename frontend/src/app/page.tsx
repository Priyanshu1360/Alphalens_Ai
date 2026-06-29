"use client";

import React, { useState, useEffect, useRef } from "react";
import Sidebar from "@/components/chat/Sidebar";
import MessageBubble from "@/components/chat/MessageBubble";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send, Loader2, Menu, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";

type Message = {
  role: "user" | "assistant";
  content: string;
  metadata?: any;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [modelStatus, setModelStatus] = useState<any>({ is_ready: true, downloaded_mb: 0, model: "" });
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);
  
  const [settings, setSettings] = useState({
    mode: "hybrid",
    limit: 5,
    prefetch_limit: 40,
    rerank: true,
    strategy: "auto"
  });

  const [threads, setThreads] = useState<any[]>([]);
  const [currentThread, setCurrentThread] = useState<string | null>(null);
  const [historyCache, setHistoryCache] = useState<any[]>([]);
  const [recentQueries, setRecentQueries] = useState<string[]>([]);

  // Auto scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  // Poll Backend Status
  useEffect(() => {
    let interval: any;
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/status`);
        if (res.ok) {
          const data = await res.json();
          setModelStatus(data);
          if (data.is_ready && interval) clearInterval(interval);
        }
      } catch (err) {}
    };
    fetchStatus();
    interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Fetch History
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/chat-history?limit=100`);
        if (res.ok) {
          const data = await res.json();
          if (data.items) {
            setHistoryCache(data.items);
            const threadMap = new Map();
            data.items.forEach((item: any) => {
              if (item.thread_id && !threadMap.has(item.thread_id)) {
                threadMap.set(item.thread_id, item.query);
              }
            });
            const uniqueThreads = Array.from(threadMap.entries()).map(([id, query]) => ({ id, query }));
            setThreads(uniqueThreads);
            
            const queries = Array.from(new Set(data.items.map((item: any) => item.query))).slice(0, 10) as string[];
            setRecentQueries(queries);
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
      
      let activeThreadId = currentThread;
      if (!activeThreadId) {
        activeThreadId = crypto.randomUUID();
        setCurrentThread(activeThreadId);
      }
      
      const payload: any = {
        query,
        mode: settings.mode,
        limit: settings.limit,
        prefetch_limit: settings.prefetch_limit,
        rerank: settings.rerank,
        thread_id: activeThreadId,
      };

      const res = await fetch(`${API_BASE_URL}${endpoint}`, {
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
          recentQueries={recentQueries}
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
        {!modelStatus.is_ready && (
          <div className="absolute top-0 left-0 right-0 bg-blue-50 border-b border-blue-200 text-blue-700 p-2 text-center text-xs font-semibold flex items-center justify-center gap-2 z-50 shadow-sm">
            <Loader2 className="animate-spin" size={14} />
            System Initializing: Downloading AI Model ({modelStatus.downloaded_mb} MB / ~1300 MB)
          </div>
        )}

        {/* Header */}
        <div className={`flex items-center justify-between p-4 border-b bg-background/95 backdrop-blur z-10 ${!modelStatus.is_ready ? 'mt-8' : ''}`}>
          <Button variant="ghost" size="icon" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
            <Menu size={20} />
          </Button>
          <div className="flex items-center justify-center">
            <h2 className="text-4xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-emerald-500">
              Alphalens AI
            </h2>
          </div>
          <div className="w-10"></div>
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 pb-32">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center opacity-60">
              <div className="w-24 h-24 bg-[#1E1C2A] border border-[#2D2A3D] rounded-full flex items-center justify-center mb-6 shadow-xl">
                <span className="text-5xl text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-indigo-400">✤</span>
              </div>
              <h1 className="text-5xl font-extrabold mb-5 text-white tracking-tight">How can I help you?</h1>
              <p className="text-[#8F8D9E] text-base text-center max-w-md mb-8">Try asking about financial data, Apple's gross margin, or analyze crypto trends directly.</p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl mt-4">
                {["What was meta most profitable year?", "Summarize Apple's gross margin in 2024", "How did Microsoft perform in Q1?", "Analyze Amazon's cloud revenue growth"].map((q) => (
                  <Button 
                    key={q} 
                    variant="outline" 
                    className="justify-start text-left h-auto py-3 px-5 text-sm whitespace-normal rounded-xl text-muted-foreground border-[#2D2A3D] bg-[#1E1C2A] hover:bg-[#2B283A] hover:text-white transition-all shadow-sm"
                    onClick={() => {
                      setInput(q);
                      setTimeout(() => {
                        const form = document.getElementById("chat-form");
                        if (form) form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
                      }, 100);
                    }}
                  >
                    {q}
                  </Button>
                ))}
              </div>
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
        <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-background via-background/90 to-transparent">
          <form 
            id="chat-form"
            onSubmit={handleSubmit} 
            className="max-w-4xl mx-auto relative bg-[#1E1C2A] rounded-2xl border border-[#2D2A3D] shadow-2xl overflow-hidden focus-within:ring-2 focus-within:ring-[#7059DB]/50 transition-all"
          >
            <Textarea 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask anything..."
              className="w-full resize-none border-0 bg-transparent py-5 px-6 pr-16 focus-visible:ring-0 min-h-[72px] max-h-48 text-white placeholder:text-[#8F8D9E] text-lg md:text-lg font-medium"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e as any);
                }
              }}
            />
            <Button 
              type="submit" 
              size="icon"
              disabled={loading || !input.trim()}
              className="absolute right-3 bottom-3 rounded-xl bg-gradient-to-r from-[#4A3AFF] to-[#7059DB] text-white hover:opacity-90 h-10 w-10 disabled:opacity-50 disabled:from-muted disabled:to-muted"
            >
              <Send size={18} className={loading ? "animate-pulse" : ""} />
            </Button>
          </form>
          <div className="text-center mt-3 text-[10px] text-[#8F8D9E] font-medium tracking-wide">
            Powered by Alphalens RAG Engine • Verify critical financial data
          </div>
        </div>
      </div>
    </div>
  );
}
