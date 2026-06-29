"use client";

import React, { useState } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Bot, User } from "lucide-react";
import Mermaid from "./Mermaid";
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis, Tooltip } from "recharts";

type MessageBubbleProps = {
  role: "user" | "assistant";
  content: string;
  metadata?: any;
};

export default function MessageBubble({ role, content, metadata }: MessageBubbleProps) {
  const isUser = role === "user";
  const [isShortened, setIsShortened] = useState(false);

  const isConversational = metadata?.route_reason === "conversational" || 
                           metadata?.generation_mode === "conversational_llm" || 
                           metadata?.generation_mode === "conversational" || 
                           content.includes("Alphalens AI") || 
                           (metadata && metadata.retrieved_docs_count === 0 && metadata.mcp_results_count === 0 && !metadata.cache_hit);

  // 5-State Guardrails Logic
  const getGuardrailState = () => {
    if (!metadata) return { color: "border-border", badge: null };
    
    if (metadata.cache_hit) {
      return { color: "border-green-500/50", badge: null };
    }
    
    if (isConversational) {
      return { color: "border-emerald-500/50", badge: null };
    }
    
    const resultsCount = metadata.results?.length || metadata.retrieved_docs_count || 0;
    if (resultsCount >= 3) {
      return { color: "border-emerald-500/50", badge: null };
    } else if (resultsCount > 0) {
      return { color: "border-amber-500/50", badge: null };
    } else if (metadata.generation_mode === "fallback") {
      return { color: "border-blue-500/50", badge: null };
    } else {
      return { color: "border-red-500/50", badge: null };
    }
  };

  const guardrail = getGuardrailState();
  const displayContent = isShortened && content.length > 300 ? content.substring(0, 300) + "..." : content;

  return (
    <div className={`flex w-full ${isUser ? "justify-end" : "justify-start"} mb-6 animate-in slide-in-from-bottom-4 duration-300 ease-out fill-mode-both`}>
      <div
        className={`flex max-w-[85%] ${
          isUser ? "flex-row-reverse" : "flex-row"
        } gap-4`}
      >
        {/* Avatar */}
        <div
          className={`h-10 w-10 flex-shrink-0 flex items-center justify-center rounded-full ${
            isUser ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
          }`}
        >
          {isUser ? <User size={20} /> : <Bot size={20} />}
        </div>

        {/* Content Box */}
        <div className="flex flex-col gap-2 min-w-0 flex-1">
          <div
            className={`px-5 py-4 rounded-2xl ${
              isUser
                ? "bg-gradient-to-r from-[#4A3AFF] to-[#7059DB] text-white rounded-tr-sm shadow-md"
                : `bg-[#1E1C2A] border border-[#2D2A3D] shadow-lg rounded-tl-sm text-white ${guardrail.color}`
            } prose prose-lg dark:prose-invert max-w-none break-words relative leading-relaxed`}
          >
            {/* Removed Guardrail Badge */}
            {isUser ? (
              <div className="whitespace-pre-wrap">{content}</div>
            ) : (
              <ReactMarkdown 
                remarkPlugins={[remarkGfm]}
                urlTransform={(url) => {
                  if (url.startsWith('data:')) return url;
                  return defaultUrlTransform(url);
                }}
                components={{
                  a: ({ node, ...props }) => {
                    const href = props.href || "";
                    if (href.startsWith("#source-") && metadata?.results) {
                      const sourceIndex = parseInt(href.replace("#source-", "")) - 1;
                      const source = metadata.results[sourceIndex];
                      if (source) {
                        return (
                          <a 
                            {...props} 
                            title={`Source: ${source.metadata?.source || 'Unknown'}\n\n${source.snippet}`}
                            className="inline-flex items-center justify-center w-5 h-5 mx-0.5 text-[10px] font-bold text-blue-600 bg-blue-100 rounded-full hover:bg-blue-200 cursor-pointer no-underline align-middle"
                            onClick={(e) => {
                              e.preventDefault();
                              const sourcesTrigger = document.querySelector('[data-value="sources"]');
                              if (sourcesTrigger) {
                                sourcesTrigger.scrollIntoView({ behavior: 'smooth', block: 'center' });
                              }
                            }}
                          >
                            {props.children}
                          </a>
                        );
                      }
                    }
                    return <a {...props} className="text-blue-600 hover:underline font-medium" />;
                  },
                  h3: ({ node, ...props }) => <h3 {...props} className="text-base font-bold text-foreground mt-6 mb-3 flex items-center gap-2" />,
                  hr: ({ node, ...props }) => <hr {...props} className="my-6 border-t border-border/80" />,
                  code: ({ node, className, children, ...props }: any) => {
                    const match = /language-(\w+)/.exec(className || "");
                    if (match && match[1] === "mermaid") {
                      return (
                        <div className="my-4 bg-muted/30 rounded-lg p-4 border border-border overflow-x-auto shadow-inner">
                          <Mermaid chart={String(children).replace(/\n$/, "")} />
                        </div>
                      );
                    }
                    if (match && match[1] === "json") {
                      try {
                        const data = JSON.parse(String(children));
                        if (data.chart_type && data.labels && data.values) {
                          const chartData = data.labels.map((label: string, i: number) => ({
                            name: label,
                            value: data.values[i]
                          }));
                          return (
                            <div className="my-6 p-4 bg-[#1E1C2A] rounded-xl border border-[#2D2A3D] shadow-lg">
                              <h4 className="text-center text-sm font-bold mb-4 text-white">{data.title || "Financial Chart"}</h4>
                              <div className="h-64 w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                  <BarChart data={chartData}>
                                    <XAxis dataKey="name" stroke="#8F8D9E" fontSize={12} tickLine={false} axisLine={false} />
                                    <YAxis stroke="#8F8D9E" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => String(val)} />
                                    <Tooltip cursor={{fill: '#2B283A'}} contentStyle={{backgroundColor: '#13111C', border: '1px solid #2D2A3D', color: '#fff', borderRadius: '8px'}} />
                                    <Bar dataKey="value" fill="#7059DB" radius={[4, 4, 0, 0]} />
                                  </BarChart>
                                </ResponsiveContainer>
                              </div>
                            </div>
                          );
                        }
                      } catch (e) {
                        // ignore parsing error and render as normal code
                      }
                    }
                    return (
                      <code className={`bg-muted px-1.5 py-0.5 rounded-md text-[13px] font-mono text-muted-foreground ${className || ""}`} {...props}>
                        {children}
                      </code>
                    );
                  },
                  table: ({ node, ...props }) => (
                    <div className="overflow-x-auto my-4 rounded-md border border-border shadow-sm">
                      <table {...props} className="w-full text-left text-sm border-collapse bg-card" />
                    </div>
                  ),
                  th: ({ node, ...props }) => <th {...props} className="bg-muted p-3 font-semibold border-b border-border text-foreground" />,
                  td: ({ node, ...props }) => <td {...props} className="p-3 border-b border-border/50 text-muted-foreground" />,
                  tr: ({ node, ...props }) => <tr {...props} className="hover:bg-muted/30 transition-colors" />,
                }}
              >
                {displayContent}
              </ReactMarkdown>
            )}
          </div>

          {/* Badges and Toggles */}
          {!isUser && (
            <div className="flex flex-wrap items-center justify-between gap-2 mt-1 px-1">
              <div className="flex gap-2"></div>
              
              {content.length > 300 && (
                <button 
                  onClick={() => setIsShortened(!isShortened)}
                  className="text-[11px] font-semibold text-[#7059DB] hover:text-[#4A3AFF] hover:underline bg-[#13111C] border border-[#2D2A3D] px-3 py-1 rounded-full transition-colors"
                >
                  {isShortened ? "Show Full Details" : "Show TL;DR Short Summary"}
                </button>
              )}
            </div>
          )}

          {/* Expanders for Assistant */}
          {!isUser && metadata && (
            <div className="mt-2">
              <Accordion className="w-full">
                {/* Execution Flow (Temporarily hidden) */}
                {/* 
                {metadata.flow_chart && (
                  <AccordionItem value="execution-flow" className="border-b-0 mb-2 border rounded-lg bg-card overflow-hidden">
                    <AccordionTrigger className="px-4 py-2 hover:bg-muted/50 text-xs font-semibold">
                      Execution Flow
                    </AccordionTrigger>
                    <AccordionContent className="p-0 border-t bg-black/40">
                      <Mermaid chart={metadata.flow_chart} />
                    </AccordionContent>
                  </AccordionItem>
                )}
                */}

                {/* Unified Sources and Raw JSON */}
                {!isConversational && (
                  <AccordionItem value="sources-data" className="border-b-0 mb-2 border border-[#2D2A3D] rounded-xl bg-[#1E1C2A] overflow-hidden shadow-md">
                    <AccordionTrigger className="px-4 py-3 hover:bg-[#2B283A] text-xs font-semibold text-white transition-colors">
                      <div className="flex items-center gap-2">
                        Sources & Inspector Data
                      </div>
                    </AccordionTrigger>
                    <AccordionContent className="border-t border-[#2D2A3D] bg-[#13111C]/50 flex flex-col">
                      
                      {/* Meta stats like latency */}
                      <div className="p-4 border-b border-[#2D2A3D] flex gap-2">
                        {metadata?.elapsed_ms !== undefined && (
                          <Badge variant="secondary" className="text-[10px] h-5 bg-muted/50 border-[#2D2A3D]">
                            ⏱️ {(metadata.elapsed_ms / 1000).toFixed(2)} s
                          </Badge>
                        )}
                        {metadata?.cache_hit && (
                          <Badge variant="secondary" className="text-[10px] h-5 bg-green-900/30 text-green-400 border-green-800">
                            ⚡ Cache Hit
                          </Badge>
                        )}
                      </div>
                      
                      {/* Citations/Sources */}
                      {metadata.results && metadata.results.length > 0 && metadata.generation_mode !== "fallback" && (
                        <div className="p-4 border-b border-[#2D2A3D]">
                          <h4 className="text-xs font-bold text-muted-foreground uppercase tracking-wider mb-3">Sources ({metadata.results.length})</h4>
                          <div className="flex flex-col gap-3">
                            {metadata.results.map((doc: any, i: number) => (
                              <details key={i} className="group bg-[#1E1C2A] rounded-lg border border-[#2D2A3D] shadow-sm [&_summary::-webkit-details-marker]:hidden">
                                <summary className="flex cursor-pointer items-center justify-between p-3 text-xs font-semibold">
                                  <span className="flex items-center gap-2 text-white">
                                    <span className="bg-[#7059DB]/20 text-[#7059DB] border border-[#7059DB]/30 w-5 h-5 flex items-center justify-center rounded-full font-bold">{i+1}</span> 
                                    📄 {doc.metadata?.source || "Unknown Source"}
                                  </span>
                                  <Badge variant="outline" className="text-[10px] bg-blue-900/30 text-blue-400 border-blue-800">Relevance: {(doc.score || 0).toFixed(3)}</Badge>
                                </summary>
                                <div className="p-3 border-t border-[#2D2A3D] text-[11px] leading-relaxed text-[#8F8D9E] bg-[#13111C]">
                                  {doc.snippet}
                                </div>
                              </details>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {/* Raw JSON */}
                      {metadata.results && metadata.results.length > 0 && metadata.generation_mode !== "fallback" && (
                        <div className="p-4 bg-black/80 overflow-x-auto">
                          <h4 className="text-xs font-bold text-muted-foreground uppercase tracking-wider mb-2">Raw JSON Payload</h4>
                          <pre className="text-[10px] text-green-400">
                            {JSON.stringify(metadata, null, 2)}
                          </pre>
                        </div>
                      )}
                    </AccordionContent>
                  </AccordionItem>
                )}
              </Accordion>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
