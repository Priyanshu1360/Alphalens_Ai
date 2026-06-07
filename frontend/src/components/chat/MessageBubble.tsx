"use client";

import React from "react";
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

  return (
    <div className={`flex w-full ${isUser ? "justify-end" : "justify-start"} mb-6`}>
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
            className={`px-5 py-3 rounded-2xl ${
              isUser
                ? "bg-primary text-primary-foreground rounded-tr-sm"
                : "bg-muted/50 border border-border rounded-tl-sm text-foreground"
            } shadow-sm prose prose-sm dark:prose-invert max-w-none break-words`}
          >
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
                            className="inline-flex items-center justify-center w-5 h-5 mx-0.5 text-[10px] font-bold text-blue-400 bg-blue-900/30 rounded-full hover:bg-blue-800/50 cursor-pointer no-underline align-middle"
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
                    return <a {...props} className="text-blue-400 hover:underline" />;
                  },
                  h3: ({ node, ...props }) => <h3 {...props} className="text-base font-extrabold text-foreground mt-6 mb-3 flex items-center gap-2" />,
                  hr: ({ node, ...props }) => <hr {...props} className="my-6 border-t-2 border-border/80 border-dashed" />,
                  code: ({ node, className, children, ...props }: any) => {
                    const match = /language-(\w+)/.exec(className || "");
                    if (match && match[1] === "mermaid") {
                      return (
                        <div className="my-4 bg-black/40 rounded-lg p-4 border border-border overflow-x-auto">
                          <Mermaid chart={String(children).replace(/\n$/, "")} />
                        </div>
                      );
                    }
                    return (
                      <code className={`bg-muted px-1.5 py-0.5 rounded-md text-[13px] font-mono ${className || ""}`} {...props}>
                        {children}
                      </code>
                    );
                  },
                  table: ({ node, ...props }) => (
                    <div className="overflow-x-auto my-4 rounded-md border border-border">
                      <table {...props} className="w-full text-left text-sm border-collapse" />
                    </div>
                  ),
                  th: ({ node, ...props }) => <th {...props} className="bg-muted/80 p-3 font-semibold border-b border-border text-foreground" />,
                  td: ({ node, ...props }) => <td {...props} className="p-3 border-b border-border/50 text-muted-foreground" />,
                  tr: ({ node, ...props }) => <tr {...props} className="hover:bg-muted/20 transition-colors" />,
                }}
              >
                {content.replace(/\[(\d+)\]/g, '[[$1]](#source-$1)')}
              </ReactMarkdown>
            )}
          </div>

          {/* Badges */}
          {!isUser && metadata && (
            <div className="flex flex-wrap gap-2 mt-1 px-1">
              {metadata.elapsed_ms !== undefined && (
                <Badge variant="secondary" className="text-[10px] h-5">
                  ⏱️ {(metadata.elapsed_ms / 1000).toFixed(2)} s
                </Badge>
              )}
              {metadata.cache_hit && (
                <Badge variant="secondary" className="text-[10px] h-5 bg-green-950 text-green-400">
                  ⚡ Cache Hit
                </Badge>
              )}

            </div>
          )}

          {/* Expanders for Assistant */}
          {!isUser && metadata && (
            <div className="mt-2">
              <Accordion type="single" collapsible="true" className="w-full">
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

                {/* Sources */}
                {metadata.results && metadata.results.length > 0 && (
                  <AccordionItem value="sources" className="border-b-0 mb-2 border rounded-lg bg-card overflow-hidden">
                    <AccordionTrigger className="px-4 py-2 hover:bg-muted/50 text-xs font-semibold">
                      Sources ({metadata.results.length} chunks)
                    </AccordionTrigger>
                    <AccordionContent className="p-4 border-t">
                      <div className="h-48 w-full mb-6">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={metadata.results}>
                            <XAxis dataKey="chunk_id" hide />
                            <YAxis domain={[0, 1]} />
                            <Tooltip
                              contentStyle={{ backgroundColor: "#1e293b", borderColor: "#334155" }}
                              itemStyle={{ color: "#e2e8f0" }}
                            />
                            <Bar dataKey="score" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="flex flex-col gap-3">
                        {metadata.results.map((doc: any, i: number) => (
                          <div key={i} className="p-3 bg-muted/50 rounded-md border border-border/50 text-xs">
                            <div className="flex justify-between items-center mb-2">
                              <span className="font-semibold text-primary/80">📄 {doc.metadata?.source || "Unknown Source"}</span>
                              <Badge variant="outline">Score: {doc.score?.toFixed(3)}</Badge>
                            </div>
                            <p className="line-clamp-4 text-muted-foreground text-[11px] leading-relaxed">
                              {doc.snippet}
                            </p>
                          </div>
                        ))}
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                )}

                {/* Raw JSON */}
                <AccordionItem value="raw-json" className="border-b-0 border rounded-lg bg-card overflow-hidden">
                  <AccordionTrigger className="px-4 py-2 hover:bg-muted/50 text-xs font-semibold">
                    🛠️ Raw JSON
                  </AccordionTrigger>
                  <AccordionContent className="p-4 border-t bg-black/60 overflow-x-auto">
                    <pre className="text-[10px] text-green-400">
                      {JSON.stringify(metadata, null, 2)}
                    </pre>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
