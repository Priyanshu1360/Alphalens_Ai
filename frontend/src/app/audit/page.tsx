"use client";

import React, { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Loader2, ArrowLeft } from "lucide-react";
import Link from "next/link";

export default function AuditLogs() {
  const [logs, setLogs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/chat-history?limit=100");
        if (res.ok) {
          const data = await res.json();
          if (data.items) {
            setLogs(data.items.sort((a: any, b: any) => new Date(b.created_at_utc || b.created_at.replace(" IST", "")).getTime() - new Date(a.created_at_utc || a.created_at.replace(" IST", "")).getTime()));
          }
        }
      } catch (err) {
        console.error("Failed to fetch history", err);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center gap-4 mb-8">
          <Link href="/">
            <Button variant="outline" size="icon">
              <ArrowLeft size={16} />
            </Button>
          </Link>
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Audit Logs</h1>
            <p className="text-muted-foreground">Comprehensive history of all RAG queries and system routing decisions.</p>
          </div>
        </div>

        {loading ? (
          <div className="flex justify-center p-20">
            <Loader2 className="animate-spin text-muted-foreground" size={40} />
          </div>
        ) : (
          <div className="border rounded-xl shadow-sm bg-card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-muted-foreground uppercase bg-muted/50 border-b">
                  <tr>
                    <th className="px-6 py-4 font-semibold">Timestamp</th>
                    <th className="px-6 py-4 font-semibold">Query</th>
                    <th className="px-6 py-4 font-semibold">Route</th>
                    <th className="px-6 py-4 font-semibold">Cache</th>
                    <th className="px-6 py-4 font-semibold">Latency</th>
                    <th className="px-6 py-4 font-semibold">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {logs.map((log) => (
                    <tr key={log.id} className="hover:bg-muted/20 transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap text-muted-foreground font-mono text-[11px]">
                        {new Date(log.created_at_utc || log.created_at.replace(" IST", "")).toLocaleString()}
                      </td>
                      <td className="px-6 py-4 font-medium max-w-xs truncate" title={log.query}>
                        {log.query}
                      </td>
                      <td className="px-6 py-4">
                        <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                          {log.route || "agent-query"}
                        </Badge>
                      </td>
                      <td className="px-6 py-4">
                        {log.metadata?.cache_hit ? (
                          <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">Hit</Badge>
                        ) : (
                          <span className="text-muted-foreground text-xs">Miss</span>
                        )}
                      </td>
                      <td className="px-6 py-4 text-muted-foreground font-mono text-[11px]">
                        {log.metadata?.elapsed_ms ? `${(log.metadata.elapsed_ms / 1000).toFixed(2)}s` : "-"}
                      </td>
                      <td className="px-6 py-4">
                        <Badge variant="outline" className="bg-emerald-50 text-emerald-700 border-emerald-200">
                          Success
                        </Badge>
                      </td>
                    </tr>
                  ))}
                  {logs.length === 0 && (
                    <tr>
                      <td colSpan={6} className="px-6 py-12 text-center text-muted-foreground">
                        No audit logs found.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
