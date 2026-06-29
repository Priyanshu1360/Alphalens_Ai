"use client";

import React, { useEffect, useState, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { useTheme } from "next-themes";
import { Moon, Sun, MessageSquarePlus, X } from "lucide-react";
import Link from "next/link";

export default function Sidebar({ settings, setSettings, onNewChat, threads, currentThread, onSelectThread, recentQueries, onSuggestionClick, closeSidebar }: any) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(288);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const isResizing = useRef(false);

  const resize = useCallback((e: MouseEvent) => {
    if (isResizing.current && sidebarRef.current) {
      const newWidth = e.clientX - sidebarRef.current.getBoundingClientRect().left;
      if (newWidth > 200 && newWidth < 800) {
        setSidebarWidth(newWidth);
      }
    }
  }, []);

  const stopResizing = useCallback(() => {
    isResizing.current = false;
    document.removeEventListener("mousemove", resize);
    document.removeEventListener("mouseup", stopResizing);
    document.body.style.cursor = "default";
  }, [resize]);

  const startResizing = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;
    document.addEventListener("mousemove", resize);
    document.addEventListener("mouseup", stopResizing);
    document.body.style.cursor = "col-resize";
  }, [resize, stopResizing]);

  useEffect(() => {
    setMounted(true);
    return () => {
      document.removeEventListener("mousemove", resize);
      document.removeEventListener("mouseup", stopResizing);
    };
  }, [resize, stopResizing]);

  return (
    <div 
      ref={sidebarRef}
      style={{ width: `${sidebarWidth}px` }}
      className="border-r border-[#2D2A3D] bg-[#13111C] text-[#ffffff] h-screen overflow-y-auto p-4 flex flex-col gap-6 absolute md:relative z-50 shadow-2xl md:shadow-none flex-shrink-0"
    >
      {/* Resizer Handle */}
      <div 
        onMouseDown={startResizing}
        className="absolute top-0 right-0 w-2 h-full cursor-col-resize hover:bg-[#7059DB] hover:opacity-100 opacity-0 transition-opacity z-[60]"
      />
      
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-indigo-400 flex items-center gap-2">
          <span className="text-4xl">✤</span> Alphalens
        </h2>
        <div className="flex gap-1">
          {mounted && (
            <div className="w-8"></div>
          )}
        <Button variant="ghost" size="icon" onClick={closeSidebar} className="md:hidden" title="Close Sidebar">
          <X size={18} />
        </Button>
        </div>
      </div>

      <Button onClick={onNewChat} className="w-full flex items-center justify-start gap-3 rounded-full bg-gradient-to-r from-[#4A3AFF] to-[#7059DB] text-white hover:opacity-90 shadow-md border-0 h-12 px-6 text-lg font-bold" variant="default">
        <MessageSquarePlus size={22} /> New Chat
      </Button>

      {threads && threads.length > 0 && (
        <div className="flex flex-col gap-2">
          <Label className="text-base font-semibold">Saved Threads</Label>
          <Select 
            value={currentThread || ""} 
            onValueChange={onSelectThread}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select a past chat..." />
            </SelectTrigger>
            <SelectContent>
              {threads.map((t: string) => (
                <SelectItem key={t} value={t}>Chat {t.substring(0, 6)}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      <Separator />

      {recentQueries && recentQueries.length > 0 && (
        <div className="flex flex-col gap-4 mt-2">
          <h3 className="text-sm uppercase tracking-wider font-bold text-muted-foreground px-2">Recent Queries</h3>
          <div className="flex flex-col gap-1">
            {recentQueries.map((q: string) => (
              <Button 
                key={q} 
                variant="ghost" 
                className="w-full justify-start text-left h-auto py-3 px-5 text-sm whitespace-normal rounded-xl text-muted-foreground hover:bg-[#2B283A] hover:text-white transition-colors"
                onClick={() => onSuggestionClick(q)}
              >
                {q}
              </Button>
            ))}
          </div>
        </div>
      )}

      <Separator />

      <div className="flex flex-col gap-4">
        <h3 className="text-lg font-bold text-primary">Retrieval Settings</h3>
        
        <div className="space-y-2">
          <Label className="text-base">Search Mode</Label>
          <Select 
            value={settings.mode} 
            onValueChange={(val) => setSettings({...settings, mode: val})}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select mode" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="hybrid">Hybrid Search</SelectItem>
              <SelectItem value="dense">Dense Only</SelectItem>
              <SelectItem value="sparse">Sparse Only</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label className="text-base">Top K Limit: {settings.limit}</Label>
          <input 
            type="range" 
            min="1" max="20" 
            value={settings.limit}
            onChange={(e) => setSettings({...settings, limit: parseInt(e.target.value)})}
            className="w-full accent-primary"
          />
        </div>

        <div className="space-y-2">
          <Label className="text-base">Prefetch Limit: {settings.prefetch_limit || 50}</Label>
          <input 
            type="range" 
            min="10" max="150" step="10"
            value={settings.prefetch_limit || 50}
            onChange={(e) => setSettings({...settings, prefetch_limit: parseInt(e.target.value)})}
            className="w-full accent-primary"
          />
        </div>

        <div className="flex items-center justify-between">
          <Label className="text-base">Enable Reranking</Label>
          <Switch 
            checked={settings.rerank} 
            onCheckedChange={(val) => setSettings({...settings, rerank: val})}
          />
        </div>
      </div>

      <Separator />

      <div className="flex flex-col gap-4">
        <h3 className="text-lg font-bold text-primary">Generation Settings</h3>
        
        <div className="space-y-2">
          <Label className="text-base">Strategy</Label>
          <Select 
            value={settings.strategy} 
            onValueChange={(val) => setSettings({...settings, strategy: val})}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select strategy" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto (Self-RAG)</SelectItem>
              <SelectItem value="fast">Fast (Llama 3)</SelectItem>
              <SelectItem value="accurate">Accurate (GPT-4)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2 mt-2">
          <Label className="text-xs text-emerald-500 font-bold">Active Agents Pipeline</Label>
          <div className="bg-muted/30 border border-border rounded-md p-3 text-xs flex flex-col gap-2">
            {[
              { id: 1, name: "Supervisor Agent", active: true },
              { id: 2, name: "Cache Agent", active: true },
              { id: 3, name: "Retrieval (RAG) Agent", active: settings.strategy !== "fast" },
              { id: 4, name: "Generation Agent", active: true },
              { id: 5, name: "Self-Reflection Agent", active: settings.strategy === "auto" },
            ].map(agent => (
              <div key={agent.id} className={`flex items-center gap-2 ${agent.active ? 'text-emerald-400 font-medium' : 'text-muted-foreground opacity-50'}`}>
                <div className={`w-4 h-4 rounded-full flex items-center justify-center text-[9px] ${agent.active ? 'bg-emerald-500/20 border border-emerald-500/50' : 'bg-muted border border-muted-foreground/30'}`}>
                  {agent.id}
                </div>
                <span>{agent.name}</span>
                {agent.active && <span className="ml-auto text-[10px]">🟢</span>}
              </div>
            ))}
            <div className="mt-1 pt-2 border-t border-border/50 text-[10px] text-muted-foreground text-center">
              Total Agents in Pipeline: 5
            </div>
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-4 mt-auto pt-6 pb-2">
        <Link href="/audit" className="w-full">
          <Button variant="ghost" className="w-full justify-start text-left bg-transparent hover:bg-[#2B283A] hover:text-white text-muted-foreground rounded-full px-4 h-11 transition-colors">
            <span className="flex items-center gap-3">
              <span className="text-xl">📊</span>
              <span className="font-medium">Audit Logs</span>
            </span>
          </Button>
        </Link>
      </div>

    </div>
  );
}
