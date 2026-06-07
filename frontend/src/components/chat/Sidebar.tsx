"use client";

import React, { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { useTheme } from "next-themes";
import { Moon, Sun, MessageSquarePlus, X } from "lucide-react";

export default function Sidebar({ settings, setSettings, onNewChat, threads, currentThread, onSelectThread, onSuggestionClick, closeSidebar }: any) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="w-80 border-r bg-background/95 backdrop-blur h-screen overflow-y-auto p-4 flex flex-col gap-6 absolute md:relative z-50 shadow-2xl md:shadow-none">
      <div className="flex justify-between items-center">
        <h3 className="text-sm font-semibold text-muted-foreground">Menu & Settings</h3>
        <div className="flex gap-1">
          {mounted && (
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            title="Toggle theme"
          >
            {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
          </Button>
        )}
        <Button variant="ghost" size="icon" onClick={closeSidebar} className="md:hidden" title="Close Sidebar">
          <X size={18} />
        </Button>
        </div>
      </div>

      <Button onClick={onNewChat} className="w-full flex items-center gap-2" variant="default">
        <MessageSquarePlus size={16} /> New Chat
      </Button>

      {threads && threads.length > 0 && (
        <div className="flex flex-col gap-2">
          <Label className="text-xs">Saved Threads</Label>
          <Select 
            value={currentThread || ""} 
            onValueChange={onSelectThread}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select a past chat..." />
            </SelectTrigger>
            <SelectContent>
              {threads.map((t: string) => (
                <SelectItem key={t} value={t}>Thread {t}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      <Separator />

      <div className="flex flex-col gap-4">
        <h3 className="text-sm font-semibold text-primary">Suggestions</h3>
        <div className="flex flex-col gap-2">
          {["What was meta most profitable year?", "Summarize Apple's gross margin in 2024", "How did Microsoft perform in Q1?"].map((q) => (
            <Button 
              key={q} 
              variant="outline" 
              className="w-full justify-start text-left h-auto py-2 px-3 text-xs whitespace-normal"
              onClick={() => onSuggestionClick(q)}
            >
              {q}
            </Button>
          ))}
        </div>
      </div>

      <Separator />

      <div className="flex flex-col gap-4">
        <h3 className="text-sm font-semibold text-primary">Retrieval Settings</h3>
        
        <div className="space-y-2">
          <Label className="text-xs">Search Mode</Label>
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
          <Label className="text-xs">Top K Limit: {settings.limit}</Label>
          <input 
            type="range" 
            min="1" max="20" 
            value={settings.limit}
            onChange={(e) => setSettings({...settings, limit: parseInt(e.target.value)})}
            className="w-full accent-primary"
          />
        </div>

        <div className="space-y-2">
          <Label className="text-xs">Prefetch Limit: {settings.prefetch_limit || 50}</Label>
          <input 
            type="range" 
            min="10" max="150" step="10"
            value={settings.prefetch_limit || 50}
            onChange={(e) => setSettings({...settings, prefetch_limit: parseInt(e.target.value)})}
            className="w-full accent-primary"
          />
        </div>

        <div className="flex items-center justify-between">
          <Label className="text-xs">Enable Reranking</Label>
          <Switch 
            checked={settings.rerank} 
            onCheckedChange={(val) => setSettings({...settings, rerank: val})}
          />
        </div>
      </div>

      <Separator />

      <div className="flex flex-col gap-4">
        <h3 className="text-sm font-semibold text-primary">Generation Settings</h3>
        
        <div className="space-y-2">
          <Label className="text-xs">Strategy</Label>
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

    </div>
  );
}
