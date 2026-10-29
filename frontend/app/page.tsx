"use client";

import { FormEvent, useRef, useState } from "react";
import { FileText, LoaderCircle, Send, Upload, UserRound, Bot, X } from "lucide-react";

type Citation = { index: number; source: string; chunk_id: string; page?: number | null; text: string; score: number };
type Message = { role: "user" | "assistant"; content: string; citations?: Citation[] };
const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  async function uploadFile(file?: File) {
    if (!file) return;
    setUploading(true); setNotice("");
    const body = new FormData(); body.append("file", file);
    try {
      const response = await fetch(`${API}/upload`, { method: "POST", body });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail ?? "上传失败");
      setNotice(`已索引 ${result.source}，共 ${result.chunks} 个文本块`);
    } catch (error) { setNotice(error instanceof Error ? error.message : "上传失败"); }
    finally { setUploading(false); if (inputRef.current) inputRef.current.value = ""; }
  }

  async function send(event: FormEvent) {
    event.preventDefault();
    const prompt = question.trim();
    if (!prompt || busy) return;
    setQuestion(""); setNotice(""); setBusy(true);
    setMessages((old) => [...old, { role: "user", content: prompt }, { role: "assistant", content: "" }]);
    try {
      const response = await fetch(`${API}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: prompt }) });
      if (!response.ok || !response.body) {
        const err = await response.json().catch(() => ({})); throw new Error(err.detail ?? "无法连接问答服务");
      }
      const reader = response.body.getReader(); const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read(); if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n"); buffer = events.pop() ?? "";
        for (const raw of events) {
          const lines = raw.split("\n"); const name = lines.find((line) => line.startsWith("event:"))?.slice(6).trim();
          const data = lines.find((line) => line.startsWith("data:"))?.slice(5).trim(); if (!data) continue;
          const parsed = JSON.parse(data);
          if (name === "citations") setMessages((old) => old.map((m, i) => i === old.length - 1 ? { ...m, citations: parsed } : m));
          else if (name === "error") throw new Error(parsed.detail);
          else if (!name && parsed.token) setMessages((old) => old.map((m, i) => i === old.length - 1 ? { ...m, content: m.content + parsed.token } : m));
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "发生未知错误";
      setMessages((old) => old.map((m, i) => i === old.length - 1 ? { ...m, content: `抱歉，${message}` } : m));
    } finally { setBusy(false); }
  }

  return <main className="flex min-h-screen flex-col bg-[#f7f8fa] text-slate-900">
    <header className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-4">
      <div><h1 className="text-lg font-semibold tracking-tight">RAG Nexus</h1><p className="text-xs text-slate-500">文档知识库 · 有据可查的回答</p></div>
      <div>
        <input ref={inputRef} type="file" accept=".pdf,.txt" className="hidden" onChange={(e) => uploadFile(e.target.files?.[0])} />
        <button onClick={() => inputRef.current?.click()} disabled={uploading} className="flex items-center gap-2 rounded-lg border border-slate-300 px-3 py-2 text-sm hover:bg-slate-50 disabled:opacity-50">
          {uploading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />} 添加文档
        </button>
      </div>
    </header>
    {notice && <div className="mx-auto mt-4 flex max-w-3xl items-center justify-between rounded-lg bg-blue-50 px-4 py-2 text-sm text-blue-800">{notice}<button onClick={() => setNotice("")} aria-label="关闭"><X className="h-4 w-4" /></button></div>}
    <section className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 px-5 py-8">
      {messages.length === 0 ? <div className="m-auto max-w-lg text-center"><div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-indigo-600 text-white"><FileText /></div><h2 className="text-2xl font-semibold">向你的文档提问</h2><p className="mt-2 text-sm leading-6 text-slate-500">上传 PDF 或 TXT 文件。回答会基于检索到的资料，并附带可核对的来源。</p></div> : messages.map((message, index) =>
        <article key={index} className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}>
          {message.role === "assistant" && <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-indigo-700"><Bot className="h-4 w-4" /></div>}
          <div className={`max-w-[88%] rounded-2xl px-4 py-3 ${message.role === "user" ? "bg-indigo-600 text-white" : "border border-slate-200 bg-white shadow-sm"}`}>
            <div className="whitespace-pre-wrap text-sm leading-6">{message.content || (busy && index === messages.length - 1 ? <LoaderCircle className="h-4 w-4 animate-spin" /> : "")}</div>
            {message.citations && message.citations.length > 0 && <div className="mt-4 border-t border-slate-100 pt-3"><p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">来源依据</p><div className="flex flex-wrap gap-2">{message.citations.map((citation) => <a key={citation.chunk_id} href={`#source-${citation.chunk_id}`} title={citation.text} className="rounded-full border border-indigo-200 bg-indigo-50 px-2.5 py-1 text-xs text-indigo-800 hover:bg-indigo-100">[{citation.index}] {citation.source}{citation.page ? ` · 第 ${citation.page} 页` : ""}</a>)}</div><div className="mt-2 space-y-2">{message.citations.map((citation) => <details id={`source-${citation.chunk_id}`} key={citation.chunk_id} className="rounded-md bg-slate-50 px-3 py-2 text-xs"><summary className="cursor-pointer text-slate-600">块 {citation.chunk_id.slice(0, 8)} · 查看原文片段</summary><p className="mt-2 whitespace-pre-wrap leading-5 text-slate-700">{citation.text}</p></details>)}</div></div>}
          </div>
          {message.role === "user" && <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-200"><UserRound className="h-4 w-4" /></div>}
        </article>)}
    </section>
    <form onSubmit={send} className="sticky bottom-0 mx-auto w-full max-w-3xl bg-[#f7f8fa] px-5 pb-5 pt-2"><div className="flex items-end gap-2 rounded-2xl border border-slate-300 bg-white p-2 shadow-sm focus-within:border-indigo-400"><textarea value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); e.currentTarget.form?.requestSubmit(); } }} placeholder="输入问题…（Enter 发送，Shift+Enter 换行）" rows={1} className="max-h-36 min-h-10 flex-1 resize-y px-2 py-2 text-sm outline-none" /><button disabled={busy || !question.trim()} aria-label="发送问题" className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40"><Send className="h-4 w-4" /></button></div><p className="mt-2 text-center text-[11px] text-slate-400">回答由知识库检索生成，请通过来源片段核对重要信息。</p></form>
  </main>;
}
