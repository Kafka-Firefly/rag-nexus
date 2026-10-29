import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Nexus | 文档知识库",
  description: "基于来源可追溯的文档问答知识库",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body>{children}</body></html>;
}
