import { useState, useRef, useEffect } from "react";
import { UploadCloud, MessageSquare, FileText, Send, Loader2, Database, Quote } from "lucide-react";
import { cn } from "./lib/utils";

type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: string[];
};

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [docId, setDocId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const selectedFile = e.target.files[0];
    
    if (selectedFile.type !== "application/pdf") {
      alert("Please upload a PDF file.");
      return;
    }

    setFile(selectedFile);
    setIsUploading(true);
    setUploadProgress("Parsing & Embedding...");
    setMessages([]); // Reset chat on new document

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.error || "Failed to upload document");
      }
      
      setDocId(data.docId);
      setMessages([
        { 
          role: "assistant", 
          content: `I've successfully processed "${selectedFile.name}" into ${data.chunks} chunks. What would you like to know about it?` 
        }
      ]);
    } catch (error: any) {
      alert(error.message);
      setFile(null);
    } finally {
      setIsUploading(false);
      setUploadProgress("");
      // Reset input value so same file can be selected again if needed
      e.target.value = '';
    }
  };

  const handleSend = async () => {
    if (!input.trim() || !docId) return;

    const userMsg = input.trim();
    setInput("");
    
    setMessages(prev => [...prev, { role: "user", content: userMsg }]);
    setIsTyping(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message: userMsg,
          docId: docId,
          history: messages.slice(-6).map(m => ({ role: m.role, content: m.content })) // send recent history
        })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to get answer");

      setMessages(prev => [...prev, { 
        role: "assistant", 
        content: data.response,
        citations: data.citations
      }]);
    } catch (error: any) {
      setMessages(prev => [...prev, { 
        role: "assistant", 
        content: `Error: ${error.message}. Please make sure GROQ_API_KEY is configured in the environment.` 
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-screen w-full bg-slate-50 overflow-hidden text-slate-900 font-sans">
      {/* Sidebar sidebar */}
      <div className="w-80 bg-white border-r border-slate-200 flex flex-col items-center py-8 px-6 drop-shadow-sm z-10 shrink-0">
        <div className="flex items-center gap-2 mb-8 text-primary">
          <Database className="w-6 h-6 text-indigo-600" />
          <h1 className="text-xl font-bold tracking-tight text-slate-800">DocuQuery RAG</h1>
        </div>

        <div className="w-full flex flex-col gap-4">
          <div className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-2">
            Knowledge Base
          </div>

          <label className="w-full cursor-pointer relative">
            <input 
              type="file" 
              accept=".pdf" 
              className="hidden" 
              onChange={handleUpload}
              disabled={isUploading}
            />
            <div className={cn(
              "w-full rounded-xl border-2 border-dashed flex flex-col items-center justify-center p-6 gap-3 transition-colors",
              isUploading ? "border-slate-300 bg-slate-50" : "border-indigo-200 bg-indigo-50/50 hover:bg-indigo-50 hover:border-indigo-300"
            )}>
              {isUploading ? (
                <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
              ) : (
                <UploadCloud className="w-8 h-8 text-indigo-500" />
              )}
              <div className="text-sm font-medium text-indigo-900 text-center">
                {isUploading ? uploadProgress : "Upload PDF Document"}
              </div>
              {!isUploading && <div className="text-xs text-indigo-500 text-center">Max 5MB recommended</div>}
            </div>
          </label>

          {file && !isUploading && (
            <div className="flex items-start gap-3 p-4 bg-slate-50 rounded-xl border border-slate-200 mt-2">
              <FileText className="w-5 h-5 text-indigo-500 shrink-0 mt-0.5" />
              <div className="flex flex-col min-w-0">
                <span className="text-sm font-medium text-slate-700 truncate">{file.name}</span>
                <span className="text-xs text-slate-500">{(file.size / 1024 / 1024).toFixed(2)} MB • Ready</span>
              </div>
            </div>
          )}
        </div>
        
        <div className="mt-auto w-full p-4 bg-blue-50/50 border border-blue-100 rounded-xl">
          <div className="text-xs text-blue-800 leading-relaxed">
            <span className="font-semibold block mb-1">Architecture details:</span>
            • Local Embeddings (Xenova BGE)<br/>
            • InMemory VecStore (Cosine)<br/>
            • Groq API (Llama 3.3)
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-full bg-slate-50 relative">
        <div className="flex-1 overflow-y-auto p-4 sm:p-8 flex flex-col gap-6">
          {!docId && messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center max-w-md mx-auto space-y-4">
              <div className="w-16 h-16 bg-white shadow-sm rounded-2xl flex items-center justify-center mb-2">
                <MessageSquare className="w-8 h-8 text-indigo-300" />
              </div>
              <h2 className="text-2xl font-semibold tracking-tight text-slate-700">Waiting for Data</h2>
              <p className="text-slate-500 leading-relaxed">
                Upload a PDF document from the sidebar to initialize the retrieval-augmented generation pipeline.
              </p>
            </div>
          ) : null}

          {messages.map((msg, idx) => (
            <div 
              key={idx} 
              className={cn(
                "flex max-w-[85%] mx-auto w-full",
                msg.role === "user" ? "justify-end" : "justify-start"
              )}
            >
              <div className={cn(
                "rounded-2xl px-6 py-4 shadow-sm relative overflow-hidden",
                msg.role === "user" 
                  ? "bg-indigo-600 text-white rounded-br-sm" 
                  : "bg-white border border-slate-200 text-slate-800 rounded-bl-sm"
              )}>
                <div className="leading-relaxed whitespace-pre-wrap">{msg.content}</div>
                
                {msg.citations && msg.citations.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-slate-100 flex flex-col gap-2">
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1">
                      <Quote className="w-3 h-3" /> Sources cited:
                    </span>
                    {msg.citations.map((cite, i) => (
                      <div key={i} className="text-xs text-slate-500 bg-slate-50 p-2 rounded-md italic border-l-2 border-indigo-200">
                        "{cite.length > 150 ? cite.slice(0, 150) + '...' : cite}"
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {isTyping && (
            <div className="flex max-w-[85%] mx-auto w-full justify-start">
              <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-sm px-6 py-5 shadow-sm">
                <div className="flex gap-1.5 items-center">
                  <div className="w-2 h-2 rounded-full bg-slate-300 animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 rounded-full bg-slate-300 animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 rounded-full bg-slate-300 animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} className="h-4" />
        </div>

        {/* Input Area */}
        <div className="p-4 sm:p-6 bg-slate-50">
          <div className="max-w-[85%] mx-auto relative flex items-center">
            <input
              type="text"
              className="w-full bg-white border border-slate-300 rounded-full pl-6 pr-14 py-4 text-base focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent shadow-sm disabled:bg-slate-50 disabled:cursor-not-allowed transition-all"
              placeholder={docId ? "Ask a question about the document..." : "Upload a document first..."}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={!docId || isTyping}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || !docId || isTyping}
              className="absolute right-2 p-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white rounded-full transition-colors flex items-center justify-center shrink-0"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
