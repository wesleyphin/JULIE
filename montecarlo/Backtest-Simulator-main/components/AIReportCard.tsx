import React, { useState, useRef, useEffect } from 'react';
import { Sparkles, BrainCircuit, RefreshCcw, AlertTriangle, FileCode, Upload, ArrowRight, Code, FilePlus, FileSpreadsheet, Trash2, X, Copy, Check, Cpu, Send, MessageSquare } from 'lucide-react';
import { generateStrategyReport, createOptimizerChat, buildOptimizerPrompt } from '../utils/aiLogic';
import { performRegressionAnalysis } from '../utils/analytics';
import { Statistics, Trade } from '../types';
import { Chat } from "@google/genai";

// Helper component for Code Blocks with Copy functionality
const CodeBlock: React.FC<{ language: string, code: string }> = ({ language, code }) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    };

    return (
        <div className="mb-4 rounded-lg overflow-hidden border border-neutral-700 bg-[#1e1e1e] group">
            <div className="flex items-center justify-between px-4 py-2 bg-neutral-800 border-b border-neutral-700">
                <span className="text-[10px] text-neutral-400 uppercase font-bold tracking-wider">{language || 'CODE'}</span>
                <button 
                    onClick={handleCopy}
                    className={`flex items-center gap-1.5 text-[10px] font-medium transition-all px-2 py-1 rounded border ${
                        copied 
                        ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20' 
                        : 'bg-neutral-800 text-neutral-400 border-neutral-700 hover:bg-neutral-700 hover:text-white'
                    }`}
                >
                    {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                    {copied ? 'Copied!' : 'Copy Code'}
                </button>
            </div>
            <pre className="p-4 overflow-x-auto text-sm font-mono text-emerald-300 leading-relaxed custom-scrollbar max-h-[500px]">
                {code}
            </pre>
        </div>
    );
};

// Robust internal Markdown Renderer to avoid dependency issues
const MarkdownRenderer = ({ content }: { content: string }) => {
  if (!content) return null;

  try {
      // Split by newlines to handle block elements
      const lines = content.split('\n');
      const elements: React.ReactNode[] = [];
      let listBuffer: React.ReactNode[] = [];
      let codeBlockBuffer: string[] = [];
      let inCodeBlock = false;
      let codeLanguage = '';

      const processInline = (text: string) => {
        // Handle Bold (**text**)
        const parts = text.split(/(\*\*.*?\*\*)/g);
        return parts.map((part, index) => {
          if (part.startsWith('**') && part.endsWith('**')) {
            return <strong key={index} className="text-white font-bold">{part.slice(2, -2)}</strong>;
          }
          return part;
        });
      };

      const flushList = () => {
        if (listBuffer.length > 0) {
          elements.push(
            <ul key={`list-${elements.length}`} className="list-disc pl-5 space-y-1 mb-4 text-neutral-300">
              {[...listBuffer]}
            </ul>
          );
          listBuffer = [];
        }
      };

      const flushCode = () => {
          if (codeBlockBuffer.length > 0) {
              elements.push(
                  <CodeBlock 
                    key={`code-${elements.length}`} 
                    language={codeLanguage} 
                    code={codeBlockBuffer.join('\n')} 
                  />
              );
              codeBlockBuffer = [];
              codeLanguage = '';
          }
      }

      for (let index = 0; index < lines.length; index++) {
        const line = lines[index];
        const trimmed = line.trim();

        // Code Block Handling
        if (trimmed.startsWith('```')) {
            if (inCodeBlock) {
                // End of block
                inCodeBlock = false;
                flushCode();
            } else {
                // Start of block
                flushList(); // Flush any pending lists
                inCodeBlock = true;
                codeLanguage = trimmed.slice(3).trim();
            }
            continue;
        }

        if (inCodeBlock) {
            codeBlockBuffer.push(line);
            continue;
        }

        // Headers
        if (trimmed.startsWith('### ')) {
          flushList();
          elements.push(<h4 key={index} className="text-md font-bold text-primary mt-6 mb-2 uppercase tracking-wide border-b border-neutral-800 pb-1">{trimmed.slice(4)}</h4>);
        } else if (trimmed.startsWith('## ')) {
          flushList();
          elements.push(<h3 key={index} className="text-lg font-bold text-white mt-6 mb-2">{trimmed.slice(3)}</h3>);
        } 
        // List Items
        else if (trimmed.startsWith('* ') || trimmed.startsWith('- ')) {
          listBuffer.push(<li key={`${index}-li`}>{processInline(trimmed.slice(2))}</li>);
        } 
        // Empty Lines
        else if (trimmed === '') {
          flushList();
          elements.push(<div key={index} className="h-2"></div>);
        } 
        // Paragraphs
        else {
          flushList();
          elements.push(<p key={index} className="text-neutral-300 text-sm leading-relaxed mb-1">{processInline(trimmed)}</p>);
        }
      }

      flushList(); 
      flushCode();

      return <div className="space-y-1">{elements}</div>;
  } catch (e) {
      return <div className="text-rose-500 p-4 border border-rose-900 rounded bg-rose-950/20">Error rendering markdown content.</div>;
  }
};

// Specialized Loading Graphic Component
const LoadingGraphic = ({ message, type }: { message: string, type: 'report' | 'optimize' }) => (
    <div className="absolute inset-0 bg-neutral-950/95 backdrop-blur-md z-50 flex flex-col items-center justify-center animate-in fade-in duration-300">
        <div className="relative w-32 h-32 mb-8">
            {/* Outer Rotating Ring */}
            <div className={`absolute inset-0 border-t-2 border-r-2 ${type === 'report' ? 'border-purple-500' : 'border-emerald-500'} rounded-full animate-spin duration-[3s]`}></div>
            {/* Inner Rotating Ring (Counter) */}
            <div className={`absolute inset-4 border-b-2 border-l-2 ${type === 'report' ? 'border-purple-800' : 'border-emerald-800'} rounded-full animate-spin duration-[2s] direction-reverse`}></div>
            
            {/* Pulsing Core */}
            <div className="absolute inset-0 flex items-center justify-center">
                <div className={`relative flex items-center justify-center w-16 h-16 rounded-full bg-neutral-900 border ${type === 'report' ? 'border-purple-500/30' : 'border-emerald-500/30'} shadow-[0_0_30px_rgba(0,0,0,0.5)]`}>
                    <Cpu className={`w-8 h-8 ${type === 'report' ? 'text-purple-400' : 'text-emerald-400'} animate-pulse`} />
                    <div className={`absolute inset-0 rounded-full ${type === 'report' ? 'bg-purple-500/20' : 'bg-emerald-500/20'} animate-ping`}></div>
                </div>
            </div>
            
            {/* Orbiting Particles */}
            <div className="absolute inset-0 animate-spin duration-[5s]">
                 <div className={`absolute top-0 left-1/2 -translate-x-1/2 w-2 h-2 ${type === 'report' ? 'bg-purple-400' : 'bg-emerald-400'} rounded-full shadow-[0_0_10px_currentColor]`}></div>
            </div>
        </div>
        
        <h3 className="text-xl font-bold text-white mb-2 animate-pulse tracking-wide">{message}</h3>
        <div className="flex items-center gap-2 text-neutral-500 text-xs uppercase tracking-widest font-mono">
            <span className="w-2 h-2 bg-neutral-600 rounded-full animate-bounce delay-75"></span>
            <span className="w-2 h-2 bg-neutral-600 rounded-full animate-bounce delay-150"></span>
            <span className="w-2 h-2 bg-neutral-600 rounded-full animate-bounce delay-300"></span>
        </div>
    </div>
);

interface Props {
  trades: Trade[];
  historicalStats: any;
  mcStats: Statistics | null;
  mcConfig: any;
}

interface UploadedFile {
    name: string;
    content: string;
}

interface ChatMessage {
    role: 'user' | 'model';
    content: string;
}

const AIReportCard: React.FC<Props> = ({ trades, historicalStats, mcStats, mcConfig }) => {
  const [activeTab, setActiveTab] = useState<'report' | 'optimize'>('report');
  
  // Report State
  const [report, setReport] = useState<string | null>(null);
  
  // Optimizer State
  const [strategyFiles, setStrategyFiles] = useState<UploadedFile[]>([]);
  const [marketDataFiles, setMarketDataFiles] = useState<UploadedFile[]>([]);
  
  // Chat State
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const chatSessionRef = useRef<Chat | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const strategyInputRef = useRef<HTMLInputElement>(null);
  const dataInputRef = useRef<HTMLInputElement>(null);

  // Shared State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  // Check if we have results to determine expanded state
  const hasResults = (activeTab === 'report' && report !== null) || (activeTab === 'optimize' && messages.length > 0);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, loading]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
        setLoading(false);
    };
  }, []);

  const handleGenerateReport = async () => {
    setLoading(true);
    setError(false);
    // Use timeout to allow UI to update to loading state before heavy async work
    setTimeout(async () => {
        try {
            const text = await generateStrategyReport(historicalStats, mcStats, mcConfig);
            setReport(text);
        } catch (e) {
            console.error(e);
            setError(true);
        } finally {
            setLoading(false);
        }
    }, 50);
  };

  const handleStrategyUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (!files) return;

      Array.from(files).forEach((file: File) => {
          if (file.size > 1024 * 1024) {
             alert(`File ${file.name} is too large (>1MB). Skipping.`);
             return;
          }
          const reader = new FileReader();
          reader.onload = (ev) => {
              const content = ev.target?.result as string;
              if (content) {
                  setStrategyFiles(prev => [...prev, { name: file.name, content }]);
              }
          };
          reader.readAsText(file);
      });
      e.target.value = '';
  };

  const removeStrategyFile = (index: number) => {
      setStrategyFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleDataUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (!files) return;

      Array.from(files).forEach((file: File) => {
          const reader = new FileReader();
          reader.onload = (ev) => {
              const content = ev.target?.result as string;
              if (content) {
                  const lines = content.split('\n');
                  const snippet = lines.slice(0, 50).join('\n');
                  setMarketDataFiles(prev => [...prev, { name: file.name, content: snippet }]);
              }
          };
          const slice = file.slice(0, 1024 * 50); 
          reader.readAsText(slice);
      });
      e.target.value = '';
  };

  const removeDataFile = (index: number) => {
      setMarketDataFiles(prev => prev.filter((_, i) => i !== index));
  };

  const startOptimizerChat = async () => {
    if (strategyFiles.length === 0) return;
    setLoading(true);
    setError(false);
    
    // Create new chat session
    chatSessionRef.current = createOptimizerChat();
    
    setTimeout(async () => {
        try {
            const regression = performRegressionAnalysis(trades);
            const regressionSummary = `
                Top Correlations (All Trades):
                ${regression.all.slice(0, 3).map(r => `- ${r.feature}: r=${r.correlation.toFixed(3)}`).join('\n')}
                
                Long Correlations:
                ${regression.long.slice(0, 2).map(r => `- ${r.feature}: r=${r.correlation.toFixed(3)}`).join('\n')}

                Short Correlations:
                ${regression.short.slice(0, 2).map(r => `- ${r.feature}: r=${r.correlation.toFixed(3)}`).join('\n')}
            `;

            const prompt = buildOptimizerPrompt(strategyFiles, marketDataFiles, historicalStats, mcStats, regressionSummary);
            
            // Send initial big prompt
            const result = await chatSessionRef.current!.sendMessage({ message: prompt });
            
            // Add ONLY the result to messages to avoid clogging chat with massive prompt
            setMessages([{ role: 'model', content: result.text || 'No response generated.' }]);

        } catch (e) {
            console.error("Optimization failed:", e);
            setError(true);
        } finally {
            setLoading(false);
        }
    }, 100);
  };

  const handleSendMessage = async () => {
      if (!input.trim() || !chatSessionRef.current) return;
      
      const userMsg = input;
      setInput('');
      setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
      setLoading(true);

      try {
          const result = await chatSessionRef.current!.sendMessage({ message: userMsg });
          setMessages(prev => [...prev, { role: 'model', content: result.text || 'No response.' }]);
      } catch (e) {
          console.error("Chat error:", e);
          setMessages(prev => [...prev, { role: 'model', content: "**Error:** Failed to get response. Please try again." }]);
      } finally {
          setLoading(false);
      }
  };

  const resetOptimizer = () => {
      setStrategyFiles([]);
      setMarketDataFiles([]);
      setMessages([]);
      setError(false);
      setLoading(false);
      chatSessionRef.current = null;
  };

  return (
    <div 
        className={`bg-surface rounded-xl border border-neutral-800 shadow-xl overflow-hidden flex flex-col relative transition-[min-height] duration-500 ease-in-out ${hasResults ? 'min-h-[800px]' : 'min-h-[350px]'}`}
    >
      
      {/* Header & Tabs */}
      <div className="border-b border-neutral-800 bg-neutral-900/50 backdrop-blur-sm z-20">
        <div className="p-4 flex justify-between items-center">
            <div className="flex items-center gap-2">
                <BrainCircuit className="w-5 h-5 text-purple-400" />
                <h3 className="font-semibold text-neutral-200">AI Quant Researcher <span className="text-xs text-neutral-600 ml-2 font-normal border border-neutral-800 px-1 rounded">Gemini 3 Pro</span></h3>
            </div>
            
            <div className="flex gap-1 bg-neutral-900 p-1 rounded-lg border border-neutral-800">
                <button 
                    onClick={() => setActiveTab('report')}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all flex items-center gap-2 ${activeTab === 'report' ? 'bg-purple-600 text-white shadow-md' : 'text-neutral-400 hover:text-white'}`}
                >
                    <FileCode className="w-3 h-3" /> Risk Report
                </button>
                <button 
                    onClick={() => setActiveTab('optimize')}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all flex items-center gap-2 ${activeTab === 'optimize' ? 'bg-emerald-600 text-white shadow-md' : 'text-neutral-400 hover:text-white'}`}
                >
                    <Code className="w-3 h-3" /> Optimizer
                </button>
            </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-hidden relative bg-neutral-950/30 flex flex-col">
        
        {/* === TAB 1: RISK REPORT === */}
        {activeTab === 'report' && (
            <div className="p-6 h-full overflow-y-auto custom-scrollbar">
                 {!report && !loading && (
                    <div className="h-[250px] flex flex-col items-center justify-center text-neutral-500 space-y-4 opacity-60">
                        <div className="p-4 bg-neutral-900/50 rounded-full border border-neutral-800 group hover:border-purple-500/50 transition-colors">
                            <BrainCircuit className="w-10 h-10 group-hover:text-purple-400 transition-colors" />
                        </div>
                        <div className="text-center space-y-2">
                            <p className="text-sm font-medium text-neutral-300">Strategy Intelligence</p>
                            <p className="text-xs max-w-xs mx-auto">
                                Generates a comprehensive risk report identifying curve fitting, tail risks, and prop firm viability.
                            </p>
                            <button 
                                onClick={handleGenerateReport}
                                className="mt-4 flex items-center gap-2 px-6 py-2 bg-neutral-800 hover:bg-purple-600 hover:text-white text-neutral-300 text-xs font-bold rounded-lg transition-all mx-auto border border-neutral-700"
                            >
                                <Sparkles className="w-3 h-3" /> Generate Report
                            </button>
                        </div>
                    </div>
                )}

                {report && (
                     <div className="animate-in fade-in duration-500 pb-4">
                        <div className="flex justify-end mb-4">
                            <button onClick={handleGenerateReport} className="text-neutral-500 hover:text-white transition-colors p-2 hover:bg-neutral-800 rounded-lg flex items-center gap-2 text-xs">
                                <RefreshCcw className="w-3 h-3" /> Refresh
                            </button>
                        </div>
                        <MarkdownRenderer content={report} />
                    </div>
                )}
            </div>
        )}

        {/* === TAB 2: OPTIMIZER === */}
        {activeTab === 'optimize' && (
            <div className="flex flex-col h-full overflow-hidden">
                
                {/* Upload Section: Only show if no messages started */}
                {messages.length === 0 && !loading && (
                    <div className="p-6 overflow-y-auto custom-scrollbar">
                         <div className="space-y-6">
                            {/* 1. Strategy Files */}
                            <div className="space-y-3">
                                <div className="flex items-center justify-between">
                                    <h4 className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
                                        <Code className="w-4 h-4 text-emerald-500" /> Strategy Codebase
                                    </h4>
                                    <div className="flex gap-2">
                                        {strategyFiles.length > 0 && (
                                            <button 
                                                onClick={() => setStrategyFiles([])}
                                                className="text-xs text-neutral-500 hover:text-rose-400 px-3 py-1.5"
                                            >
                                                Clear All
                                            </button>
                                        )}
                                        <button 
                                            onClick={() => strategyInputRef.current?.click()}
                                            className="text-xs bg-neutral-800 hover:bg-neutral-700 text-white px-3 py-1.5 rounded border border-neutral-700 flex items-center gap-1 transition-colors"
                                        >
                                            <FilePlus className="w-3 h-3" /> Add Files
                                        </button>
                                    </div>
                                    <input 
                                        type="file" 
                                        multiple 
                                        ref={strategyInputRef} 
                                        onChange={handleStrategyUpload} 
                                        className="hidden" 
                                        accept=".txt,.pine,.py,.cs,.cpp" 
                                    />
                                </div>

                                {/* Strategy File List */}
                                <div className="bg-neutral-900/40 border border-neutral-800 rounded-lg p-4 min-h-[80px]">
                                    {strategyFiles.length === 0 ? (
                                        <div className="flex flex-col items-center justify-center h-20 text-neutral-600 text-xs border border-dashed border-neutral-800 rounded">
                                            <p>No strategy files loaded.</p>
                                            <p>Upload .py, .pine, .cpp files.</p>
                                        </div>
                                    ) : (
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                                            {strategyFiles.map((file, idx) => (
                                                <div key={idx} className="flex items-center justify-between bg-neutral-800/50 p-2 rounded border border-neutral-700 group">
                                                    <div className="flex items-center gap-2 overflow-hidden">
                                                        <FileCode className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                                                        <span className="text-xs text-neutral-300 truncate">{file.name}</span>
                                                        <span className="text-[10px] text-neutral-500">({(file.content.length / 1024).toFixed(1)} KB)</span>
                                                    </div>
                                                    <button onClick={() => removeStrategyFile(idx)} className="text-neutral-500 hover:text-rose-400 p-1 rounded hover:bg-neutral-900 transition-colors">
                                                        <Trash2 className="w-3 h-3" />
                                                    </button>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* 2. Market Data (Optional) */}
                            <div className="space-y-3">
                                 <div className="flex items-center justify-between">
                                    <h4 className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
                                        <FileSpreadsheet className="w-4 h-4 text-amber-500" /> Market Data <span className="text-[10px] text-neutral-500 font-normal ml-1">(Optional)</span>
                                    </h4>
                                    <div className="flex gap-2">
                                         {marketDataFiles.length > 0 && (
                                            <button 
                                                onClick={() => setMarketDataFiles([])}
                                                className="text-xs text-neutral-500 hover:text-rose-400 px-3 py-1.5"
                                            >
                                                Clear All
                                            </button>
                                         )}
                                        <button 
                                            onClick={() => dataInputRef.current?.click()}
                                            className="text-xs bg-neutral-800 hover:bg-neutral-700 text-white px-3 py-1.5 rounded border border-neutral-700 flex items-center gap-1 transition-colors"
                                        >
                                            <Upload className="w-3 h-3" /> Upload CSVs
                                        </button>
                                    </div>
                                    <input 
                                        type="file" 
                                        multiple
                                        ref={dataInputRef} 
                                        onChange={handleDataUpload} 
                                        className="hidden" 
                                        accept=".csv" 
                                    />
                                 </div>
                                 
                                 {/* Data File List */}
                                 <div className="bg-neutral-900/40 border border-neutral-800 rounded-lg p-4 min-h-[80px]">
                                     {marketDataFiles.length === 0 ? (
                                        <div 
                                            onClick={() => dataInputRef.current?.click()}
                                            className="flex flex-col items-center justify-center h-20 text-neutral-600 text-xs border border-dashed border-neutral-800 rounded cursor-pointer hover:bg-neutral-800/30 transition-colors"
                                        >
                                            <Upload className="w-4 h-4 mb-2 text-neutral-500" />
                                            <p>No market data loaded.</p>
                                            <p>Upload CSV files for AI context.</p>
                                        </div>
                                     ) : (
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                                            {marketDataFiles.map((file, idx) => (
                                                <div key={idx} className="flex items-center justify-between bg-neutral-800/50 p-2 rounded border border-neutral-700 group">
                                                    <div className="flex items-center gap-2 overflow-hidden">
                                                        <FileSpreadsheet className="w-4 h-4 text-amber-500 flex-shrink-0" />
                                                        <span className="text-xs text-neutral-300 truncate">{file.name}</span>
                                                        <span className="text-[10px] text-neutral-500">(Snippet Loaded)</span>
                                                    </div>
                                                    <button onClick={() => removeDataFile(idx)} className="text-neutral-500 hover:text-rose-400 p-1 rounded hover:bg-neutral-900 transition-colors">
                                                        <Trash2 className="w-3 h-3" />
                                                    </button>
                                                </div>
                                            ))}
                                        </div>
                                     )}
                                 </div>
                            </div>

                            <div className="flex justify-end pt-4 border-t border-neutral-800">
                                <button 
                                    onClick={startOptimizerChat}
                                    disabled={strategyFiles.length === 0}
                                    className={`flex items-center gap-2 px-6 py-3 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold rounded-lg transition-all shadow-lg shadow-emerald-900/20 ${strategyFiles.length === 0 ? 'opacity-50 cursor-not-allowed' : ''}`}
                                >
                                    <Sparkles className="w-4 h-4" /> Run Optimizer
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {/* Chat Interface */}
                {messages.length > 0 && (
                     <div className="flex flex-col flex-1 min-h-0 bg-neutral-950/20">
                        {/* Header Actions */}
                        <div className="flex justify-between items-center p-3 border-b border-neutral-800 bg-neutral-900/50">
                            <h3 className="text-sm font-bold text-white flex items-center gap-2">
                                <MessageSquare className="w-4 h-4 text-emerald-500" /> Optimization Session
                            </h3>
                            <div className="flex gap-2">
                                <button onClick={resetOptimizer} className="text-xs text-neutral-400 hover:text-white px-3 py-1.5 hover:bg-neutral-800 rounded">
                                    New Session
                                </button>
                            </div>
                        </div>

                        {/* Messages List */}
                        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-6">
                            {messages.map((msg, i) => (
                                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    <div 
                                        className={`max-w-[85%] rounded-2xl p-4 shadow-sm ${
                                            msg.role === 'user' 
                                            ? 'bg-neutral-800 text-neutral-100 rounded-tr-none' 
                                            : 'bg-neutral-900/80 border border-neutral-800 rounded-tl-none'
                                        }`}
                                    >
                                        {msg.role === 'model' && (
                                            <div className="flex items-center gap-2 mb-3 pb-2 border-b border-neutral-800/50">
                                                <BrainCircuit className="w-4 h-4 text-emerald-500" />
                                                <span className="text-xs font-bold text-emerald-500">AI Researcher</span>
                                            </div>
                                        )}
                                        <div className={`text-sm leading-relaxed ${msg.role === 'user' ? 'whitespace-pre-wrap' : ''}`}>
                                            {msg.role === 'user' ? msg.content : <MarkdownRenderer content={msg.content} />}
                                        </div>
                                    </div>
                                </div>
                            ))}
                            <div ref={messagesEndRef} />
                        </div>

                        {/* Input Area */}
                        <div className="p-4 bg-surface border-t border-neutral-800">
                            <div className="relative">
                                <input
                                    type="text"
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && !loading && handleSendMessage()}
                                    placeholder="Ask a follow-up question or report an error to fix..."
                                    className="w-full bg-neutral-900 border border-neutral-700 rounded-lg pl-4 pr-12 py-3 text-sm focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500/50 transition-all shadow-inner"
                                    disabled={loading}
                                />
                                <button 
                                    onClick={handleSendMessage}
                                    disabled={loading || !input.trim()}
                                    className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                                >
                                    <Send className="w-3 h-3" />
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        )}

        {/* === SHARED LOADING GRAPHIC === */}
        {loading && (
            <LoadingGraphic 
                message={messages.length > 0 ? "Analyzing Request..." : "AI Quant Researcher is working..."} 
                type={activeTab} 
            />
        )}

        {/* === SHARED ERROR STATE === */}
        {error && (
            <div className="absolute inset-0 bg-neutral-950 z-50 flex flex-col items-center justify-center text-rose-500">
                <AlertTriangle className="w-10 h-10 mb-2" />
                <p className="font-medium">AI Analysis Failed</p>
                <p className="text-xs text-neutral-500 mt-2">Check console for details.</p>
                <button onClick={() => setError(false)} className="mt-4 text-xs bg-neutral-800 px-4 py-2 rounded text-white">Dismiss</button>
            </div>
        )}

      </div>
    </div>
  );
};

export default AIReportCard;