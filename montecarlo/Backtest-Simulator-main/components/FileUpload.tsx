import React, { useRef } from 'react';
import { UploadCloud } from 'lucide-react';

interface FileUploadProps {
  onFileUpload: (content: string) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result;
      if (typeof text === 'string') {
        onFileUpload(text);
      }
    };
    reader.readAsText(file);
  };

  return (
    <div 
      className="border-2 border-dashed border-neutral-600 rounded-lg p-8 flex flex-col items-center justify-center cursor-pointer hover:border-primary hover:bg-neutral-800/50 transition-colors group"
      onClick={() => fileInputRef.current?.click()}
    >
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        accept=".csv" 
        className="hidden" 
      />
      <div className="bg-neutral-700 p-3 rounded-full mb-3 group-hover:bg-primary/20 group-hover:text-primary transition-colors">
        <UploadCloud className="w-8 h-8 text-neutral-300 group-hover:text-primary" />
      </div>
      <p className="text-neutral-300 font-medium">Click to upload TradingView Trade List CSV</p>
      <p className="text-neutral-500 text-sm mt-1">Supports standard CSV exports</p>
    </div>
  );
};

export default FileUpload;