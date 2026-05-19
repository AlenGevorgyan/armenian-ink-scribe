import { useRef, useState } from "react";
import { Upload, ImageIcon, Loader2, Download, Sparkles } from "lucide-react";
import { ocrImage, ocrCorrect, ocrExport } from "@/lib/hf";

export function UploadCard() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [useLLM, setUseLLM] = useState(true);
  const [contentType, setContentType] = useState<string | null>(null);

  const handleFile = (f: File) => {
    if (!f.type.startsWith("image/")) return;
    setFile(f);
    setFileName(f.name);
    setResult(null);
    setContentType(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const runOcr = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setContentType(null);
    try {
      if (useLLM) {
        const data = await ocrCorrect(file, file.name);
        setResult(data.corrected_text || "(no text detected)");
        setContentType(data.content_type);
      } else {
        const { text } = await ocrImage(file, file.name);
        setResult(text || "(no text detected)");
      }
    } catch (e: any) {
      setResult(`Error: ${e?.message ?? e}`);
    } finally {
      setLoading(false);
    }
  };

  const runExport = async () => {
    if (!file) return;
    setExporting(true);
    try {
      const { blob, filename } = await ocrExport(file, file.name, undefined, useLLM);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (e: any) {
      alert(`Export failed: ${e?.message ?? e}`);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="mx-auto w-full max-w-3xl">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`group relative cursor-pointer rounded-2xl border border-dashed bg-card/40 p-10 transition-all ${
          dragOver ? "border-foreground bg-card/70" : "border-border hover:border-foreground/40"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
          }}
        />

        {preview ? (
          <div className="flex flex-col items-center gap-4">
            <img
              src={preview}
              alt="Uploaded preview"
              className="max-h-80 w-auto rounded-lg border border-border object-contain"
            />
            <p className="text-sm text-muted-foreground">{fileName}</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 py-6 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full border border-border bg-background/40">
              <Upload className="h-5 w-5 text-muted-foreground" />
            </div>
            <p className="text-base font-medium">Drop an image with cursive Armenian text</p>
            <p className="text-sm text-muted-foreground">PNG or JPG · click anywhere in this box</p>
          </div>
        )}
      </div>

      <div className="mt-6 flex flex-col items-center gap-4">
        {preview && (
          <label className="flex cursor-pointer items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
            <input
              type="checkbox"
              checked={useLLM}
              onChange={(e) => setUseLLM(e.target.checked)}
              className="accent-foreground"
            />
            <Sparkles className="h-4 w-4" />
            Enable LLM Grammar Correction & Smart Formatting
          </label>
        )}
        <div className="flex items-center justify-center gap-3">
          <button
            onClick={runOcr}
            disabled={!preview || loading || exporting}
            className="inline-flex items-center gap-2 rounded-full bg-foreground px-5 py-2.5 text-sm font-medium text-background transition hover:opacity-90 disabled:opacity-40"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <ImageIcon className="h-4 w-4" />}
            {loading ? "Reading…" : "Recognize text"}
          </button>
          
          <button
            onClick={runExport}
            disabled={!preview || loading || exporting}
            className="inline-flex items-center gap-2 rounded-full border border-foreground bg-transparent px-5 py-2.5 text-sm font-medium text-foreground transition hover:bg-foreground/5 disabled:opacity-40"
          >
            {exporting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
            {exporting ? "Exporting…" : "Export File"}
          </button>

          {preview && (
            <button
              onClick={() => { setPreview(null); setFile(null); setFileName(null); setResult(null); setContentType(null); }}
              className="rounded-full border border-border px-5 py-2.5 text-sm text-muted-foreground transition hover:text-foreground"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {result && (
        <div className="mt-6 rounded-2xl border border-border bg-card/50 p-6 animate-float-up">
          <div className="mb-4 flex items-center justify-between">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">Output</p>
            {contentType && (
              <span className="rounded bg-muted px-2 py-1 text-xs font-medium uppercase text-muted-foreground">
                Format: {contentType}
              </span>
            )}
          </div>
          <p className="whitespace-pre-wrap text-base leading-relaxed">{result}</p>
        </div>
      )}
    </div>
  );
}
