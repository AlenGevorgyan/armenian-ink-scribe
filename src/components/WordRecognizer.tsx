import { useRef, useState } from "react";
import { Upload, Type, Loader2 } from "lucide-react";
import { recognizeImage } from "@/lib/hf";

export function WordRecognizer() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleFile = (f: File) => {
    if (!f.type.startsWith("image/")) return;
    setFile(f);
    setFileName(f.name);
    setResult(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const runRecognize = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    try {
      const { text } = await recognizeImage(file, file.name);
      setResult(text || "(empty)");
    } catch (e: any) {
      setResult(`Error: ${e?.message ?? e}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto w-full max-w-3xl">
      {/* Info */}
      <div className="mb-6 rounded-xl border border-border bg-card/30 p-4 text-sm text-muted-foreground">
        <p>
          Upload a <strong className="text-foreground">single cropped word</strong> image.
          The CRNN recognizer will transcribe it directly — no detection step needed.
        </p>
      </div>

      {/* Drop zone */}
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
              alt="Word crop preview"
              className="max-h-40 w-auto rounded-lg border border-border object-contain bg-white/5 px-4 py-3"
            />
            <p className="text-sm text-muted-foreground">{fileName}</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 py-6 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full border border-border bg-background/40">
              <Type className="h-5 w-5 text-muted-foreground" />
            </div>
            <p className="text-base font-medium">Drop a single word image</p>
            <p className="text-sm text-muted-foreground">
              A cropped Armenian word · PNG or JPG
            </p>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="mt-6 flex items-center justify-center gap-3">
        <button
          onClick={runRecognize}
          disabled={!preview || loading}
          className="inline-flex items-center gap-2 rounded-full bg-foreground px-5 py-2.5 text-sm font-medium text-background transition hover:opacity-90 disabled:opacity-40"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Type className="h-4 w-4" />}
          {loading ? "Reading…" : "Recognize word"}
        </button>

        {preview && (
          <button
            onClick={() => { setPreview(null); setFile(null); setFileName(null); setResult(null); }}
            className="rounded-full border border-border px-5 py-2.5 text-sm text-muted-foreground transition hover:text-foreground"
          >
            Clear
          </button>
        )}
      </div>

      {/* Result */}
      {result && (
        <div className="mt-6 rounded-2xl border border-border bg-card/50 p-6 animate-float-up">
          <p className="mb-2 text-xs uppercase tracking-wider text-muted-foreground">Prediction</p>
          <p className="text-center text-3xl font-semibold tracking-wide">{result}</p>
        </div>
      )}
    </div>
  );
}
