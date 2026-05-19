import { useRef, useState } from "react";
import { Loader2, Sparkles, Trash2 } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { recognizeImage, dataUrlToBlob } from "@/lib/hf";

type WordResult = {
  text: string;
  dataUrl: string;
  prediction: string | null;
  loading: boolean;
};

/** Render one word on a canvas using a cursive-style stroke, mimicking handwriting. */
function renderWordToDataUrl(word: string): string {
  const canvas = document.createElement("canvas");
  const padding = 24;
  const fontSize = 64;
  // Estimate width by measuring with a temp ctx
  const tmp = document.createElement("canvas").getContext("2d")!;
  tmp.font = `${fontSize}px "Poqrik", cursive`;
  const metrics = tmp.measureText(word);
  const width = Math.max(120, Math.ceil(metrics.width) + padding * 2);
  const height = fontSize + padding * 2;

  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  // Cream paper background
  ctx.fillStyle = "#f7f4ec";
  ctx.fillRect(0, 0, width, height);

  // Faint baseline
  ctx.strokeStyle = "rgba(0,0,0,0.06)";
  ctx.beginPath();
  ctx.moveTo(0, height - padding + 4);
  ctx.lineTo(width, height - padding + 4);
  ctx.stroke();

  // Slight rotation for handwritten feel
  const angle = (Math.random() - 0.5) * 0.04;
  ctx.translate(width / 2, height / 2);
  ctx.rotate(angle);
  ctx.translate(-width / 2, -height / 2);

  ctx.fillStyle = "#1a1a1a";
  ctx.font = `${fontSize}px "Poqrik", cursive`;
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  ctx.fillText(word, width / 2, height / 2);

  return canvas.toDataURL("image/png");
}

export function SyntheticDemo() {
  const [text, setText] = useState("Բարեւ աշխարհ");
  const [words, setWords] = useState<WordResult[]>([]);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const generate = async () => {
    const tokens = text.trim().split(/\s+/).filter(Boolean);
    if (!tokens.length) return;
    // Make sure the custom font is loaded before measuring/drawing on canvas
    if (typeof document !== "undefined" && document.fonts) {
      try { await document.fonts.load('64px "Poqrik"'); } catch {}
    }
    const items: WordResult[] = tokens.map((w) => ({
      text: w,
      dataUrl: renderWordToDataUrl(w),
      prediction: null,
      loading: false,
    }));
    setWords(items);
  };

  const recognizeAll = async () => {
    if (!words.length) return;
    setBusy(true);
    const updated = await Promise.all(
      words.map(async (w) => {
        try {
          const blob = await dataUrlToBlob(w.dataUrl);
          const { text } = await recognizeImage(blob, `${w.text}.png`);
          return { ...w, prediction: text, loading: false };
        } catch (e: any) {
          return { ...w, prediction: `⚠ ${e?.message ?? "error"}`, loading: false };
        }
      })
    );
    setWords(updated);
    setBusy(false);
  };

  const fullText = words
    .map((w) => w.prediction ?? "·")
    .join(" ");

  return (
    <div className="mx-auto w-full max-w-3xl">
      <div className="rounded-2xl border border-border bg-card/40 p-6">
        <label className="mb-2 block text-xs uppercase tracking-wider text-muted-foreground">
          Type Armenian text
        </label>
        <Textarea
          ref={inputRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={3}
          placeholder="Բարեւ աշխարհ"
          className="resize-none border-border bg-background/60"
        />
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <button
            onClick={generate}
            className="inline-flex items-center gap-2 rounded-full border border-border bg-background/60 px-4 py-2 text-sm transition hover:bg-background"
          >
            <Sparkles className="h-4 w-4" />
            Generate word images
          </button>
          <button
            onClick={recognizeAll}
            disabled={!words.length || busy}
            className="inline-flex items-center gap-2 rounded-full bg-foreground px-4 py-2 text-sm font-medium text-background transition hover:opacity-90 disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            {busy ? "Recognizing…" : "Run recognizer"}
          </button>
          {words.length > 0 && (
            <button
              onClick={() => setWords([])}
              className="ml-auto inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Clear
            </button>
          )}
        </div>
      </div>

      {words.length > 0 && (
        <>
          <div className="mt-6 flex flex-wrap gap-3">
            {words.map((w, i) => (
              <div
                key={i}
                className="overflow-hidden rounded-xl border border-border bg-card/40 animate-float-up"
              >
                <img src={w.dataUrl} alt={w.text} className="block h-20 w-auto" />
                <div className="border-t border-border px-3 py-2 text-xs">
                  <div className="text-muted-foreground">input · {w.text}</div>
                  <div className="font-medium">
                    pred · {w.prediction ?? <span className="text-muted-foreground">—</span>}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {words.some((w) => w.prediction) && (
            <div className="mt-6 rounded-2xl border border-border bg-card/50 p-6">
              <p className="mb-2 text-xs uppercase tracking-wider text-muted-foreground">
                Stitched output
              </p>
              <p className="whitespace-pre-wrap text-base leading-relaxed">{fullText}</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
