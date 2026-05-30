import { useState } from "react";
import type { DetectedWord } from "@/lib/hf";

const COLOR_CLASSES: Record<string, { border: string; bg: string; chip: string }> = {
  emerald: { border: "border-emerald-400", bg: "bg-emerald-400/10", chip: "bg-emerald-500 text-black" },
  amber:   { border: "border-amber-400",   bg: "bg-amber-400/10",   chip: "bg-amber-500 text-black" },
  rose:    { border: "border-rose-400",    bg: "bg-rose-400/10",    chip: "bg-rose-500 text-white" },
};

function colorFor(conf: number | undefined) {
  if (conf == null) return "emerald";
  if (conf >= 70) return "emerald";
  if (conf >= 40) return "amber";
  return "rose";
}

export function OcrOverlay({
  imageUrl,
  words,
}: { imageUrl: string; words: DetectedWord[] }) {
  const [showBoxes, setShowBoxes] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);

  return (
    <div className="mt-6 rounded-2xl border border-border bg-card/50 p-6 animate-float-up">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <p className="text-xs uppercase tracking-wider text-muted-foreground">
          Detected words ({words.length})
        </p>
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={showBoxes}
              onChange={(e) => setShowBoxes(e.target.checked)}
              className="accent-foreground"
            />
            Boxes
          </label>
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={showLabels}
              onChange={(e) => setShowLabels(e.target.checked)}
              className="accent-foreground"
            />
            Labels
          </label>
        </div>
      </div>

      <div className="relative inline-block max-w-full">
        <img
          src={imageUrl}
          alt="OCR target"
          className="block max-h-[640px] w-auto max-w-full rounded-lg border border-border"
          onLoad={(e) => {
            const img = e.currentTarget;
            setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
          }}
        />
        {naturalSize &&
          words.map((w, i) => {
            const left = (w.x1 / naturalSize.w) * 100;
            const top = (w.y1 / naturalSize.h) * 100;
            const width = ((w.x2 - w.x1) / naturalSize.w) * 100;
            const height = ((w.y2 - w.y1) / naturalSize.h) * 100;
            const c = COLOR_CLASSES[colorFor(w.conf)];
            return (
              <div
                key={i}
                className={`absolute ${showBoxes ? `border-2 ${c.border} ${c.bg}` : ""}`}
                style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}
                title={`${w.text || "—"}${w.conf != null ? `  (conf ${Math.round(w.conf)})` : ""}`}
              >
                {showLabels && (
                  <span
                    className={`pointer-events-none absolute -top-[18px] left-0 whitespace-nowrap rounded px-1.5 py-0.5 text-[11px] font-medium shadow ${c.chip}`}
                  >
                    {w.text || "—"}
                  </span>
                )}
              </div>
            );
          })}
      </div>

      <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
        Each box is one word the <strong>detector</strong> (Tesseract PSM 11) found. The label above
        each box is the <strong>CRNN's</strong> prediction. Missing boxes → words the detector did
        not find. Wrong labels → recognition errors. Colors reflect detector confidence:{" "}
        <span className="text-emerald-400">high</span>,{" "}
        <span className="text-amber-400">medium</span>,{" "}
        <span className="text-rose-400">low</span>.
      </p>
    </div>
  );
}
