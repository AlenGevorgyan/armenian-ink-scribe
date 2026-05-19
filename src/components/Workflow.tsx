import { ScanSearch, Crop, Type, Sparkles } from "lucide-react";

const steps = [
  {
    icon: ScanSearch,
    title: "Detect",
    body: "A YOLOv8 model scans the uploaded page and predicts a bounding box around every handwritten word.",
  },
  {
    icon: Crop,
    title: "Crop",
    body: "Each word region is cropped from the original image and normalized to a fixed height for the recognizer.",
  },
  {
    icon: Type,
    title: "Recognize",
    body: "A CRNN model reads each crop one by one, then words are sorted line-by-line and stitched back into the full text.",
  },
  {
    icon: Sparkles,
    title: "Correct & Export",
    body: "An LLM fixes grammar errors, detects tables vs prose, and exports the result as PDF, XLSX, CSV, or TXT.",
  },
];

export function Workflow() {
  return (
    <section className="mx-auto w-full max-w-5xl px-6 py-24">
      <div className="mb-14 text-center">
        <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">How it works</p>
        <h2 className="mt-3 text-3xl font-semibold tracking-tight sm:text-4xl">
          Two models, one LLM — a clean pipeline
        </h2>
      </div>

      <div className="grid gap-px overflow-hidden rounded-2xl border border-border bg-border sm:grid-cols-4">
        {steps.map((s, i) => {
          const Icon = s.icon;
          return (
            <div key={s.title} className="bg-card p-8">
              <div className="mb-5 flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-border bg-background">
                  <Icon className="h-4 w-4" />
                </div>
                <span className="text-xs text-muted-foreground">0{i + 1}</span>
              </div>
              <h3 className="text-lg font-medium">{s.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{s.body}</p>
            </div>
          );
        })}
      </div>
    </section>
  );
}
