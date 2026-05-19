import { createFileRoute } from "@tanstack/react-router";
import { CursorField } from "@/components/CursorField";
import { UploadCard } from "@/components/UploadCard";
import { SyntheticDemo } from "@/components/SyntheticDemo";
import { Workflow } from "@/components/Workflow";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Armenian Cursive OCR — Handwriting Recognition" },
      {
        name: "description",
        content:
          "Upload a photo of cursive Armenian handwriting and let neural networks detect, transcribe, correct grammar with AI, and export as PDF or Excel.",
      },
    ],
  }),
  component: Index,
});

function Index() {
  return (
    <main className="min-h-screen bg-background text-foreground">
      {/* Hero */}
      <section className="relative h-[88vh] min-h-[600px] w-full overflow-hidden">
        <CursorField />
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_30%,var(--color-background)_85%)]" />

        <nav className="relative z-10 mx-auto flex max-w-6xl items-center justify-between px-6 py-6">
          <div className="flex items-center gap-2 text-sm font-medium tracking-tight">
            <div className="h-2 w-2 rounded-full bg-foreground" />
            ArmOCR
          </div>
          <a
            href="#upload"
            className="text-sm text-muted-foreground transition hover:text-foreground"
          >
            Try it
          </a>
        </nav>

        <div className="relative z-10 mx-auto flex h-[calc(88vh-96px)] min-h-[504px] max-w-3xl flex-col items-center justify-center px-6 text-center">
          <span className="mb-6 inline-flex items-center gap-2 rounded-full border border-border bg-card/40 px-3 py-1 text-xs text-muted-foreground backdrop-blur">
            <span className="h-1.5 w-1.5 rounded-full bg-foreground" />
            Cursive Armenian handwriting OCR
          </span>
          <h1 className="text-balance text-5xl font-semibold leading-[1.05] tracking-tight sm:text-6xl">
            Read handwriting<br />machines couldn’t.
          </h1>
          <p className="mt-6 max-w-xl text-balance text-base leading-relaxed text-muted-foreground sm:text-lg">
            A two-stage neural pipeline that finds every word in a page of
            cursive Armenian and transcribes it into editable text.
          </p>
          <a
            href="#upload"
            className="mt-10 inline-flex items-center gap-2 rounded-full bg-foreground px-6 py-3 text-sm font-medium text-background transition hover:opacity-90"
          >
            Upload an image
          </a>
        </div>
      </section>

      {/* Try */}
      <section id="upload" className="px-6 py-24">
        <div className="mx-auto mb-10 max-w-3xl text-center">
          <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Try the model</p>
          <h2 className="mt-3 text-3xl font-semibold tracking-tight sm:text-4xl">
            Upload or generate
          </h2>
          <p className="mt-3 text-sm text-muted-foreground">
            Drop a real photo, or type any Armenian text and let us synthesize word images for the model to read.
          </p>
        </div>

        <Tabs defaultValue="upload" className="mx-auto w-full max-w-3xl">
          <TabsList className="mx-auto mb-8 grid w-full max-w-xs grid-cols-2 rounded-full border border-border bg-card/40 p-1">
            <TabsTrigger value="upload" className="rounded-full">Upload image</TabsTrigger>
            <TabsTrigger value="synthetic" className="rounded-full">Synthetic demo</TabsTrigger>
          </TabsList>
          <TabsContent value="upload"><UploadCard /></TabsContent>
          <TabsContent value="synthetic"><SyntheticDemo /></TabsContent>
        </Tabs>
      </section>

      {/* Workflow */}
      <Workflow />

      <footer className="border-t border-border px-6 py-10 text-center text-xs text-muted-foreground">
        Built with YOLOv8 + CRNN + LLM · Hosted on Hugging Face
      </footer>
    </main>
  );
}
