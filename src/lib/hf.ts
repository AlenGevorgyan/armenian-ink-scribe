// Set this to your Hugging Face Space URL, e.g.
// "https://your-username-your-space.hf.space"
export const HF_SPACE_URL = "https://alengevorgyan-ocrm.hf.space";

type ApiPath = "/ocr" | "/recognize" | "/ocr/correct" | "/ocr/export";

function getSpaceUrl(path: ApiPath) {
  const rawUrl = HF_SPACE_URL.trim();
  const spacesMatch = rawUrl.match(/^https:\/\/huggingface\.co\/spaces\/([^/]+)\/([^/?#]+)/i);
  const baseUrl = spacesMatch
    ? `https://${spacesMatch[1]}-${spacesMatch[2]}.hf.space`
    : rawUrl.replace(/\/+$/, "");

  return `${baseUrl}${path}`;
}

async function postImage(path: ApiPath, file: Blob, filename = "image.png") {
  const fd = new FormData();
  fd.append("file", file, filename);
  const res = await fetch(getSpaceUrl(path), { method: "POST", body: fd });
  if (!res.ok) throw new Error(`${path} failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export const ocrImage = (file: Blob, filename?: string) =>
  postImage("/ocr", file, filename) as Promise<{ text: string; words: unknown[] }>;

export const recognizeImage = (file: Blob, filename?: string) =>
  postImage("/recognize", file, filename) as Promise<{ text: string }>;

export const ocrCorrect = (file: Blob, filename?: string) =>
  postImage("/ocr/correct", file, filename) as Promise<{ 
    raw_text: string; 
    corrected_text: string; 
    content_type: string; 
    rows?: string[][]; 
    words: unknown[];
  }>;

export async function ocrExport(file: Blob, filename?: string, format?: string, fixGrammar = true) {
  const fd = new FormData();
  fd.append("file", file, filename || "image.png");
  
  let url = getSpaceUrl("/ocr/export");
  const params = new URLSearchParams();
  if (format) params.append("format", format);
  params.append("fix_grammar", String(fixGrammar));
  if (params.toString()) url += "?" + params.toString();

  const res = await fetch(url, { method: "POST", body: fd });
  if (!res.ok) throw new Error(`/ocr/export failed: ${res.status} ${await res.text()}`);
  
  const blob = await res.blob();
  const contentDisposition = res.headers.get("Content-Disposition");
  let downloadedFilename = "result.pdf";
  if (contentDisposition) {
    const match = contentDisposition.match(/filename="?([^"]+)"?/);
    if (match) downloadedFilename = match[1];
  }
  
  return { blob, filename: downloadedFilename };
}

export async function dataUrlToBlob(dataUrl: string): Promise<Blob> {
  const res = await fetch(dataUrl);
  return res.blob();
}
