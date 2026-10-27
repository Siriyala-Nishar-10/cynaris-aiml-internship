"""Convert the PDFs in ./pdfs into plain .txt files in ./data (LlamaIndex indexes the .txt files)."""
import glob, os
import pypdf

os.makedirs("data", exist_ok=True)
pdfs = sorted(glob.glob("pdfs/*.pdf"))
if not pdfs:
    raise SystemExit("No PDFs in ./pdfs. Copy your 5 PDFs there (or put your own .txt files straight into ./data).")
for p in pdfs:
    text = "\n".join((pg.extract_text() or "") for pg in pypdf.PdfReader(p).pages)
    out = os.path.join("data", os.path.splitext(os.path.basename(p))[0] + ".txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{p} -> {out} ({len(text.split())} words)")
