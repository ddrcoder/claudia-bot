# /// script
# dependencies = ["torch", "diffusers>=0.35", "transformers", "accelerate", "sentencepiece", "protobuf", "pillow"]
# ///
"""HF Job: chained FLUX.1-Kontext-dev edits. SRCS: files under /out; EDITS: prompts applied in order (each seed's chain is
independent); writes /out/<OUTDIR>/<src>_s<seed>_e<k>.png and _contact.jpg (final step of each chain)."""
import json, os, time
from pathlib import Path
import torch
from diffusers import FluxKontextPipeline
from PIL import Image, ImageDraw

OUT = Path("/out") / os.environ.get("OUTDIR", "kedit"); OUT.mkdir(parents=True, exist_ok=True)
SRCS, EDITS = json.loads(os.environ["SRCS"]), json.loads(os.environ["EDITS"])
SEEDS = [int(s) for s in os.environ.get("SEEDS", "11,23,37").split(",")]
pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to("cuda")
t0, finals = time.time(), []
for src in SRCS:
    for seed in SEEDS:
        im = Image.open(Path("/out") / src).convert("RGB")
        for k, e in enumerate(EDITS):
            p = OUT / f"{Path(src).stem}_s{seed}_e{k}.png"
            if p.exists():
                im = Image.open(p).convert("RGB"); continue
            im = pipe(image=im, prompt=e, guidance_scale=2.5, num_inference_steps=28,
                      generator=torch.Generator("cuda").manual_seed(seed + k)).images[0]
            im.save(p); print(f"{time.time() - t0:6.0f}s {p.name}", flush=True)
        finals.append(p)
tw, th, cols = 384, 384, 3
rows = (len(finals) + cols - 1) // cols
sheet = Image.new("RGB", (tw * cols, (th + 22) * rows), (16, 16, 16)); d = ImageDraw.Draw(sheet)
for i, p in enumerate(finals):
    im = Image.open(p).convert("RGB"); im.thumbnail((tw, th))
    x, y = (i % cols) * tw, (i // cols) * (th + 22)
    sheet.paste(im, (x, y)); d.text((x + 6, y + th + 4), p.stem, fill=(230, 230, 230))
sheet.save(OUT / "_contact.jpg", quality=88)
print("done", f"{time.time() - t0:.0f}s")
