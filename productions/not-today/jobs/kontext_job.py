# /// script
# dependencies = ["torch", "diffusers>=0.35", "transformers", "accelerate", "sentencepiece", "protobuf", "pillow"]
# ///
"""HF Job: Not Today character shots with FLUX.1-Kontext-dev from one reference image.
REF env: reference file in /out; SHOTS env: {id: prompt}; writes /out/<OUTDIR>/<id>_<seed>.png + _contact.jpg."""
import json, os, time
from pathlib import Path
import torch
from diffusers import FluxKontextPipeline
from PIL import Image, ImageDraw

OUT = Path("/out") / os.environ.get("OUTDIR", "stills_v3k"); OUT.mkdir(parents=True, exist_ok=True)
REF = Image.open(Path("/out") / os.environ["REF"]).convert("RGB")
SHOTS = json.loads(os.environ["SHOTS"])
SEEDS = [int(s) for s in os.environ.get("SEEDS", "11,23").split(",")]

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to("cuda")
t0 = time.time()
done = []
for sid, prompt in SHOTS.items():
    for seed in SEEDS:
        p = OUT / f"{sid}_{seed}.png"
        if not p.exists():
            im = pipe(image=REF, prompt=prompt, guidance_scale=2.5, num_inference_steps=28, width=1392, height=752,
                      generator=torch.Generator("cuda").manual_seed(seed)).images[0]
            im.save(p)
            print(f"{time.time() - t0:7.0f}s {p.name} {im.size}", flush=True)
        done.append(p)

tw, th, cols = 448, 256, 4
rows = (len(done) + cols - 1) // cols
sheet = Image.new("RGB", (tw * cols, (th + 22) * rows), (16, 16, 16))
d = ImageDraw.Draw(sheet)
for i, p in enumerate(done):
    x, y = (i % cols) * tw, (i // cols) * (th + 22)
    sheet.paste(Image.open(p).convert("RGB").resize((tw, th)), (x, y))
    d.text((x + 6, y + th + 4), p.stem, fill=(230, 230, 230))
sheet.save(OUT / "_contact.jpg", quality=88)
print("done", len(done), f"{time.time() - t0:.0f}s")
