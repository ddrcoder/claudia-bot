# /// script
# dependencies = ["torch", "diffusers>=0.35", "transformers", "accelerate", "sentencepiece", "protobuf", "pillow"]
# ///
"""HF Job: Not Today stills with FLUX.1-dev. Writes /out/stills_v2/<id>_<seed>.png and a contact sheet."""
import json, os, sys, time
from pathlib import Path
import torch
from diffusers import FluxPipeline
from PIL import Image, ImageDraw

OUT = Path("/out/stills_v2"); OUT.mkdir(parents=True, exist_ok=True)
SHOTS = json.loads(os.environ["SHOTS"])
SEEDS = [int(s) for s in os.environ.get("SEEDS", "11,23").split(",")]
W, H = 1344, 768

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
t0 = time.time()
done = []
for sid, prompt in SHOTS.items():
    for seed in SEEDS:
        p = OUT / f"{sid}_{seed}.png"
        if p.exists():
            done.append(p); continue
        im = pipe(prompt, width=W, height=H, num_inference_steps=28, guidance_scale=3.5,
                  generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        im.save(p)
        done.append(p)
        print(f"{time.time() - t0:7.0f}s {p.name}", flush=True)

# contact sheet, 4 across
tw, th = 448, 256
cols = 4
rows = (len(done) + cols - 1) // cols
sheet = Image.new("RGB", (tw * cols, (th + 22) * rows), (16, 16, 16))
d = ImageDraw.Draw(sheet)
for i, p in enumerate(done):
    x, y = (i % cols) * tw, (i // cols) * (th + 22)
    sheet.paste(Image.open(p).convert("RGB").resize((tw, th)), (x, y))
    d.text((x + 6, y + th + 4), p.stem, fill=(230, 230, 230))
sheet.save(OUT / "_contact.jpg", quality=88)
print("done", len(done), f"{time.time() - t0:.0f}s")
