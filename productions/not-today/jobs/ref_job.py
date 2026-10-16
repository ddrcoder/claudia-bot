# /// script
# dependencies = ["torch", "diffusers>=0.35", "transformers", "accelerate", "sentencepiece", "protobuf", "pillow"]
# ///
"""HF Job: Not Today character reference. FLUX.1-dev makes empty-handed full-body candidates; FLUX.1-Kontext-dev then puts
the portal gun from /out/refs/portal_gun.png into her hands (candidate | gun, side by side). Writes /out/ref3/."""
import gc, json, os, time
from pathlib import Path
import torch
from diffusers import FluxPipeline, FluxKontextPipeline
from PIL import Image, ImageDraw

OUT = Path("/out") / os.environ.get("OUTDIR", "ref3"); OUT.mkdir(parents=True, exist_ok=True)
L = json.loads(os.environ["LOOKS"])
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1,2,3,4,5,6").split(",")]
t0 = time.time()

cand = ("Full-body wide cinematic photo of " + L["CHELL"] + ", wearing " + L["BOOTS"] + ". Her hands are empty and relaxed at her sides. "
        "She stands in the middle of a bright sterile white test chamber of square white wall panels with thin dark seams and grey "
        "metal floor tiles, facing the camera, deadpan expression, the whole body in frame from head to feet with space above and below. "
        "Photoreal, no text, no logos.")
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
for s in SEEDS:
    p = OUT / f"cand_{s}.png"
    if not p.exists():
        pipe(cand, width=1344, height=768, num_inference_steps=28, guidance_scale=3.5,
             generator=torch.Generator("cuda").manual_seed(s)).images[0].save(p)
        print(f"{time.time() - t0:6.0f}s {p.name}", flush=True)
del pipe; gc.collect(); torch.cuda.empty_cache()

gun = Image.open("/out/refs/portal_gun.png").convert("RGB")
gun = gun.resize((576, round(gun.height * 576 / gun.width)))
kpipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to("cuda")
edit = ("The woman on the left now holds the exact white-and-black device shown on the right, at its true large size, in both hands "
        "across her body like a rifle: the white rear shell tucked under her right arm, the black ribbed barrel and three black claw "
        "prongs pointing forward and slightly down, a blue glow in the barrel. Keep her face, hair, outfit and boots exactly the same. "
        "Remove the right panel and extend the white test chamber so the image shows only her, full body, photoreal.")
done = []
for s in SEEDS:
    c = Image.open(OUT / f"cand_{s}.png").convert("RGB")
    comp = Image.new("RGB", (1344, 768), (225, 225, 225))
    comp.paste(c.crop((288, 0, 1056, 768)), (0, 0))
    comp.paste(gun, (768, (768 - gun.height) // 2))
    comp.save(OUT / f"comp_{s}.jpg", quality=90)
    for ks in (11, 23):
        p = OUT / f"ref_{s}_{ks}.png"
        if not p.exists():
            kpipe(image=comp, prompt=edit, guidance_scale=2.5, num_inference_steps=28,
                  generator=torch.Generator("cuda").manual_seed(ks)).images[0].save(p)
            print(f"{time.time() - t0:6.0f}s {p.name}", flush=True)
        done.append(p)

tw, th, cols = 448, 256, 4
items = [OUT / f"cand_{s}.png" for s in SEEDS] + done
rows = (len(items) + cols - 1) // cols
sheet = Image.new("RGB", (tw * cols, (th + 22) * rows), (16, 16, 16))
d = ImageDraw.Draw(sheet)
for i, p in enumerate(items):
    im = Image.open(p).convert("RGB"); im.thumbnail((tw, th))
    x, y = (i % cols) * tw, (i // cols) * (th + 22)
    sheet.paste(im, (x, y)); d.text((x + 6, y + th + 4), p.stem, fill=(230, 230, 230))
sheet.save(OUT / "_contact.jpg", quality=88)
print("done", f"{time.time() - t0:.0f}s")
