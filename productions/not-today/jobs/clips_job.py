# /// script
# dependencies = ["torch", "diffusers>=0.35", "transformers", "accelerate", "sentencepiece", "protobuf", "pillow", "imageio", "imageio-ffmpeg", "ftfy"]
# ///
"""HF Job: Not Today image-to-video clips, Wan 2.2 I2V A14B + Lightning (4 steps).
CLIPS env: JSON list of [id, still file in /out/stills_v2, motion prompt]. Writes /out/clips_v2/<id>.mp4 (1280x720, 16 fps, 81 frames)."""
import json, os, time
from pathlib import Path
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

SRC, OUT = Path("/out/stills_v2"), Path("/out/clips_v2")
OUT.mkdir(parents=True, exist_ok=True)
CLIPS = json.loads(os.environ["CLIPS"])
W, H = 1280, 720
NEG = ("static, frozen, blurry, low quality, jpeg artifacts, text, subtitles, watermark, logo, deformed hands, extra fingers, "
       "extra limbs, distorted face, flicker, morphing")

pipe = WanImageToVideoPipeline.from_pretrained("magespace/Wan2.2-I2V-A14B-Lightning-Diffusers", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
t0 = time.time()
for cid, still, prompt in CLIPS:
    p = OUT / f"{cid}.mp4"
    if p.exists():
        continue
    im = Image.open(SRC / still).convert("RGB")
    s = max(W / im.width, H / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    l, t = (im.width - W) // 2, (im.height - H) // 2
    im = im.crop((l, t, l + W, t + H))
    frames = pipe(image=im, prompt=prompt, negative_prompt=NEG, height=H, width=W, num_frames=81,
                  num_inference_steps=4, guidance_scale=1.0, guidance_scale_2=1.0,
                  generator=torch.Generator("cpu").manual_seed(7)).frames[0]
    export_to_video(frames, str(p), fps=16)
    print(f"{time.time() - t0:7.0f}s {p.name}", flush=True)
print("done", f"{time.time() - t0:.0f}s")
