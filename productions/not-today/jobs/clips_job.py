# /// script
# dependencies = ["torch", "diffusers>=0.35", "transformers", "accelerate", "sentencepiece", "protobuf", "pillow", "imageio", "imageio-ffmpeg", "ftfy", "peft"]
# ///
"""HF Job: Not Today image-to-video clips, Wan 2.2 I2V A14B + Lightning (4 steps).
CLIPS env: JSON list of [id, still path under /out, motion prompt]. Writes /out/clips_v2/<id>.mp4 (1280x720, 16 fps, 81 frames)."""
import json, os, time
from pathlib import Path
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

SRC, OUT = Path("/out"), Path("/out") / os.environ.get("OUTDIR", "clips_v3")
OUT.mkdir(parents=True, exist_ok=True)
CLIPS = json.loads(os.environ["CLIPS"])
W, H = 1280, 720
NEG = ("static, frozen, blurry, low quality, jpeg artifacts, text, subtitles, watermark, logo, deformed hands, extra fingers, "
       "extra limbs, distorted face, flicker, morphing")

try:
    # official weights + the official 4-step Lightning LoRA (lightx2v), one LoRA per expert
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", torch_dtype=torch.bfloat16)
    lora = "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"
    pipe.load_lora_weights("lightx2v/Wan2.2-Lightning", weight_name=f"{lora}/high_noise_model.safetensors", adapter_name="high")
    pipe.load_lora_weights("lightx2v/Wan2.2-Lightning", weight_name=f"{lora}/low_noise_model.safetensors", adapter_name="low",
                           load_into_transformer_2=True)
    pipe.set_adapters(["high", "low"], adapter_weights=[1.0, 1.0])
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=5.0)
    print("official weights + lightx2v LoRA", flush=True)
except Exception as e:
    print("LoRA path failed, using the merged Lightning repo:", repr(e)[:300], flush=True)
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
