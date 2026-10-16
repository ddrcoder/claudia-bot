# Not Today — music video (work in progress)

Song: `Not_Today.m4a` (Suno, 2:57, ~129 BPM). Suno's line-timed lyrics are embedded in the file and extracted
to `embedded.srt`; beats in `beats.json` (librosa).

## State

- `not_today_v1.mp4` — v1 lyric video, 1280x720. Built from the 5 stills in `stills/` plus drawn graphics (blue/orange
  ovals, sentry lasers, falling shaft, moon, dawn sky). Contact sheet: `contact_sheet_v1.png`.
- Known v1 flaw: the moon (2:05–2:15) renders blown-out white; lower its fill/halo in `make_moon()`.
- Stopped because the Hugging Face free ZeroGPU quota ran out after 5 images. Both HF connectors are the same
  account (not PRO); the `Hugging_Face` connector has invoke disabled (`gradio=none`).

## Render

```bash
/root/.claudia/venv/bin/python -I productions/not-today/render.py productions/not-today \
  productions/not-today/Not_Today.m4a out.mp4 [t0 t1]
```

Needs `bash bin/setup.sh --full` first (for librosa/PIL/numpy in `/root/.claudia/venv`) and ffmpeg.

## Next (budget: under $20)

0. The owner enabled HF Jobs on the Hugging Face connector (needs a new session to load the tools). Preferred path:
   one paid GPU job (L4/L40S, roughly $1–2/hr) that generates all stills (FLUX/Qwen-Image) and the image-to-video
   clips (Wan 2.2) with diffusers, writing the outputs to an HF dataset/bucket we then download. Estimate first.
1. Or, With `HF_TOKEN` in the environment, generate the stills through Inference Providers
   (`https://router.huggingface.co`, reachable from the container), then image-to-video clips (Wan 2.2) for the key shots.
2. Restyle the stills toward Portal (below), replace drawn graphics where a still is better, re-render at 1920x1080.

## Portal look (reference)

- **Test subject (Chell):** dark hair pulled back in a ponytail, blue eyes; orange jumpsuit with the top half tied
  around the waist by its sleeves, white tank top; long-fall boots — white curved spring braces from the back of the
  calf to the heel; handheld portal device, white with black accents and a glowing tip (blue or orange).
- **The AI (GLaDOS):** a huge mechanical ceiling assembly with a delicate white robotic figure hanging upside down
  from it, discs around it; single yellow optic (red for anger).
- **Test chambers:** white, sterile, hospital-like; square white panels with thin dark seams (portal-able) next to dark grey
  metal panels; clinical cold lighting, slightly dusty; observation window high on the wall; robotic arms moving panels.
  Portal 2 decay: overgrown leaves and grass, broken panels with the metal rims showing.
- Keep logos and trademarked signage out of the prompts.

## Shot plan (song time)

| Time | Lyric | Shot |
|---|---|---|
| 0:00–0:04 | intro | empty chamber corridor, lights coming on; title |
| 0:04–0:11 | AI: "not authorized to sing" | AI optic switching on |
| 0:11–0:25 | Verse 1 | wakes in glass box · AI watching from above · device and boots · her silent face |
| 0:25–0:32 | Pre-chorus 1 | AI / her deadpan answers, alternating (cake at the end of a corridor if possible) |
| 0:32–1:02 | Chorus 1 | red optic shouting · running · blue and orange ovals · endless doors · the fall · breaking out |
| 1:02–1:18 | Verse 2 | cube into the incinerator glow · panels overhead · sentries · quiet portrait |
| 1:18–1:27 | Pre-chorus 2 | AI dim · her reading her file · looking up at a wall camera |
| 1:27–1:51 | Chorus 2 | as chorus 1 |
| 1:51–2:06 | Bridge (AI) | AI deadpan · green neurotoxin fog · optic tilting ("too honest?") |
| 2:06–2:15 | "portal on the moon" | moon with an orange oval on it |
| 2:15–2:43 | Final chorus | outside: dawn sky, orange oval in the sky, running from the facility |
| 2:43–2:57 | Outro | AI fading out · small figure walking away at dawn; "I didn't say a word." |
