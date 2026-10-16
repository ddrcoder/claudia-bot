"""Not Today - music video v2 (stills + Wan clips, 1080p). Usage: python render.py MVDIR AUDIO OUT.mp4 [t0 t1]"""
import json, math, random, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance

MV, AUDIO, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
T0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
T1 = float(sys.argv[5]) if len(sys.argv) > 5 else 177.16
W, H, FPS = 1920, 1080, 30
K = W / 1280  # layout scale vs v1
FONTS = Path("/home/user/claudia-bot/engine/fonts")
F_AI = ImageFont.truetype(str(FONTS / "ShareTechMono-Regular.ttf"), int(40 * K))
F_HER = ImageFont.truetype(str(FONTS / "BarlowCondensed-SemiBold.ttf"), int(62 * K))
F_TITLE = ImageFont.truetype(str(FONTS / "DINishCondensed-Black.ttf"), int(150 * K))
AMBER, WHITE = (255, 176, 64), (245, 245, 240)
random.seed(3)

beats = json.loads((MV / "beats.json").read_text())["beats"]

# ------------------------------------------------------------------ sources (all W*1.3 x H*1.3)
SW, SH = int(W * 1.3), int(H * 1.3)


def load(name):
    return Image.open(MV / "stills" / name).convert("RGB").resize((SW, SH), Image.LANCZOS)


def grade(im, rgb=(1, 1, 1), sat=1.0, bright=1.0):
    im = ImageEnhance.Color(im).enhance(sat)
    im = ImageEnhance.Brightness(im).enhance(bright)
    a = np.asarray(im).astype(np.float32) * np.array(rgb, np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def glow_ellipse(size, box, color, width=18, blur=22):
    layer = Image.new("RGB", size, (0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse(box, outline=color, width=width)
    g = layer.filter(ImageFilter.GaussianBlur(blur))
    inner = Image.new("RGB", size, (0, 0, 0))
    ImageDraw.Draw(inner).ellipse(box, fill=tuple(int(c * .35) for c in color))
    inner = inner.filter(ImageFilter.GaussianBlur(8))
    return np.asarray(layer).astype(np.float32) * 1.6 + np.asarray(g).astype(np.float32) * 2.2 + np.asarray(inner)


def add(im, *layers):
    a = np.asarray(im).astype(np.float32)
    for l in layers:
        a = a + l
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


class Clip:
    """A Wan clip (1280x720, 16 fps), decoded lazily; plays at its own speed and ping-pongs when the shot outlasts it."""
    cache = {}

    def __init__(self, path, fps=16, start=0.0):
        self.path, self.fps, self.start = path, fps, start

    def frames(self):
        if self.path not in Clip.cache:
            if len(Clip.cache) >= 3:
                Clip.cache.pop(next(iter(Clip.cache)))
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(self.path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                 capture_output=True, check=True).stdout
            a = np.frombuffer(raw, np.uint8).reshape(-1, 720, 1280, 3)
            Clip.cache[self.path] = [Image.fromarray(f) for f in a]
        return Clip.cache[self.path]

    def frame(self, dt):
        fr = self.frames()
        n = len(fr)
        i = int((dt + self.start) * self.fps)
        i %= 2 * n - 2
        return fr[i if i < n else 2 * n - 2 - i]


def clip(name, **kw):
    return Clip(MV / "clips" / f"{name}.mp4", **kw)


S = {}
S["corridor"] = load("s00_corridor.webp")
S["lens"] = load("ai_lens.webp")
S["watched"] = load("s_watched.webp")
S["device"] = load("s_device.webp")
S["glass"] = load("s_glassbox.webp")
S["lens_red"] = grade(S["lens"], (1.5, .55, .45), sat=1.4)
S["lens_dim"] = grade(S["lens"], bright=.45)
S["poison"] = grade(S["lens"], (.55, 1.25, .6))
S["device_hot"] = grade(S["device"], (1.35, .8, .55))
S["corridor_ceil"] = S["corridor"].transpose(Image.FLIP_TOP_BOTTOM)
S["portals"] = add(grade(S["corridor"], bright=.7),
                   glow_ellipse((SW, SH), (300, 250, 520, 700), (40, 140, 255)),
                   glow_ellipse((SW, SH), (680, 690, 1150, 860), (255, 120, 20)))
S["face"] = S["glass"].crop((585, 150, 1075, 426)).resize((SW, SH), Image.LANCZOS)  # her face from the glass box


def make_turrets():
    im = Image.new("RGB", (SW, SH), (6, 8, 12))
    d = ImageDraw.Draw(im)
    lay = Image.new("RGB", (SW, SH), (0, 0, 0))
    dl = ImageDraw.Draw(lay)
    for i in range(7):
        x = 160 + i * 220
        y = 620 + (i % 2) * 30
        d.ellipse((x - 34, y - 90, x + 34, y + 60), fill=(205, 208, 212))
        d.ellipse((x - 9, y - 40, x + 9, y - 22), fill=(255, 40, 30))
        tx = random.randint(0, SW)
        dl.line((x, y - 31, tx, 0), fill=(255, 30, 20), width=3)
        dl.ellipse((x - 16, y - 47, x + 16, y - 15), fill=(255, 60, 40))
    return add(im, np.asarray(lay.filter(ImageFilter.GaussianBlur(6))) * 2.0, np.asarray(lay) * .8)


def make_moon():
    im = Image.new("RGB", (SW, SH), (4, 6, 18))
    d = ImageDraw.Draw(im)
    for _ in range(500):
        x, y = random.randint(0, SW), random.randint(0, SH)
        b = random.randint(80, 255)
        d.point((x, y), fill=(b, b, b))
    cx, cy, r = SW // 2 + 120, SH // 2 - 40, 230
    moon = Image.new("RGB", (SW, SH), (0, 0, 0))
    dm = ImageDraw.Draw(moon)
    dm.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(150, 150, 142))
    for _ in range(40):
        a, rr = random.random() * 6.28, random.random() * r * .85
        x, y, s = cx + rr * math.cos(a), cy + rr * math.sin(a), random.randint(8, 30)
        dm.ellipse((x - s, y - s, x + s, y + s), fill=(112, 112, 106))
    halo = np.asarray(moon.filter(ImageFilter.GaussianBlur(40))) * .2
    orange = glow_ellipse((SW, SH), (cx - 90, cy + 20, cx + 30, cy + 150), (255, 120, 20), width=10, blur=14)
    base = add(im, np.asarray(moon), halo)
    return add(base, orange)


def make_sky(t_bright=1.0):
    y = np.linspace(0, 1, SH)[:, None, None]
    top, mid, bot = np.array([30, 70, 150]), np.array([250, 150, 80]), np.array([255, 205, 140])
    g = np.where(y < .6, top + (mid - top) * (y / .6), mid + (bot - mid) * ((y - .6) / .4))
    a = np.broadcast_to(g, (SH, SW, 3)).astype(np.float32) * t_bright
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    pts = [(0, SH)] + [(x, int(SH * .78 + 30 * math.sin(x / 140) + 18 * math.sin(x / 47))) for x in range(0, SW + 20, 20)] + [(SW, SH)]
    d.polygon(pts, fill=(25, 20, 30))
    # the facility behind her, small and far
    d.rectangle((150, int(SH * .70), 420, int(SH * .80)), fill=(40, 36, 46))
    # a tiny figure walking away
    fx, fy = SW // 2 + 60, int(SH * .80)
    d.ellipse((fx - 7, fy - 62, fx + 7, fy - 48), fill=(20, 16, 24))
    d.rectangle((fx - 8, fy - 48, fx + 8, fy - 14), fill=(20, 16, 24))
    d.line((fx - 4, fy - 14, fx - 8, fy + 4), fill=(20, 16, 24), width=6)
    d.line((fx + 4, fy - 14, fx + 9, fy + 2), fill=(20, 16, 24), width=6)
    return add(im, glow_ellipse((SW, SH), (SW // 2 + 180, 120, SW // 2 + 420, 300), (255, 110, 10), width=12, blur=20) * .8)


S["turrets"] = make_turrets()
S["moon"] = make_moon()
S["sky"] = make_sky()
S["sky_dim"] = grade(S["sky"], bright=.55, sat=.7)
S["black"] = Image.new("RGB", (SW, SH), (0, 0, 0))
ARR = {k: v for k, v in S.items()}

# ------------------------------------------------------------------ edit list
# (start, source, move) ; move: zin zout pl pr push shake fall tilt still
def beat_cuts(t0, t1, every, seq):
    bs = [b for b in beats if t0 - .05 <= b < t1 - .2][::every]
    if not bs or bs[0] > t0 + .1:
        bs = [t0] + bs
    bs[0] = t0
    return [(b, seq[i % len(seq)][0], seq[i % len(seq)][1]) for i, b in enumerate(bs)]


def chorus(t, final=False):
    # t = chorus start; the lines are at fixed offsets from the start
    shout = "lens_red"
    ed = [(t, shout, "shake"), (t + 2.31, "face", "zin"), (t + 3.75, shout, "shake"),
          (t + 5.75, "corridor", "push"), (t + 8.14, "sky" if final else "portals", "pr"),
          (t + 12.05, "portals" if final else "corridor", "pl"), (t + 15.64, "black", "fall")]
    end = t + 18.59
    tail_seq = ([("sky", "zin"), ("face", "zin"), ("portals", "pr"), ("watched", "zout")] if final else
                [("portals", "zin"), ("watched", "zout"), ("device", "pl"), ("face", "zin")])
    return ed, end, tail_seq


E = [(0.0, "corridor", "push"), (3.75, "lens", "zin"), (10.69, "glass", "zin"), (14.2, "watched", "zout"),
     (17.47, "device", "pr"), (21.14, "face", "zin"), (25.21, "lens", "zin"), (27.05, "face", "zout"),
     (29.12, "lens", "zin"), (30.8, "watched", "pl")]
ed, end, tail = chorus(32.47)
E += ed + beat_cuts(end, 62.39, 4, tail)
E += [(62.39, "device_hot", "zin"), (66.14, "corridor_ceil", "pl"), (70.29, "turrets", "pr"), (73.8, "glass", "zin"),
      (78.43, "lens_dim", "zin"), (79.79, "face", "zout"), (81.54, "lens", "zin"), (83.38, "watched", "zin")]
ed, end, tail = chorus(86.65)
E += ed + beat_cuts(end, 110.67, 4, tail)
E += [(110.67, "lens", "zin"), (117.53, "poison", "zout"), (123.51, "lens", "tilt"), (125.59, "moon", "zin")]
ed, end, tail = chorus(134.84, final=True)
E += ed + beat_cuts(end, 162.77, 4, tail)
E += [(162.77, "lens_dim", "zout"), (171.22, "sky_dim", "zin"), (177.16, "black", "still")]
E.sort(key=lambda e: e[0])

FLASH_AT = [e[0] for e in E if 32 < e[0] < 62.4 or 86 < e[0] < 110.7 or 134 < e[0] < 162.8]


def shot_at(t):
    for i in range(len(E) - 1, -1, -1):
        if E[i][0] <= t:
            return E[i], (E[i + 1][0] if i + 1 < len(E) else 177.16)
    return E[0], E[1][0]


def frame_bg(t):
    (s0, src, mv), s1 = shot_at(t)
    p = min(1, max(0, (t - s0) / max(.01, s1 - s0)))
    e = p * p * (3 - 2 * p)
    z, dx, dy, rot = 1.0, 0.0, 0.0, 0.0
    if mv == "zin": z = 1 + .12 * e
    elif mv == "zout": z = 1.12 - .12 * e
    elif mv == "pl": dx = .08 * (1 - 2 * e); z = 1.06
    elif mv == "pr": dx = -.08 * (1 - 2 * e); z = 1.06
    elif mv == "push": z = 1 + .3 * p
    elif mv == "tilt": rot = 9 * e; z = 1.1
    elif mv == "shake":
        z = 1.08; dx = .012 * math.sin(t * 71); dy = .012 * math.cos(t * 53)
    if src == "black" and mv == "fall":
        return fall_frame(t - s0)
    im = ARR[src]
    vw, vh = SW / 1.15 / z, SH / 1.15 / z
    cx, cy = SW / 2 + dx * SW, SH / 2 + dy * SH
    if isinstance(im, Clip):
        # clips carry their own motion: keep the move small and view nearly the whole frame
        im = im.frame(t - s0)
        sc = im.width / SW
        z2 = 1 + (z - 1) * .5
        vw, vh = im.width / z2, im.height / z2
        cx, cy = im.width / 2 + dx * .4 * im.width, im.height / 2 + dy * .4 * im.height
        vw, vh = min(vw, im.width), min(vh, im.height)
        cx = min(max(cx, vw / 2), im.width - vw / 2); cy = min(max(cy, vh / 2), im.height - vh / 2)
    box = (cx - vw / 2, cy - vh / 2, cx + vw / 2, cy + vh / 2)
    if rot:
        im = im.rotate(rot, resample=Image.BILINEAR, center=(cx, cy))
    return im.transform((W, H), Image.EXTENT, box, Image.BILINEAR)


def fall_frame(dt):
    """Falling down a shaft: light streaks rushing up, slowing as she lands."""
    im = Image.new("RGB", (W, H), (8, 10, 14))
    d = ImageDraw.Draw(im)
    rng = random.Random(9)
    speed = 2600 * max(.15, 1 - dt / 3.2)
    for i in range(70):
        x = rng.randint(0, W)
        y = (rng.randint(0, H * 3) - dt * speed * rng.uniform(.6, 1.4)) % (H + 400) - 200
        ln = rng.randint(60, 260)
        c = rng.choice([(200, 210, 220), (60, 140, 255), (255, 130, 40)])
        d.line((x, y, x, y + ln), fill=c, width=rng.randint(1, 4))
    return im.filter(ImageFilter.GaussianBlur(1.2))


# ------------------------------------------------------------------ lyrics
def parse_srt(p):
    lines, who = [], "her"
    for b in p.read_text().strip().split("\n\n"):
        l = b.split("\n")
        if len(l) < 3:
            continue
        a, z = [sum(float(x) * m for x, m in zip(ts.replace(",", ".").split(":"), (3600, 60, 1))) for ts in l[1].split(" --> ")]
        txt = " ".join(l[2:])
        if txt.startswith("["):
            if "GLaDOS" in txt: who = "ai"
            elif "Chell" in txt: who = "her"
            continue
        lines.append((a, z, txt, who))
    return lines


LYR = parse_srt(MV / "embedded.srt")
LYR = [(a, min(z, a + 7.0), t, w) for a, z, t, w in LYR]


def wrap(text, font, maxw):
    words, rows, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if font.getlength(test) > maxw and cur:
            rows.append(cur); cur = w
        else:
            cur = test
    return rows + [cur]


def draw_lyrics(im, t):
    d = ImageDraw.Draw(im, "RGBA")
    for a, z, text, who in LYR:
        if not (a - .05 <= t < z + .25):
            continue
        out = 1 - min(1, max(0, (t - z) / .25))
        if who == "ai":
            n = int(len(text) * min(1, (t - a) / min(1.6, (z - a) * .6)))
            shown = text[:n].upper()
            rows = wrap(text.upper(), F_AI, W - 220)
            y = 70
            k = 0
            for r in rows:
                part = shown[k:k + len(r)]; k += len(r) + 1
                d.text((90 + 2, y + 2), part, font=F_AI, fill=(0, 0, 0, int(180 * out)))
                d.text((90, y), part, font=F_AI, fill=AMBER + (int(255 * out),))
                y += 50
        else:
            words = text.split()
            rev = min(z - a, 3.2) * .9
            rows = wrap(text, F_HER, W - 200)
            y = H - 80 - 72 * len(rows)
            wi = 0
            for r in rows:
                rw = F_HER.getlength(r)
                x = (W - rw) / 2
                for w in r.split():
                    wt = a + rev * wi / max(1, len(words))
                    al = min(1, max(0, (t - wt) / .18)) * out
                    if al > 0:
                        d.text((x + 3, y + 3), w, font=F_HER, fill=(0, 0, 0, int(170 * al)))
                        d.text((x, y), w, font=F_HER, fill=WHITE + (int(255 * al),))
                    x += F_HER.getlength(w + " ")
                    wi += 1
                y += 72


def title(im, t):
    if 0.6 < t < 3.6:
        al = min(1, (t - .6) / .5, (3.6 - t) / .4)
        d = ImageDraw.Draw(im, "RGBA")
        s = "NOT TODAY"
        d.text(((W - F_TITLE.getlength(s)) / 2, H / 2 - 95), s, font=F_TITLE, fill=(250, 250, 250, int(255 * al)))


# ------------------------------------------------------------------ render
proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                         "-i", "-", "-ss", str(T0), "-t", str(T1 - T0), "-i", AUDIO, "-map", "0:v", "-map", "1:a",
                         "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
                         "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
vign = np.ones((H, W, 1), np.float32)
yy, xx = np.mgrid[0:H, 0:W]
vign[..., 0] = 1 - .45 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 1.4
vign = np.clip(vign, .35, 1)
n0, n1 = int(T0 * FPS), int(T1 * FPS)
for n in range(n0, n1):
    t = n / FPS
    im = frame_bg(t)
    a = np.asarray(im).astype(np.float32) * vign
    # fades and flashes
    if t < 1.0: a *= t
    if t > 174.5: a *= max(0, (177.16 - t) / 2.66)
    for f in FLASH_AT:
        if 0 <= t - f < .12:
            a = a + (255 - a) * (1 - (t - f) / .12) * .55
    if 162.77 <= t < 171.2:  # the machine fades
        a *= 1 - .6 * (t - 162.77) / 8.45
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    title(im, t)
    draw_lyrics(im, t)
    proc.stdin.write(im.tobytes())
    if n % 300 == 0:
        print(f"{t:6.1f}s", flush=True)
proc.stdin.close()
proc.wait()
print("done", OUT)
