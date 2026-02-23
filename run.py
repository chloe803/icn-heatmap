from pathlib import Path
import io
import math

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import matplotlib.cm as cm
from streamlit_autorefresh import st_autorefresh

BASE_DIR = Path(file).resolve().parent
NPZ_PATH = BASE_DIR / "frames_cache_1min_grid15px_roll3_WITH_META.npz"
FLOORPLAN_PATH = BASE_DIR / "ICN_Airport_3F (1).png"

CMAP_NAME = "jet"
ALPHA_MAX = 0.85
ALPHA_GAMMA = 0.6
ALPHA_CUTOFF = 0.02
MAX_DISPLAY_WIDTH = 1200
JPEG_QUALITY = 82
TICK_MS = 180

def minute_to_hhmm(m):
return f"{(m//60)%24:02d}:{m%60:02d}"

def fmt_time(i, time_bin_min):
s = i * time_bin_min
e = (i + 1) * time_bin_min
return f"{s//60:02d}:{s%60:02d} ~ {e//60:02d}:{e%60:02d}"

@st.cache_resource
def load_npz(path):
z = np.load(str(path), allow_pickle=True)
frames = z["frames"].astype(np.float32)
meta = {k: z[k] for k in z.files}
return frames, meta

@st.cache_resource
def load_floorplan_scaled(path, max_width):
img = Image.open(path).convert("RGBA")
W, H = img.size
if W <= max_width:
return img, 1.0
s = max_width / float(W)
return img.resize((int(Ws), int(Hs))), s

@st.cache_resource
def get_lut(name):
cmap = cm.get_cmap(name, 256)
return (cmap(np.arange(256)) * 255).astype(np.uint8)

def grid_to_rgba(grid, vmax, lut):
norm = np.clip(grid / vmax, 0, 1)
alpha = (norm ** ALPHA_GAMMA) * ALPHA_MAX
alpha = np.where(norm < ALPHA_CUTOFF, 0.0, alpha)
idx = (norm * 255).astype(np.uint8)
rgba = lut[idx].copy()
rgba[..., 3] = (alpha * 255).astype(np.uint8)
return rgba

def paste_extent(base, overlay_rgba, extent_scaled):
out = base.copy()
xL, xR, yB, yT = extent_scaled
left = int(round(xL)); right = int(round(xR))
top = int(round(yT)); bottom = int(round(yB))
W, H = out.size
left = max(0, min(W, left)); right = max(0, min(W, right))
top = max(0, min(H, top)); bottom = max(0, min(H, bottom))
w = max(1, right-left); h = max(1, bottom-top)
ov = Image.fromarray(overlay_rgba, "RGBA").resize((w, h))
out.paste(ov, (left, top), mask=ov.split()[-1])
return out

def to_jpeg(img):
buf = io.BytesIO()
img.convert("RGB").save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
return buf.getvalue()

st.set_page_config(layout="wide")
st.title("ICN Heatmap")

frames, meta = load_npz(NPZ_PATH)
floor, scale = load_floorplan_scaled(FLOORPLAN_PATH, MAX_DISPLAY_WIDTH)

T = frames.shape[0]
GRID_PX = int(meta["GRID_PX"])
TIME_BIN_MIN = int(meta["TIME_BIN_MIN"])
vmax = float(meta["vmax"])
lut = get_lut(CMAP_NAME)

min_gx = int(meta["min_gx"]); max_gx = int(meta["max_gx"])
min_gy = int(meta["min_gy"]); max_gy = int(meta["max_gy"])

extent = [min_gx*GRID_PX, (max_gx+1)GRID_PX, (max_gy+1)GRID_PX, min_gyGRID_PX]
extent_scaled = [vscale for v in extent]

if "playing" not in st.session_state: st.session_state.playing = False
if "pos" not in st.session_state: st.session_state.pos = 540.0
if "prev_start" not in st.session_state: st.session_state.prev_start = None
if "prev_end" not in st.session_state: st.session_state.prev_end = None

start = st.slider("Start Time", 0, T-1, 540, key="start_min")
end = st.slider("End Time", 0, T-1, 600, key="end_min")
speed = st.slider("Speed", 0.5, 6.0, 2.0, 0.25, key="speed")

start = int(start); end = int(end)
if start > end: start, end = end, start

if st.session_state.prev_start != start or st.session_state.prev_end != end:
st.session_state.prev_start = start
st.session_state.prev_end = end
st.session_state.playing = False
st.session_state.pos = float(start)

st.session_state.pos = float(max(start, min(st.session_state.pos, end)))

c1, c2, c3 = st.columns([1.2, 1.2, 7.6])
if c1.button("▶ Play"):
st.session_state.playing = True
if c2.button("⏸ Pause"):
st.session_state.playing = False
if c3.button("🔄 Reset"):
st.session_state.playing = False
st.session_state.pos = float(start)

if st.session_state.playing:
st_autorefresh(interval=TICK_MS, key="loop")
st.session_state.pos += float(speed)
if st.session_state.pos >= float(end):
st.session_state.pos = float(end)
st.session_state.playing = False
else:
picked = st.slider("Minute (현재 시각)", start, end, int(round(st.session_state.pos)), key="pos_pick")
st.session_state.pos = float(picked)

pos = float(st.session_state.pos)
i0 = int(math.floor(pos))
i0 = max(start, min(i0, end))

grid = frames[i0]
overlay = grid_to_rgba(grid, vmax, lut)
img = paste_extent(floor, overlay, extent_scaled)

d = ImageDraw.Draw(img)
d.text((16, 16), fmt_time(i0, TIME_BIN_MIN), fill=(255, 255, 255, 255))

st.image(to_jpeg(img), use_container_width=True)
