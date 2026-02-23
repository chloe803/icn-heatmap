from pathlib import Path
import io
import math

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# =============================
# Files (repo root)
# =============================
BASE_DIR = Path(__file__).resolve().parent
NPZ_PATH = BASE_DIR / "frames_cache_1min_grid15px_roll3_WITH_META.npz"
FLOORPLAN_PATH = BASE_DIR / "ICN_Airport_3F (1).png"

# =============================
# Visual params
# =============================
CMAP_NAME = "jet"
ALPHA_MAX = 0.85
ALPHA_GAMMA = 0.6
ALPHA_CUTOFF = 0.02

MAX_DISPLAY_WIDTH = 1200
JPEG_QUALITY = 80
TICK_MS = 180

# =============================
# Utils
# =============================
def minute_to_hhmm(m: int) -> str:
    h = (m // 60) % 24
    mm = m % 60
    return f"{h:02d}:{mm:02d}"

def fmt_time(i: int, time_bin_min: int) -> str:
    s = i * time_bin_min
    e = (i + 1) * time_bin_min
    return f"{s//60:02d}:{s%60:02d} ~ {e//60:02d}:{e%60:02d}"

@st.cache_resource
def load_npz(path: Path):
    z = np.load(str(path), allow_pickle=True)
    frames = z["frames"].astype(np.float32)  # (T,Hg,Wg)
    meta = {k: z[k] for k in z.files}
    return frames, meta

@st.cache_resource
def load_floorplan_scaled(path: Path, max_width: int):
    img = Image.open(path).convert("RGBA")
    W, H = img.size
    if W <= max_width:
        return img, 1.0
    scale = max_width / float(W)
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))
    return img.resize((new_w, new_h), resample=Image.BILINEAR), scale

@st.cache_resource
def get_cmap_lut(name: str):
    cmap = cm.get_cmap(name, 256)
    lut = (cmap(np.arange(256)) * 255).astype(np.uint8)  # (256,4)
    return lut

@st.cache_resource
def make_colorbar_png(cmap_name: str, vmin: float, vmax: float, label: str = "count (per 1-min bin)"):
    fig, ax = plt.subplots(figsize=(1.05, 4.8), dpi=160)
    fig.subplots_adjust(left=0.55, right=0.95)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_name), cax=ax)
    cb.set_label(label)
    ax.tick_params(labelsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def frame_to_overlay_rgba(grid: np.ndarray, vmax: float, lut_rgba: np.ndarray) -> np.ndarray:
    mask = grid <= 0
    norm = np.clip(grid / vmax, 0, 1)

    alpha = (norm ** ALPHA_GAMMA) * ALPHA_MAX
    alpha = np.where(norm < ALPHA_CUTOFF, 0.0, alpha)
    alpha[mask] = 0.0

    idx = (norm * 255).astype(np.uint8)
    rgba = lut_rgba[idx].copy()
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    return rgba

def paste_overlay_on_floorplan_safe(floor: Image.Image, overlay_rgba: np.ndarray, heat_extent_scaled: list):
    out = floor.copy()
    x_left, x_right, y_bottom, y_top = heat_extent_scaled

    left = int(round(x_left)); right = int(round(x_right))
    top = int(round(y_top)); bottom = int(round(y_bottom))

    W, H = out.size
    left2 = max(0, min(W, left)); right2 = max(0, min(W, right))
    top2 = max(0, min(H, top)); bottom2 = max(0, min(H, bottom))

    target_w = max(1, right2 - left2)
    target_h = max(1, bottom2 - top2)

    overlay_img = Image.fromarray(overlay_rgba, mode="RGBA").resize((target_w, target_h), resample=Image.BILINEAR)
    mask = overlay_img.split()[-1]
    out.paste(overlay_img, (left2, top2), mask=mask)
    return out

def draw_time_badge(img: Image.Image, text: str):
    out = img.copy()
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()

    pad_x, pad_y = 14, 8
    x, y = 16, 16
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    box = (x, y, x + tw + pad_x * 2, y + th + pad_y * 2)
    d.rounded_rectangle(box, radius=12, fill=(0, 0, 0, 150))
    d.text((x + pad_x, y + pad_y), text, fill=(255, 255, 255, 255), font=font)
    return out

def to_jpeg_bytes(img_rgb: Image.Image, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

# =============================
# Streamlit
# =============================
st.set_page_config(layout="wide")
st.title("ICN Heatmap")

if not NPZ_PATH.exists():
    st.error(f"NPZ 파일이 없습니다: {NPZ_PATH.name}")
    st.stop()
if not FLOORPLAN_PATH.exists():
    st.error(f"도면 파일이 없습니다: {FLOORPLAN_PATH.name}")
    st.stop()

frames, meta = load_npz(NPZ_PATH)
floor_small, scale = load_floorplan_scaled(FLOORPLAN_PATH, MAX_DISPLAY_WIDTH)

T, Hg, Wg = frames.shape
GRID_PX = int(meta["GRID_PX"])
TIME_BIN_MIN = int(meta["TIME_BIN_MIN"])
vmax = float(meta["vmax"])

lut = get_cmap_lut(CMAP_NAME)
cbar_png = make_colorbar_png(CMAP_NAME, 0.0, vmax)

heat_extent = [
    int(meta["min_gx"]) * GRID_PX,
    (int(meta["max_gx"]) + 1) * GRID_PX,
    (int(meta["max_gy"]) + 1) * GRID_PX,
    int(meta["min_gy"]) * GRID_PX,
]
heat_extent_scaled = [v * scale for v in heat_extent]

# -------------------------
# State (위젯 key와 겹치는 값은 직접 대입하지 않음!)
# -------------------------
if "playing" not in st.session_state:
    st.session_state.playing = False
if "pos" not in st.session_state:
    st.session_state.pos = 540.0
if "skip_once" not in st.session_state:
    st.session_state.skip_once = False
if "play_seq" not in st.session_state:
    st.session_state.play_seq = 0

def on_play():
    st.session_state.playing = True
    st.session_state.skip_once = True
    st.session_state.play_seq += 1

def on_pause():
    st.session_state.playing = False
    st.session_state.skip_once = True

def on_reset():
    st.session_state.playing = False
    # start_min 위젯 값은 st.session_state["start_min"]에 있음
    st.session_state.pos = float(st.session_state.get("start_min", 0))
    st.session_state.skip_once = True

# -------------------------
# UI
# -------------------------
st.markdown("## ⏰ Time Range")

# ✅ 위젯은 변수로만 받기 (session_state에 직접 대입 X)
start = st.slider("Start Time", 0, T - 1, 540, key="start_min")
end = st.slider("End Time", 0, T - 1, 600, key="end_min")
if start > end:
    start, end = end, start

speed = st.slider("Speed", 0.5, 6.0, 2.0, 0.25, key="speed")

# clamp pos
st.session_state.pos = float(max(start, min(st.session_state.pos, end)))

big1, big2 = st.columns(2)
with big1:
    st.markdown(f"<div style='text-align:center; font-size:44px;'>⏰ {minute_to_hhmm(int(start))}</div>", unsafe_allow_html=True)
with big2:
    st.markdown(f"<div style='text-align:center; font-size:44px;'>⏰ {minute_to_hhmm(int(end))}</div>", unsafe_allow_html=True)

if not st.session_state.playing:
    picked = st.slider("Minute (현재 시각)", int(start), int(end), int(round(st.session_state.pos)), key="pos_pick")
    st.session_state.pos = float(picked)
else:
    st.info(f"Playing... 현재 프레임: {int(st.session_state.pos)} ({minute_to_hhmm(int(st.session_state.pos))})")

b1, b2, b3 = st.columns([1.2, 1.2, 7.6])
with b1:
    st.button("▶ Play", use_container_width=True, on_click=on_play)
with b2:
    st.button("⏸ Pause", use_container_width=True, on_click=on_pause)
with b3:
    st.button("🔄 Reset", use_container_width=True, on_click=on_reset)

st.divider()

# -------------------------
# Playback loop
# -------------------------
if st.session_state.playing:
    st_autorefresh(interval=TICK_MS, key=f"loop_{st.session_state.play_seq}")

if st.session_state.skip_once:
    st.session_state.skip_once = False
else:
    if st.session_state.playing:
        st.session_state.pos += float(speed)
        if st.session_state.pos >= float(end):
            st.session_state.pos = float(end)
            st.session_state.playing = False

# -------------------------
# Smooth render
# -------------------------
pos = float(st.session_state.pos)
i0 = int(math.floor(pos))
i0 = max(int(start), min(i0, int(end)))
i1 = min(i0 + 1, int(end))
frac = float(pos - i0) if i1 != i0 else 0.0

grid0 = frames[i0]
if frac > 0.0 and i1 != i0:
    grid1 = frames[i1]
    grid = (1.0 - frac) * grid0 + frac * grid1
else:
    grid = grid0

title_text = fmt_time(i0, TIME_BIN_MIN)

overlay_rgba = frame_to_overlay_rgba(grid, vmax=vmax, lut_rgba=lut)
composed = paste_overlay_on_floorplan_safe(floor_small, overlay_rgba, heat_extent_scaled)
composed = draw_time_badge(composed, title_text)
img_bytes = to_jpeg_bytes(composed.convert("RGB"), quality=JPEG_QUALITY)

left, right = st.columns([8, 1])
with left:
    st.image(img_bytes, use_container_width=True)
with right:
    st.image(cbar_png, use_container_width=True)
