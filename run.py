# run.py
from pathlib import Path
import io
import math

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import matplotlib.cm as cm
from streamlit_autorefresh import st_autorefresh

# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
NPZ_PATH = BASE_DIR / "frames_cache_1min_grid15px_roll3_WITH_META.npz"
FLOORPLAN_PATH = BASE_DIR / "ICN_Airport_3F (1).png"

CMAP_NAME = "jet"
ALPHA_MAX = 0.85
ALPHA_GAMMA = 0.6
ALPHA_CUTOFF = 0.02

MAX_DISPLAY_WIDTH = 1200
JPEG_QUALITY = 82

# 깜빡임 줄이기 (180ms는 너무 빠름)
TICK_MS = 500


def minute_to_hhmm(m: int) -> str:
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"


def fmt_time(i: int, time_bin_min: int) -> str:
    s = i * time_bin_min
    e = (i + 1) * time_bin_min
    return f"{s//60:02d}:{s%60:02d} ~ {e//60:02d}:{e%60:02d}"


@st.cache_resource
def load_npz(path: Path):
    z = np.load(str(path), allow_pickle=True)
    frames = z["frames"].astype(np.float32)
    meta = {k: z[k] for k in z.files if k != "frames"}
    return frames, meta


@st.cache_resource
def load_floorplan_scaled(path: Path, max_width: int):
    img = Image.open(path).convert("RGBA")
    W, H = img.size
    if W <= max_width:
        return img, 1.0
    s = max_width / float(W)
    Ws, Hs = int(round(W * s)), int(round(H * s))
    return img.resize((Ws, Hs), resample=Image.BILINEAR), s


@st.cache_resource
def get_lut(name: str):
    cmap = cm.get_cmap(name, 256)
    return (cmap(np.arange(256)) * 255).astype(np.uint8)


def grid_to_rgba(grid: np.ndarray, vmax: float, lut: np.ndarray) -> np.ndarray:
    # vmax 안전장치
    if vmax <= 0:
        vmax = float(np.max(grid)) if np.max(grid) > 0 else 1.0

    norm = np.clip(grid / vmax, 0.0, 1.0)

    alpha = (norm ** ALPHA_GAMMA) * ALPHA_MAX
    alpha = np.where(norm < ALPHA_CUTOFF, 0.0, alpha)

    idx = (norm * 255).astype(np.uint8)
    rgba = lut[idx].copy()
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    return rgba


def paste_extent(base: Image.Image, overlay_rgba: np.ndarray, extent_scaled):
    """
    extent_scaled = [xL, xR, yT, yB] (scaled floorplan pixel coords)
    """
    out = base.copy()
    xL, xR, yT, yB = extent_scaled

    left = int(round(xL))
    right = int(round(xR))
    top = int(round(yT))
    bottom = int(round(yB))

    W, H = out.size
    left = max(0, min(W, left))
    right = max(0, min(W, right))
    top = max(0, min(H, top))
    bottom = max(0, min(H, bottom))

    # 순서 안전장치
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top

    w = max(1, right - left)
    h = max(1, bottom - top)

    ov = Image.fromarray(overlay_rgba, "RGBA").resize((w, h), resample=Image.BILINEAR)
    out.paste(ov, (left, top), mask=ov.split()[-1])
    return out


def to_jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()


# -----------------------------
# UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("ICN Heatmap")

# 이미지 자리를 고정(깜빡임 감소)
img_slot = st.empty()

# Load data
frames, meta = load_npz(NPZ_PATH)
floor, scale = load_floorplan_scaled(FLOORPLAN_PATH, MAX_DISPLAY_WIDTH)

T = int(frames.shape[0])

# meta values (np scalar -> python)
GRID_PX = int(meta.get("GRID_PX", 15))
TIME_BIN_MIN = int(meta.get("TIME_BIN_MIN", 1))
vmax = float(meta.get("vmax", np.max(frames) if frames.size else 1.0))

lut = get_lut(CMAP_NAME)

min_gx = int(meta.get("min_gx", 0))
max_gx = int(meta.get("max_gx", frames.shape[2] - 1 if frames.ndim == 3 else 0))
min_gy = int(meta.get("min_gy", 0))
max_gy = int(meta.get("max_gy", frames.shape[1] - 1 if frames.ndim == 3 else 0))

# Extent in original floorplan pixels (unscaled)
xL = min_gx * GRID_PX
xR = (max_gx + 1) * GRID_PX
yT = min_gy * GRID_PX
yB = (max_gy + 1) * GRID_PX

extent = [xL, xR, yT, yB]
extent_scaled = [v * scale for v in extent]

# Session state
if "playing" not in st.session_state:
    st.session_state.playing = False
if "pos" not in st.session_state:
    st.session_state.pos = 540.0
if "prev_range" not in st.session_state:
    st.session_state.prev_range = None

# ✅ Start/End 두 개 대신 "범위 슬라이더" 하나로 (UI 안정 + 숫자 표시 문제 회피)
default_start = min(540, T - 1)
default_end = min(600, T - 1)
start, end = st.slider(
    "Time Range (index)",
    0,
    T - 1,
    (default_start, default_end),
    key="range_min",
)

start = int(start)
end = int(end)
if start > end:
    start, end = end, start

speed = st.slider("Speed", 0.5, 6.0, 2.0, 0.25, key="speed")

# ✅ 슬라이더 아래에 값이 안 보이더라도, 우리가 확실히 표시
st.caption(
    f"선택 범위: {minute_to_hhmm(start * TIME_BIN_MIN)} ~ {minute_to_hhmm(end * TIME_BIN_MIN)} "
    f"(index {start} ~ {end})"
)

# 범위 바뀌면 재생 상태 초기화
cur_range = (start, end)
if st.session_state.prev_range != cur_range:
    st.session_state.prev_range = cur_range
    st.session_state.playing = False
    st.session_state.pos = float(start)

# pos clamp
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
    picked = st.slider(
        "Minute (현재 시각)",
        start,
        end,
        int(round(st.session_state.pos)),
        key="pos_pick",
    )
    st.session_state.pos = float(picked)

pos = float(st.session_state.pos)
i0 = int(math.floor(pos))
i0 = max(start, min(i0, end))

grid = frames[i0]
overlay = grid_to_rgba(grid, vmax, lut)
img = paste_extent(floor, overlay, extent_scaled)

d = ImageDraw.Draw(img)
d.text((16, 16), fmt_time(i0, TIME_BIN_MIN), fill=(255, 255, 255, 255))

# ✅ 고정 슬롯에 업데이트 (깜빡임 감소)
img_slot.image(to_jpeg(img), use_container_width=True)
