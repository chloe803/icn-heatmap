# run.py
from pathlib import Path
import io
import math

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import matplotlib.cm as cm
from streamlit_autorefresh import st_autorefresh

# -------------------
# Paths / Config
# -------------------
BASE_DIR = Path(__file__).resolve().parent
NPZ_PATH = BASE_DIR / "frames_cache_1min_grid15px_roll3_WITH_META.npz"
FLOORPLAN_PATH = BASE_DIR / "ICN_Airport_3F (1).png"

CMAP_NAME = "jet"
ALPHA_MAX = 0.85
ALPHA_GAMMA = 0.6
ALPHA_CUTOFF = 0.02
MAX_DISPLAY_WIDTH = 1200
JPEG_QUALITY = 82

# Play 시 깜빡임 줄이기: 너무 짧으면 매 rerun이 눈에 보임
TICK_MS = 500  # 필요하면 700~900까지 올려도 됨


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

    # safety ordering
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


def draw_time_badge(img: Image.Image, text: str):
    """
    히트맵 위 시간표시가 안 보인다고 해서,
    흰 글씨 + 반투명 검은 박스(가독성)로 강제 표시.
    """
    d = ImageDraw.Draw(img, "RGBA")
    x, y = 16, 16

    # 대략적인 박스 크기(폰트 측정이 환경마다 달라서 안전하게 여유)
    pad_x, pad_y = 10, 6
    box_w = 8 * len(text) + pad_x * 2
    box_h = 18 + pad_y * 2

    d.rounded_rectangle(
        [x - pad_x, y - pad_y, x - pad_x + box_w, y - pad_y + box_h],
        radius=8,
        fill=(0, 0, 0, 140),
    )
    d.text((x, y), text, fill=(255, 255, 255, 255))


# -------------------
# App
# -------------------
st.set_page_config(layout="wide")
st.title("ICN Heatmap")

# ✅ 이미지 영역을 고정 슬롯으로 만들면 Play 시 깜빡임이 확 줄어듦
img_slot = st.empty()

frames, meta = load_npz(NPZ_PATH)
floor, scale = load_floorplan_scaled(FLOORPLAN_PATH, MAX_DISPLAY_WIDTH)

T = int(frames.shape[0])
GRID_PX = int(meta.get("GRID_PX", 15))
TIME_BIN_MIN = int(meta.get("TIME_BIN_MIN", 1))
vmax = float(meta.get("vmax", np.max(frames) if frames.size else 1.0))
lut = get_lut(CMAP_NAME)

min_gx = int(meta.get("min_gx", 0))
max_gx = int(meta.get("max_gx", frames.shape[2] - 1 if frames.ndim == 3 else 0))
min_gy = int(meta.get("min_gy", 0))
max_gy = int(meta.get("max_gy", frames.shape[1] - 1 if frames.ndim == 3 else 0))

# extent in unscaled floorplan pixels
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
if "prev_start" not in st.session_state:
    st.session_state.prev_start = None
if "prev_end" not in st.session_state:
    st.session_state.prev_end = None

# ---- Controls (원래 포맷 유지) ----
start = st.slider("Start Time", 0, T - 1, min(540, T - 1), key="start_min")
end = st.slider("End Time", 0, T - 1, min(600, T - 1), key="end_min")
speed = st.slider("Speed", 0.5, 6.0, 2.0, 0.25, key="speed")

start = int(start)
end = int(end)
if start > end:
    start, end = end, start

# ✅ 혹시 슬라이더 값 표시가 UI에서 안 보일 때 대비(확실히 보이게)
st.caption(
    f"선택 범위: {minute_to_hhmm(start * TIME_BIN_MIN)} ~ {minute_to_hhmm(end * TIME_BIN_MIN)} "
    f"(index {start} ~ {end})"
)

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

# ✅ 레이아웃 점프(깜빡임 체감)를 줄이려고 아래 슬라이더 자리도 고정
pos_slot = st.empty()

if st.session_state.playing:
    st_autorefresh(interval=TICK_MS, key="loop")
    st.session_state.pos += float(speed)
    if st.session_state.pos >= float(end):
        st.session_state.pos = float(end)
        st.session_state.playing = False

    # Play 중에도 같은 자리에 "읽기전용 느낌"으로 표시(레이아웃 안정)
    with pos_slot:
        st.slider(
            "Minute (현재 시각)",
            start,
            end,
            int(round(st.session_state.pos)),
            key="pos_pick_play_readonly",
            disabled=True,
        )
else:
    with pos_slot:
        picked = st.slider(
            "Minute (현재 시각)",
            start,
            end,
            int(round(st.session_state.pos)),
            key="pos_pick",
        )
    st.session_state.pos = float(picked)

# ---- Render frame ----
pos = float(st.session_state.pos)
i0 = int(math.floor(pos))
i0 = max(start, min(i0, end))

grid = frames[i0]
overlay = grid_to_rgba(grid, vmax, lut)
img = paste_extent(floor, overlay, extent_scaled)

# ✅ 시간 표시(가독성 배지)
draw_time_badge(img, fmt_time(i0, TIME_BIN_MIN))

# ✅ 고정 슬롯에만 업데이트 (Play 깜빡임 대폭 감소)
img_slot.image(to_jpeg(img), use_container_width=True)
