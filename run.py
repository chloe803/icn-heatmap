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
TICK_MS = 350  # 필요하면 500~800으로 올려도 됨

# =============================
# Utils
# =============================
def minute_to_hhmm(m: int) -> str:
    h = (m // 60) % 24
    mm = m % 60
    return f"{h:02d}:{mm:02d}"

def idx_to_hhmm(idx: int, time_bin_min: int) -> str:
    return minute_to_hhmm(int(idx) * int(time_bin_min))

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

def _load_font(size: int):
    for name in ["arial.ttf", "AppleGothic.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except:
            pass
    return ImageFont.load_default()

def draw_badge(d: ImageDraw.ImageDraw, xy, text, font, pad=(14, 8), radius=12):
    x, y = xy
    pad_x, pad_y = pad
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    box = (x, y, x + tw + pad_x * 2, y + th + pad_y * 2)
    d.rounded_rectangle(box, radius=radius, fill=(0, 0, 0, 150))
    d.text((x + pad_x, y + pad_y), text, fill=(255, 255, 255, 255), font=font)

def draw_time_overlays(img: Image.Image, cur_text: str, start_text: str, end_text: str):
    out = img.copy()
    d = ImageDraw.Draw(out, "RGBA")

    font_big = _load_font(28)
    font_mid = _load_font(22)

    draw_badge(d, (16, 16), cur_text, font_big)

    W, _ = out.size
    start_label = f"START {start_text}"
    end_label = f"END {end_text}"

    bbox1 = d.textbbox((0, 0), start_label, font=font_mid)
    bbox2 = d.textbbox((0, 0), end_label, font=font_mid)
    w1 = (bbox1[2] - bbox1[0]) + 14 * 2
    w2 = (bbox2[2] - bbox2[0]) + 14 * 2

    x_start = max(16, W - w1 - 16)
    x_end = max(16, W - w2 - 16)

    draw_badge(d, (x_start, 16), start_label, font_mid)
    draw_badge(d, (x_end, 16 + 44), end_label, font_mid)

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
# State (안전한 구조)
# -------------------------
if "playing" not in st.session_state:
    st.session_state.playing = False

# ✅ 재생 위치(프로그램이 바꿔도 되는 값)
if "pos_val" not in st.session_state:
    st.session_state.pos_val = 540.0

# ✅ 사용자 드래그용(슬라이더가 소유하는 값)
if "pos_pick" not in st.session_state:
    st.session_state.pos_pick = int(round(st.session_state.pos_val))

# 버튼 클릭 직후 1회는 전진 건너뛰기
if "skip_once" not in st.session_state:
    st.session_state.skip_once = False

# 버튼 클릭이 있던 rerun에서는 슬라이더가 pos_val을 덮어쓰지 못하게
if "ui_locked" not in st.session_state:
    st.session_state.ui_locked = False

# -------------------------
# UI (포맷 유지)
# -------------------------
st.markdown("## ⏰ Time Range")

start = st.slider("Start Time", 0, T - 1, 540, key="start_min")
end = st.slider("End Time", 0, T - 1, 600, key="end_min")
if start > end:
    start, end = end, start

speed = st.slider("Speed", 0.5, 6.0, 2.0, 0.25, key="speed")

# clamp pos_val
st.session_state.pos_val = float(max(start, min(st.session_state.pos_val, end)))

# 큰 시간 표시
big1, big2 = st.columns(2)
with big1:
    st.markdown(
        f"<div style='text-align:center; font-size:44px;'>⏰ START {idx_to_hhmm(int(start), TIME_BIN_MIN)}</div>",
        unsafe_allow_html=True,
    )
with big2:
    st.markdown(
        f"<div style='text-align:center; font-size:44px;'>⏰ END {idx_to_hhmm(int(end), TIME_BIN_MIN)}</div>",
        unsafe_allow_html=True,
    )

# 버튼
b1, b2, b3 = st.columns([1.2, 1.2, 7.6])
with b1:
    play_clicked = st.button("▶ Play", use_container_width=True, key="btn_play")
with b2:
    pause_clicked = st.button("⏸ Pause", use_container_width=True, key="btn_pause")
with b3:
    reset_clicked = st.button("🔄 Reset", use_container_width=True, key="btn_reset")

# ✅ 버튼 처리 (Reset > Pause > Play)
st.session_state.ui_locked = False
if reset_clicked:
    st.session_state.playing = False
    st.session_state.pos_val = float(start)
    st.session_state.pos_pick = int(round(st.session_state.pos_val))
    st.session_state.skip_once = True
    st.session_state.ui_locked = True

elif pause_clicked:
    st.session_state.playing = False
    # pos_val 유지 (멈춘 자리)
    st.session_state.pos_pick = int(round(st.session_state.pos_val))
    st.session_state.skip_once = True
    st.session_state.ui_locked = True

elif play_clicked:
    st.session_state.playing = True
    # ✅ 멈춘 자리에서 그대로 재개: pos_val 건드리지 않음
    st.session_state.pos_pick = int(round(st.session_state.pos_val))
    st.session_state.skip_once = True
    st.session_state.ui_locked = True

# Minute 슬라이더 (항상 같은 자리)
picked = st.slider(
    "Minute (현재 시각)",
    int(start),
    int(end),
    int(round(st.session_state.pos_val)),
    key="pos_pick",
    disabled=st.session_state.playing,
)

# 사용자가 드래그한 경우에만 pos_val 갱신 (버튼 클릭 rerun에서는 막음)
if (not st.session_state.playing) and (not st.session_state.ui_locked):
    st.session_state.pos_val = float(picked)

# 상태 텍스트
status_slot = st.empty()
status_slot.info(
    f"{'Playing...' if st.session_state.playing else 'Paused.'}  "
    f"현재 프레임: {int(float(st.session_state.pos_val))}  "
    f"({idx_to_hhmm(int(float(st.session_state.pos_val)), TIME_BIN_MIN)})"
)

st.divider()

# -------------------------
# Playback loop
# -------------------------
if st.session_state.playing:
    st_autorefresh(interval=TICK_MS, key="loop")

if st.session_state.skip_once:
    st.session_state.skip_once = False
else:
    if st.session_state.playing:
        st.session_state.pos_val += float(speed)
        if st.session_state.pos_val >= float(end):
            st.session_state.pos_val = float(end)
            st.session_state.playing = False

# -------------------------
# Smooth render
# -------------------------
pos = float(st.session_state.pos_val)
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

cur_text = fmt_time(i0, TIME_BIN_MIN)

overlay_rgba = frame_to_overlay_rgba(grid, vmax=vmax, lut_rgba=lut)
composed = paste_overlay_on_floorplan_safe(floor_small, overlay_rgba, heat_extent_scaled)

start_hhmm = idx_to_hhmm(int(start), TIME_BIN_MIN)
end_hhmm = idx_to_hhmm(int(end), TIME_BIN_MIN)
composed = draw_time_overlays(composed, cur_text, start_hhmm, end_hhmm)

img_bytes = to_jpeg_bytes(composed.convert("RGB"), quality=JPEG_QUALITY)

# Layout
left, right = st.columns([8, 1])
with left:
    st.image(img_bytes, use_container_width=True)
with right:
    st.image(cbar_png, use_container_width=True)
