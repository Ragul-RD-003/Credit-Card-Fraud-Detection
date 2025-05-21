import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from skimage import color
from io import BytesIO
import base64
import colorsys

st.set_page_config(page_title="Unique Smart Color Detector", layout="wide", initial_sidebar_state="expanded")

# ========== STYLES ==========
st.markdown("""
<style>
/* Dark mode base */
body {
    background-color: #121212;
    color: #e0e0e0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
/* Sidebar */
.css-1d391kg {
    background: #1f1f1f !important;
    color: #ddd !important;
    border-right: 2px solid #333;
}
/* Sidebar header */
h2 {
    color: #61dafb;
    font-weight: 700;
    margin-bottom: 12px;
}
/* Main container cards */
.card {
    background: #1e1e1e;
    padding: 24px;
    margin-bottom: 24px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(97, 218, 251, 0.25);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    user-select:none;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 16px 40px rgba(97, 218, 251, 0.45);
}

/* Big color preview block */
.color-preview {
    border-radius: 20px;
    padding: 40px 0;
    text-align: center;
    font-weight: 900;
    font-size: 2.5rem;
    letter-spacing: 1.6px;
    box-shadow: 0 8px 20px rgba(97, 218, 251, 0.6);
    margin-bottom: 16px;
    user-select:none;
}

/* Text small */
.text-small {
    font-size: 1.1rem;
    font-weight: 500;
    color: #a0a0a0;
}

/* Swatch */
.swatch {
    height: 44px;
    border-radius: 12px;
    box-shadow: 0 3px 12px rgba(97, 218, 251, 0.8);
    margin: 6px 0;
    border: 1.8px solid #3a3a3a;
    transition: transform 0.3s ease;
    cursor: pointer;
}
.swatch:hover {
    transform: scale(1.2);
}

/* Contrast preview boxes */
.contrast-box {
    padding: 18px;
    border-radius: 12px;
    margin-top: 12px;
    font-weight: 700;
    text-align: center;
    box-shadow: 0 6px 22px rgba(97, 218, 251, 0.5);
    user-select:none;
}

/* Zoomed image with smooth scale on hover */
.zoomed {
    border-radius: 16px;
    box-shadow: 0 6px 24px rgba(97, 218, 251, 0.4);
    transition: transform 0.35s ease;
    user-select:none;
}
.zoomed:hover {
    transform: scale(1.3);
    cursor: zoom-in;
}

/* Crosshair pointer styling (white with glow) */
.crosshair {
    stroke: white;
    stroke-width: 3;
    filter: drop-shadow(0 0 5px #61dafb);
}

/* Label */
.label {
    font-weight: 700;
    font-size: 1.3rem;
    margin-bottom: 8px;
    color: #61dafb;
}

</style>
""", unsafe_allow_html=True)

# ========== LOAD COLORS ==========
@st.cache_data
def load_colors():
    df = pd.read_csv("colors.csv")
    df["LAB"] = df.apply(lambda r: color.rgb2lab(
        np.array([[r[["R","G","B"]]]], dtype=np.uint8)/255.0)[0][0], axis=1)
    return df

colors = load_colors()

# ========== FUNCTIONS ==========
def rgb_to_lab(rgb):
    return color.rgb2lab(np.array([[rgb]], dtype=np.uint8)/255.0)[0][0]

def get_closest_colors(R, G, B, top_n=5):
    input_lab = rgb_to_lab([R, G, B])
    distances = colors["LAB"].apply(lambda lab: np.linalg.norm(input_lab - lab))
    closest = colors.loc[distances.nsmallest(top_n).index]
    closest = closest.assign(DeltaE=distances[closest.index])
    return closest

def get_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def pil_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def brightness(rgb):
    r, g, b = rgb
    return np.sqrt(0.299*r**2 + 0.587*g**2 + 0.114*b**2)

def saturation(rgb):
    r, g, b = [x/255 for x in rgb]
    _, s, _ = colorsys.rgb_to_hsv(r, g, b)
    return s

def text_contrast(rgb):
    def luminance(c):
        c = c/255
        return c/12.92 if c <= 0.03928 else ((c+0.055)/1.055)**2.4

    L = 0.2126*luminance(rgb[0]) + 0.7152*luminance(rgb[1]) + 0.0722*luminance(rgb[2])
    white_contrast = (1.05) / (L + 0.05)
    black_contrast = (L + 0.05) / 0.05
    return white_contrast, black_contrast

def classify_tone(rgb):
    b = brightness(rgb)
    s = saturation(rgb)
    if b > 180:
        tone = "Light"
    elif b < 80:
        tone = "Dark"
    else:
        tone = "Medium"

    if s > 0.5:
        tone += " & Vibrant"
    elif s < 0.25:
        tone += " & Muted"
    else:
        tone += " & Neutral"

    return tone

# ========== MAIN APP ==========
st.title("Unique Smart Color Detection")

uploaded = st.sidebar.file_uploader("Upload your image (png, jpg)", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # Sidebar controls
    st.sidebar.markdown("<div class='label'>Select Pixel Coordinates</div>", unsafe_allow_html=True)
    x = st.sidebar.slider("X", 2, w - 3, w // 2)
    y = st.sidebar.slider("Y", 2, h - 3, h // 2)

    # Extract 5x5 pixel region
    region = img_np[y-2:y+3, x-2:x+3]
    avg_color = region.mean(axis=(0, 1)).astype(int)
    R, G, B = avg_color

    # Mark the pixel with crosshair on image
    marked = img.copy()
    draw = ImageDraw.Draw(marked)
    pointer_r = 8
    draw.ellipse((x - pointer_r, y - pointer_r, x + pointer_r, y + pointer_r), fill='transparent', outline='#61dafb', width=4)
    draw.line((x - pointer_r - 6, y, x + pointer_r + 6, y), fill='#61dafb', width=3)
    draw.line((x, y - pointer_r - 6, x, y + pointer_r + 6), fill='#61dafb', width=3)

    # Layout
    col_img, col_info = st.columns([3, 2])

    with col_img:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(marked, caption="Image with Marked Pixel", use_column_width=True)
        zoom_img = Image.fromarray(region).resize((140, 140), Image.NEAREST)
        b64_zoom = pil_to_base64(zoom_img)
        st.markdown(f'<img src="data:image/png;base64,{b64_zoom}" class="zoomed" alt="Zoomed region">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        hex_val = get_hex((R, G, B))
        lab_val = rgb_to_lab([R, G, B])
        top_colors = get_closest_colors(R, G, B, top_n=5)
        main_color_name = top_colors.iloc[0]['color_name']

        # Color preview block
        text_color = 'white' if brightness((R, G, B)) < 130 else '#121212'
        st.markdown(f"""
            <div class="color-preview" style="background-color:{hex_val}; color:{text_color};">
                {main_color_name}<br>
                <span class="text-small">{hex_val} | RGB: ({R}, {G}, {B})</span>
            </div>
        """, unsafe_allow_html=True)

        # Tone and contrast
        tone = classify_tone((R, G, B))
        white_contrast, black_contrast = text_contrast((R, G, B))
        recommended_text = "White" if white_contrast > black_contrast else "Black"
        best_contrast = max(white_contrast, black_contrast)

        st.markdown(f"### Color Tone")
        st.write(f"{tone}")

        st.markdown(f"### Text Contrast Recommendation")
        st.write(f"Use **{recommended_text}** text on this color for best readability (Contrast Ratio: {best_contrast:.2f})")

        st.markdown("### Top 5 Closest Colors")
        for _, row in top_colors.iterrows():
            c_hex = get_hex((int(row["R"]), int(row["G"]), int(row["B"])))
            cname = row["color_name"]
            delta_e = row["DeltaE"]
            st.markdown(f"""
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                    <div class="swatch" style="background-color:{c_hex}; width:40px;"></div>
                    <div>{cname} — ΔE: {delta_e:.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        # Complementary color
        comp_rgb = [255 - R, 255 - G, 255 - B]
        comp_hex = get_hex(comp_rgb)
        st.markdown("### Complementary Color")
        st.markdown(f'<div class="swatch" style="background-color:{comp_hex}; width:50px; margin-bottom:8px;"></div>', unsafe_allow_html=True)
        st.markdown(f"HEX: {comp_hex}")

        # Contrast preview boxes
        st.markdown("### Contrast Preview")
        st.markdown(f'<div class="contrast-box" style="background-color:{hex_val}; color:white;">White Text</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="contrast-box" style="background-color:{hex_val}; color:black;">Black Text</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.sidebar.info("Upload an image to get started.")
    st.markdown("""
    <div style="margin-top:80px; text-align:center; font-size:1.5rem; color:#888;">
        Upload an image from the sidebar<br>to detect and analyze colors.
    </div>
    """, unsafe_allow_html=True)
