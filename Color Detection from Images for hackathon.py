import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from skimage import color
from io import BytesIO
import base64
import colorsys

# ========== PAGE STYLE ==========
st.set_page_config(page_title="Smart Color Detector", layout="wide")
st.markdown("""
    <style>
    .color-box {
        text-align: center;
        padding: 18px;
        margin-top: 12px;
        font-size: 28px;
        font-weight: 700;
        border-radius: 12px;
        background: linear-gradient(90deg, #f0f0f5 0%, #dcdde1 100%);
        color: #222;
        box-shadow: 0 6px 14px rgba(0,0,0,0.15);
        user-select:none;
    }
    .swatch {
        height: 40px;
        border-radius: 10px;
        box-shadow: 1px 1px 8px rgba(0,0,0,0.3);
        margin: 5px 0;
        border: 1.5px solid #888;
    }
    .zoomed:hover {
        transform: scale(1.3);
        transition: 0.3s ease-in-out;
        cursor: zoom-in;
    }
    .contrast-box {
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ========== LOAD COLORS ==========
@st.cache_data
def load_colors():
    df = pd.read_csv("colors.csv")
    # Precompute LAB values once for speed
    df["LAB"] = df.apply(lambda r: color.rgb2lab(
        np.array([[r[["R","G","B"]]]], dtype=np.uint8)/255.0)[0][0], axis=1)
    return df

colors = load_colors()

# ========== HELPERS ==========
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
    # Perceived brightness (0-255)
    r, g, b = rgb
    return np.sqrt(0.299*r**2 + 0.587*g**2 + 0.114*b**2)

def saturation(rgb):
    # Convert RGB to HSV and get saturation
    r, g, b = [x/255 for x in rgb]
    _, s, _ = colorsys.rgb_to_hsv(r, g, b)
    return s

def text_contrast(rgb):
    # Calculate contrast ratio for black and white text
    # Using relative luminance formula
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

# ========== APP UI ==========
st.title("Smart Color Detection with Tone & Contrast Advisor")

uploaded = st.file_uploader("Upload an image (jpg, png)", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    col1, col2 = st.columns([4, 2])

    with col1:
        st.image(img, caption="Original Image", use_column_width=True)
        x = st.slider("X Coordinate", 2, w - 3, w // 2)
        y = st.slider("Y Coordinate", 2, h - 3, h // 2)

        # Average color from 5x5 region
        region = img_np[y-2:y+3, x-2:x+3]
        avg_color = region.mean(axis=(0, 1)).astype(int)
        R, G, B = avg_color

        # Mark pixel with crosshair
        marked = img.copy()
        draw = ImageDraw.Draw(marked)
        pointer_r = 7
        draw.ellipse((x - pointer_r, y - pointer_r, x + pointer_r, y + pointer_r), fill='red', outline='white', width=2)
        draw.line((x - pointer_r - 4, y, x + pointer_r + 4, y), fill='white', width=2)
        draw.line((x, y - pointer_r - 4, x, y + pointer_r + 4), fill='white', width=2)
        st.image(marked, caption="Marked Pixel", use_column_width=True)

        # Zoomed patch
        zoom = Image.fromarray(region).resize((120, 120), resample=Image.NEAREST)
        b64 = pil_to_base64(zoom)
        st.markdown(f"<img class='zoomed' src='data:image/png;base64,{b64}' width='120'>", unsafe_allow_html=True)

    with col2:
        st.subheader("Detected Color Overview")

        hex_val = get_hex((R, G, B))
        lab_val = rgb_to_lab([R, G, B])

        # Closest color names
        top_colors = get_closest_colors(R, G, B, top_n=5)
        main_color_name = top_colors.iloc[0]['color_name']

        # Big color preview + name
        st.markdown(f"""
            <div style="
                background-color:{hex_val}; 
                border-radius:15px; 
                padding:30px; 
                color: {'white' if brightness((R,G,B))<130 else 'black'};
                text-align:center;
                font-weight:800;
                font-size:28px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                user-select:none;
                ">
                {main_color_name}
                <br><small style='font-size:16px;'>{hex_val} | RGB: ({R}, {G}, {B})</small>
            </div>
        """, unsafe_allow_html=True)

        # Tone classification
        tone = classify_tone((R, G, B))
        st.markdown(f"**Color Tone:** {tone}")

        # Contrast ratios
        white_contrast, black_contrast = text_contrast((R, G, B))
        recommended_text = "White" if white_contrast > black_contrast else "Black"
        best_contrast = max(white_contrast, black_contrast)
        st.markdown(f"**Recommended Text Color:** {recommended_text} (Contrast Ratio: {best_contrast:.2f})")

        # Top 5 closest colors with swatches and ΔE
        st.subheader("Top 5 Closest Colors:")
        for _, row in top_colors.iterrows():
            c_hex = get_hex((int(row["R"]), int(row["G"]), int(row["B"])))
            cname = row["color_name"]
            delta_e = row["DeltaE"]
            swatch_html = f"<div class='swatch' style='background-color:{c_hex}'></div>"
            st.markdown(f"{swatch_html} **{cname}** - ΔE: {delta_e:.2f}", unsafe_allow_html=True)

        # Complementary color
        comp_rgb = [255 - R, 255 - G, 255 - B]
        comp_hex = get_hex(comp_rgb)
        st.subheader("Complementary Color:")
        st.markdown(f"<div class='swatch' style='background-color:{comp_hex}'></div>", unsafe_allow_html=True)
        st.markdown(f"HEX: {comp_hex}")

        # Contrast preview boxes
        st.subheader("Text Contrast Preview:")
        st.markdown(f"<div class='contrast-box' style='background-color:{hex_val}; color:white;'>White Text</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='contrast-box' style='background-color:{hex_val}; color:black;'>Black Text</div>", unsafe_allow_html=True)

else:
    st.info("Upload an image to start detecting colors!")
