import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from skimage import color
from io import BytesIO
import base64

# ========== PAGE STYLE ==========
st.set_page_config(page_title="Unique Color Detector", layout="wide")
st.markdown("""
    <style>
    .color-box {
        text-align: center;
        padding: 12px;
        margin-top: 10px;
        font-size: 24px;
        font-weight: bold;
        border-radius: 10px;
        background: linear-gradient(90deg, #fdfbfb 0%, #ebedee 100%);
        color: #333;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .swatch {
        height: 35px;
        border-radius: 8px;
        box-shadow: 1px 1px 6px rgba(0,0,0,0.2);
    }
    .zoomed:hover {
        transform: scale(1.2);
        transition: 0.3s ease-in-out;
        cursor: zoom-in;
    }
    </style>
""", unsafe_allow_html=True)

# ========== LOAD COLOR DATA ==========
@st.cache_data
def load_colors():
    df = pd.read_csv("colors.csv")
    df["LAB"] = df.apply(lambda row: color.rgb2lab(
        np.array([[row[["R", "G", "B"]]]], dtype=np.uint8) / 255.0)[0][0], axis=1)
    return df

colors = load_colors()

# ========== UTILITIES ==========
def rgb_to_lab(rgb):
    return color.rgb2lab(np.array([[rgb]], dtype=np.uint8)/255.0)[0][0]

def get_closest_colors(R, G, B, top_n=3):
    input_lab = rgb_to_lab([R, G, B])
    distances = colors["LAB"].apply(lambda lab: np.linalg.norm(input_lab - lab))
    closest = colors.loc[distances.nsmallest(top_n).index]
    closest = closest.assign(DeltaE=distances[closest.index])
    return closest

def get_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def pil_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ========== APP ==========
st.title("Unique Color Detection App")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    col1, col2 = st.columns([4, 2])
    with col1:
        st.image(img, caption="Original Image", use_column_width=True)

        # Pixel Selector
        x = st.slider("X coordinate", 2, w - 3, w // 2)
        y = st.slider("Y coordinate", 2, h - 3, h // 2)

        region = img_np[y - 2:y + 3, x - 2:x + 3]
        avg_color = region.mean(axis=(0, 1)).astype(int)
        R, G, B = avg_color
        hex_val = get_hex((R, G, B))
        lab_val = rgb_to_lab([R, G, B])

        # Draw pointer
        marked = img.copy()
        draw = ImageDraw.Draw(marked)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill='red', outline='white')
        draw.line((x - 10, y, x + 10, y), fill='red', width=2)
        draw.line((x, y - 10, x, y + 10), fill='red', width=2)
        st.image(marked, caption="Marked Pixel", use_column_width=True)

        # Zoom Preview
        zoom = Image.fromarray(region).resize((120, 120), resample=Image.NEAREST)
        b64 = pil_to_base64(zoom)
        st.markdown(f"<img class='zoomed' src='data:image/png;base64,{b64}' width='120'>", unsafe_allow_html=True)

    with col2:
        st.subheader("Detected Color Info")

        # Nearest Color Name
        top_colors = get_closest_colors(R, G, B)
        color_name = top_colors.iloc[0]["color_name"]
        st.markdown(f"<div class='color-box'>{color_name}</div>", unsafe_allow_html=True)

        st.write(f"**RGB:** ({R}, {G}, {B})")
        st.write(f"**HEX:** `{hex_val}`")
        st.write(f"**LAB:** {np.round(lab_val, 2)}")

        # Show swatches of top 3 matches
        st.subheader("Top 3 Closest Matches")
        for _, row in top_colors.iterrows():
            swatch = get_hex((int(row["R"]), int(row["G"]), int(row["B"])))
            st.markdown(f"<div class='swatch' style='background-color:{swatch}'></div>", unsafe_allow_html=True)
            st.write(f"**{row['color_name']}** - ΔE: {row['DeltaE']:.2f}")

        # Harmony suggestion (complementary)
        comp_rgb = [255 - R, 255 - G, 255 - B]
        comp_hex = get_hex(comp_rgb)
        st.subheader("Complementary Color")
        st.markdown(f"<div class='swatch' style='background-color:{comp_hex}'></div>", unsafe_allow_html=True)
        st.write(f"HEX: `{comp_hex}`")

        # Text contrast
        st.subheader("Text Contrast Preview")
        st.markdown(f"<div style='background-color:{hex_val}; padding:15px; color:white;'>White Text</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:{hex_val}; padding:15px; color:black;'>Black Text</div>", unsafe_allow_html=True)

else:
    st.info("Upload an image to begin color detection.")
