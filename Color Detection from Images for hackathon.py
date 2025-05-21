import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from skimage import color
from io import BytesIO
import base64

# ========== CONFIGURATION ==========
st.set_page_config(page_title="Pro Color Detection Tool", layout="wide")
st.markdown("""
    <style>
        .css-1offfwp { padding: 1rem 2rem; }
        .zoomed:hover { transform: scale(1.3); transition: 0.3s ease-in-out; z-index: 100; cursor: zoom-in; }
        .centered { display: flex; justify-content: center; align-items: center; }
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

def simulate_colorblind(img_np):
    return np.dot(img_np[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)

# ========== MAIN APP ==========
st.title("Advanced Interactive Color Detection")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    col1, col2 = st.columns([4, 2])
    with col1:
        st.image(img, caption="Original Image", use_column_width=True)

        # Select pixel
        x = st.slider("X coordinate", 2, w - 3, w // 2)
        y = st.slider("Y coordinate", 2, h - 3, h // 2)

        region = img_np[y - 2:y + 3, x - 2:x + 3]
        avg_color = region.mean(axis=(0, 1)).astype(int)
        R, G, B = avg_color
        hex_val = get_hex((R, G, B))
        lab_val = rgb_to_lab([R, G, B])

        # Marked image
        marked = img.copy()
        draw = ImageDraw.Draw(marked)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='red', outline='white')
        draw.line((x - 8, y, x + 8, y), fill='red', width=2)
        draw.line((x, y - 8, x, y + 8), fill='red', width=2)

        st.image(marked, caption="Pointer Marked", use_column_width=True)

        # Colorblind Sim
        if st.toggle("Colorblind Simulation"):
            cb_img = simulate_colorblind(img_np)
            st.image(cb_img, caption="Simulated Colorblind View (Grayscale)")

        # Zoom & Download
        zoom = Image.fromarray(region).resize((120, 120), resample=Image.NEAREST)
        b64 = pil_to_base64(zoom)
        st.markdown(f"""
        <div class='centered'>
        <img class='zoomed' src='data:image/png;base64,{b64}' width='120'>
        </div>
        <a download="zoom_patch.png" href="data:image/png;base64,{b64}" style="font-size:16px">Download Zoom Patch</a>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Detected Color")
        st.markdown(f"<div style='width:100%; height:50px; background-color:{hex_val}; border-radius:8px;'></div>", unsafe_allow_html=True)
        st.write(f"**RGB:** ({R}, {G}, {B})")
        st.write(f"**HEX:** `{hex_val}`")
        st.code(hex_val, language='text')
        st.write(f"**LAB:** {np.round(lab_val, 2)}")

        top_matches = get_closest_colors(R, G, B)
        st.subheader("Top 3 Closest Colors")
        for _, row in top_matches.iterrows():
            r, g, b = int(row["R"]), int(row["G"]), int(row["B"])
            swatch = get_hex((r, g, b))
            st.markdown(f"<div style='width:100%; height:30px; background-color:{swatch}; border-radius:5px'></div>", unsafe_allow_html=True)
            st.write(f"{row['color_name']} — RGB({r},{g},{b}) — ΔE={row['DeltaE']:.2f}")

        st.subheader("Text Contrast Preview")
        st.markdown(f"<div style='background-color:{hex_val}; padding:20px; color:white; font-weight:bold;'>White Text</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:{hex_val}; padding:20px; color:black; font-weight:bold;'>Black Text</div>", unsafe_allow_html=True)

        st.subheader("Color Palette from Region")
        palette = np.unique(region.reshape(-1, 3), axis=0)
        for c in palette[:6]:
            st.markdown(f"<div style='background-color:{get_hex(c)}; height:25px; border-radius:3px'></div>", unsafe_allow_html=True)

else:
    st.info("Please upload an image to begin.")
