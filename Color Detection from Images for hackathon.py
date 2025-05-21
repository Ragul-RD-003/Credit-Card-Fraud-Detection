import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from skimage import color

# Load color data
@st.cache_data
def load_colors():
    return pd.read_csv("colors.csv")

colors_df = load_colors()

# Convert RGB to LAB
def rgb_to_lab(rgb):
    return color.rgb2lab(np.array([[rgb]], dtype=np.uint8) / 255.0)[0][0]

# Find closest color name from dataset using LAB distance
def get_closest_color_name(r, g, b, df):
    input_lab = rgb_to_lab([r, g, b])
    min_dist = float("inf")
    closest_name = None
    for _, row in df.iterrows():
        try:
            lab = rgb_to_lab([row["R"], row["G"], row["B"]])
            dist = np.linalg.norm(input_lab - lab)
            if dist < min_dist:
                min_dist = dist
                closest_name = row["color_name"]
        except:
            continue
    return closest_name

# Streamlit UI
st.set_page_config(page_title="Advanced Color Detection", layout="wide")
st.title("Interactive Color Detection from Image")
st.markdown("Upload an image, click anywhere on it, and get the color name with live preview.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.markdown("### Select Pixel Coordinates")
    col1, col2 = st.columns(2)
    with col1:
        x = st.slider("X Coordinate", 0, img.width - 1, img.width // 2)
    with col2:
        y = st.slider("Y Coordinate", 0, img.height - 1, img.height // 2)

    # Show selected pixel and average nearby region
    region = img_np[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
    avg_color = region.mean(axis=(0, 1)).astype(int)
    r, g, b = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
    color_name = get_closest_color_name(r, g, b, colors_df)

    # Mark the image
    marked_img = img.copy()
    draw = ImageDraw.Draw(marked_img)
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 0, 0), outline="white", width=2)

    st.markdown("### Result")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(marked_img, caption="Image with Pointer", use_column_width=True)
    with col2:
        st.markdown(f"**Closest Color Name:** `{color_name}`")
        st.markdown(f"**RGB:** ({r}, {g}, {b})")
        st.markdown(
            f"<div style='width:100px;height:60px;border:2px solid #000;background-color:rgb({r},{g},{b});'></div>",
            unsafe_allow_html=True
        )
