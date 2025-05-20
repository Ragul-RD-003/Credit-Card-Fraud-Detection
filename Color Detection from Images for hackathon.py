import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from skimage import color

# Load color dataset
@st.cache_data
def load_colors():
    return pd.read_csv("colors.csv")

colors = load_colors()

# Convert RGB to LAB
def rgb_to_lab(rgb):
    return color.rgb2lab(np.array([[rgb]], dtype=np.uint8) / 255.0)[0][0]

# Match input color to closest known color in LAB space
def get_closest_color_lab(R, G, B):
    input_lab = rgb_to_lab([R, G, B])
    min_dist = float('inf')
    cname = ""
    for _, row in colors.iterrows():
        target_lab = rgb_to_lab([row["R"], row["G"], row["B"]])
        dist = np.linalg.norm(input_lab - target_lab)
        if dist < min_dist:
            min_dist = dist
            cname = row["color_name"]
    return cname

st.title("Accurate Color Detection from Image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    st.image(img, caption="Original Image", use_column_width=True)
    st.write(f"**Image Size:** {w} x {h} (Width x Height)")

    st.markdown("Use coordinates to select a pixel. A 5x5 region around it will be averaged for color detection.")
    x = st.number_input("X position", min_value=2, max_value=w-3, value=w//2)
    y = st.number_input("Y position", min_value=2, max_value=h-3, value=h//2)

    if st.button("Detect Color"):
        region = img_np[int(y)-2:int(y)+3, int(x)-2:int(x)+3]
        avg_color = region.mean(axis=(0, 1)).astype(int)
        R, G, B = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
        color_name = get_closest_color_lab(R, G, B)

        # Draw red pointer on the image
        marked_img = img.copy()
        draw = ImageDraw.Draw(marked_img)
        pointer_radius = 5
        draw.ellipse((x - pointer_radius, y - pointer_radius, x + pointer_radius, y + pointer_radius), fill='red', outline='white')

        st.markdown("### Marked Image with Selected Pixel")
        st.image(marked_img, caption="Pixel Marked", use_column_width=True)

        # Zoomed-in patch display
        zoom_patch = img_np[int(y)-2:int(y)+3, int(x)-2:int(x)+3]
        zoom_image = Image.fromarray(zoom_patch).resize((100, 100), resample=Image.NEAREST)

        st.markdown("### Detected Color Details")
        st.image(zoom_image, caption="Zoomed Region (5x5)", width=100)
        st.write(f"**Closest Color Name:** {color_name}")
        st.write(f"**RGB (Averaged):** ({R}, {G}, {B})")
        st.markdown(
            f"<div style='width:300px;height:60px;border:2px solid #000;background-color:rgb({R},{G},{B});'></div>",
            unsafe_allow_html=True
        )
