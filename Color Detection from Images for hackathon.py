import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from skimage import color

# Load and cache colors with LAB precomputed
@st.cache_data
def load_colors_lab():
    df = pd.read_csv("colors.csv")
    df["LAB"] = df.apply(lambda row: color.rgb2lab(np.array([[row[["R","G","B"]].values]], dtype=np.uint8)/255.0)[0][0], axis=1)
    return df

colors = load_colors_lab()

# Convert RGB to LAB
def rgb_to_lab(rgb):
    return color.rgb2lab(np.array([[rgb]], dtype=np.uint8) / 255.0)[0][0]

# Find closest color by LAB distance
def get_closest_color_lab(R, G, B):
    input_lab = rgb_to_lab([R, G, B])
    dists = colors["LAB"].apply(lambda lab: np.linalg.norm(input_lab - lab))
    idx = dists.idxmin()
    return colors.loc[idx, "color_name"]

st.title("Interactive Accurate Color Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    st.write(f"**Image Size:** {w} x {h} (Width x Height)")

    # Coordinate sliders for live adjustment
    x = st.slider("X position (width)", 2, w - 3, w // 2)
    y = st.slider("Y position (height)", 2, h - 3, h // 2)

    # Average 5x5 region color
    region = img_np[y-2:y+3, x-2:x+3]
    avg_color = region.mean(axis=(0,1)).astype(int)
    R, G, B = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])

    color_name = get_closest_color_lab(R, G, B)

    # Draw pointer with crosshair
    marked_img = img.copy()
    draw = ImageDraw.Draw(marked_img)
    pointer_radius = 5

    # Red circle with white outline
    draw.ellipse(
        (x - pointer_radius, y - pointer_radius, x + pointer_radius, y + pointer_radius),
        fill='red', outline='white'
    )

    # White crosshair lines
    crosshair_length = 7
    draw.line((x - crosshair_length, y, x + crosshair_length, y), fill='white', width=2)
    draw.line((x, y - crosshair_length, x, y + crosshair_length), fill='white', width=2)

    # Zoomed patch
    zoom_patch = Image.fromarray(region).resize((100, 100), resample=Image.NEAREST)

    # Display images and info side by side
    col1, col2, col3 = st.columns([3, 3, 1])

    with col1:
        st.image(img, caption="Original Image", use_column_width=True)

    with col2:
        st.image(marked_img, caption="Marked Image with Pointer", use_column_width=True)
        st.markdown(f"**Closest Color:** {color_name}")
        st.markdown(f"**RGB (Averaged 5x5):** ({R}, {G}, {B})")
        st.markdown(
            f"<div style='width:150px;height:60px;border:2px solid #000;background-color:rgb({R},{G},{B});'></div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.image(zoom_patch, caption="Zoomed 5x5 Region", width=100)
