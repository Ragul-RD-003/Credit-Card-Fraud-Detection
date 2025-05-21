import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage import color
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config("Interactive Color Picker", layout="centered")

@st.cache_data
def load_colors():
    # Load your color dataset, example: color_name,R,G,B columns
    return pd.read_csv("colors.csv")

def get_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def rgb_to_lab(rgb):
    return color.rgb2lab(np.array([[rgb]], dtype=np.uint8) / 255.0)[0][0]

def get_closest_color_name(r, g, b, df):
    input_lab = rgb_to_lab([r, g, b])
    min_dist = float('inf')
    name = ""
    for _, row in df.iterrows():
        target_lab = rgb_to_lab([row["R"], row["G"], row["B"]])
        dist = np.linalg.norm(input_lab - target_lab)
        if dist < min_dist:
            min_dist = dist
            name = row["color_name"]
    return name

colors_df = load_colors()

st.title("Interactive Color Picker - Click to Preview Color")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.markdown("**Click anywhere on the image below to detect the color**")

    coords = streamlit_image_coordinates(img, key="click_coords")

    if coords:
        x, y = int(coords["x"]), int(coords["y"])
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        if 0 <= x < w and 0 <= y < h:
            r, g, b = img_np[y, x]
            hex_val = get_hex((r, g, b))
            color_name = get_closest_color_name(r, g, b, colors_df)

            # Mark clicked point
            marked_img = img.copy()
            draw = ImageDraw.Draw(marked_img)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='red', outline='white')

            st.image(marked_img, caption="Clicked Position Marked", use_column_width=True)

            st.markdown("### Color Details")
            st.write(f"**Coordinates:** ({x}, {y})")
            st.write(f"**Color Name:** {color_name}")
            st.write(f"**HEX:** {hex_val}")
            st.write(f"**RGB:** ({r}, {g}, {b})")
            st.markdown(
                f"<div style='width:250px;height:50px;background-color:{hex_val};border: 1px solid #000;'></div>",
                unsafe_allow_html=True
            )
    else:
        st.image(img, caption="Click on image to detect color", use_column_width=True)
else:
    st.info("Upload an image to get started.")
