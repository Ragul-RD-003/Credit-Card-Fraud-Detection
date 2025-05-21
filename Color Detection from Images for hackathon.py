import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from skimage import color

st.set_page_config(page_title="Image Color Detector", layout="centered")

# Load colors from CSV
@st.cache_data
def load_colors():
    try:
        df = pd.read_csv("colors.csv")
        return df
    except FileNotFoundError:
        st.error("colors.csv file not found. Make sure it is in the same folder.")
        st.stop()

colors_df = load_colors()

# Convert RGB to LAB
def rgb_to_lab(rgb):
    return color.rgb2lab(np.array([[rgb]]) / 255.0)[0][0]

# Find closest color by LAB distance
def get_closest_color_name(r, g, b, color_df):
    input_lab = rgb_to_lab([r, g, b])
    min_dist = float('inf')
    closest_name = ""
    for _, row in color_df.iterrows():
        try:
            lab = rgb_to_lab([row["R"], row["G"], row["B"]])
            dist = np.linalg.norm(input_lab - lab)
            if dist < min_dist:
                min_dist = dist
                closest_name = row["color_name"]
        except:
            continue
    return closest_name

# Main UI
def main():
    st.title("Advanced Color Detection from Image")
    st.markdown("Upload an image and click a point to detect its closest color name.")

    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image (click to select a pixel)", use_column_width=True)

        # Get click position
        click = st.experimental_data_editor({'Click': ['Click on the image above to select a color.']}, disabled=True)
        st.write("Note: Touch-based selection is limited in basic Streamlit. Use coordinates below.")

        # Manual coordinates input fallback
        w, h = img.size
        x = st.number_input("X Coordinate", min_value=0, max_value=w-1, value=w//2)
        y = st.number_input("Y Coordinate", min_value=0, max_value=h-1, value=h//2)

        if st.button("Detect Color"):
            np_img = np.array(img)
            region = np_img[max(0, y-2):y+3, max(0, x-2):x+3]
            avg_color = region.mean(axis=(0, 1)).astype(int)
            r, g, b = avg_color

            color_name = get_closest_color_name(r, g, b, colors_df)

            st.markdown("### Detected Color Info")
            st.write(f"**Closest Color Name:** `{color_name}`")
            st.write(f"**RGB Values:** ({r}, {g}, {b})")

            # Display color swatch
            st.markdown(
                f"<div style='width:100%;height:60px;border:2px solid #000;background-color:rgb({r},{g},{b});'></div>",
                unsafe_allow_html=True
            )

            # Draw pointer
            draw_img = img.copy()
            draw = ImageDraw.Draw(draw_img)
            draw.ellipse((x-5, y-5, x+5, y+5), fill='red', outline='white')
            st.image(draw_img, caption="Marked Image", use_column_width=True)

if __name__ == "__main__":
    main()
