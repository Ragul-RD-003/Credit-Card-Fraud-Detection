import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage import color
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Color Detector", layout="centered")

@st.cache_data
def load_colors():
    # Load colors.csv which has columns: color_name,R,G,B
    return pd.read_csv("colors.csv")

def rgb_to_lab(rgb):
    # Convert RGB [0-255] to LAB color space for accurate color distance
    return color.rgb2lab(np.array([[rgb]], dtype=np.uint8) / 255.0)[0][0]

def get_closest_color_name(r, g, b, df):
    input_lab = rgb_to_lab([r, g, b])
    min_dist = float('inf')
    closest_name = None
    for _, row in df.iterrows():
        lab = rgb_to_lab([row["R"], row["G"], row["B"]])
        dist = np.linalg.norm(input_lab - lab)
        if dist < min_dist:
            min_dist = dist
            closest_name = row["color_name"]
    return closest_name

def rgb_to_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def mark_point(img, x, y):
    # Draw a red circle to mark clicked position
    img_marked = img.copy()
    draw = ImageDraw.Draw(img_marked)
    radius = 7
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='white', width=2)
    return img_marked

def main():
    st.title("Interactive Color Detection from Image")

    colors_df = load_colors()

    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.markdown("**Click anywhere on the image below to detect the color at that pixel.**")

        coords = streamlit_image_coordinates(img, key="img_coords")

        if coords:
            x, y = int(coords['x']), int(coords['y'])
            img_np = np.array(img)
            h, w = img_np.shape[:2]

            if 0 <= x < w and 0 <= y < h:
                r, g, b = img_np[y, x]
                hex_code = rgb_to_hex((r, g, b))
                color_name = get_closest_color_name(r, g, b, colors_df)

                img_marked = mark_point(img, x, y)
                st.image(img_marked, caption=f"Clicked position marked (x={x}, y={y})", use_column_width=True)

                st.markdown("### Color Information")
                st.write(f"**Coordinates:** ({x}, {y})")
                st.write(f"**Color Name:** {color_name}")
                st.write(f"**HEX:** {hex_code}")
                st.write(f"**RGB:** ({r}, {g}, {b})")
                st.markdown(
                    f"<div style='width:250px; height:50px; background-color:{hex_code}; border:1px solid black;'></div>",
                    unsafe_allow_html=True
                )
            else:
                st.warning("Click coordinates are outside the image boundaries.")
        else:
            st.image(img, caption="Upload and click on the image to detect color", use_column_width=True)
    else:
        st.info("Please upload an image to get started.")

if __name__ == "__main__":
    main()
