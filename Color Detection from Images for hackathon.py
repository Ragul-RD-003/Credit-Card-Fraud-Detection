
import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image

# Load color dataset
@st.cache_data
def load_colors():
    return pd.read_csv("colors.csv")

colors = load_colors()

# Calculate closest color
def get_color_name(R, G, B):
    min_dist = float('inf')
    cname = ""
    for i in range(len(colors)):
        d = abs(R - colors.loc[i, "R"]) + abs(G - colors.loc[i, "G"]) + abs(B - colors.loc[i, "B"])
        if d <= min_dist:
            min_dist = d
            cname = colors.loc[i, "color_name"]
    return cname

st.title("Color Detection from Image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_np = np.array(img)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.write("Click anywhere on the image to detect color.")
    clicked = st.image(img_np, use_column_width=True)

    # Get pixel position input
    x = st.number_input("X position", min_value=0, max_value=img_np.shape[1]-1, value=0)
    y = st.number_input("Y position", min_value=0, max_value=img_np.shape[0]-1, value=0)

    if st.button("Detect Color"):
        pixel = img_np[int(y), int(x)]
        R, G, B = int(pixel[0]), int(pixel[1]), int(pixel[2])
        color_name = get_color_name(R, G, B)

        st.write(f"**Color Name:** {color_name}")
        st.write(f"**RGB:** ({R}, {G}, {B})")
        st.markdown(
            f"<div style='width:150px;height:50px;background-color:rgb({R},{G},{B});'></div>",
            unsafe_allow_html=True
        )
