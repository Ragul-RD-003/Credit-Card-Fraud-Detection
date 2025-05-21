import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from skimage import color
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Clickable Color Detector", layout="wide")

@st.cache_data
def load_colors():
    df = pd.read_csv("colors.csv")
    df["LAB"] = df.apply(lambda r: color.rgb2lab(np.array([[r[["R","G","B"]]]], dtype=np.uint8)/255.0)[0][0], axis=1)
    return df

colors = load_colors()

def rgb_to_lab(rgb):
    return color.rgb2lab(np.array([[rgb]], dtype=np.uint8)/255.0)[0][0]

def get_closest_colors(R, G, B, top_n=3):
    input_lab = rgb_to_lab([R, G, B])
    distances = colors["LAB"].apply(lambda lab: np.linalg.norm(input_lab - lab))
    closest = colors.loc[distances.nsmallest(top_n).index]
    closest = closest.assign(DeltaE=distances[closest.index])
    return closest

def get_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def brightness(rgb):
    r, g, b = rgb
    return np.sqrt(0.299*r**2 + 0.587*g**2 + 0.114*b**2)

st.title("Clickable Color Detector")

uploaded = st.file_uploader("Upload image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    st.markdown("**Click or tap on the image below to select a pixel:**")
    coords = streamlit_image_coordinates(img, key="image_coords")

    if coords:
        x, y = coords["x"], coords["y"]

        # Clamp coordinates to avoid edges
        x = max(2, min(w - 3, x))
        y = max(2, min(h - 3, y))

        region = img_np[y-2:y+3, x-2:x+3]
        avg_color = region.mean(axis=(0, 1)).astype(int)
        R, G, B = avg_color

        marked = img.copy()
        draw = ImageDraw.Draw(marked)
        pointer_r = 8
        draw.ellipse((x - pointer_r, y - pointer_r, x + pointer_r, y + pointer_r), fill=None, outline='#61dafb', width=4)
        draw.line((x - pointer_r - 6, y, x + pointer_r + 6, y), fill='#61dafb', width=3)
        draw.line((x, y - pointer_r - 6, x, y + pointer_r + 6), fill='#61dafb', width=3)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(marked, caption=f"Selected pixel at ({x}, {y})", use_column_width=True)
            zoom_img = Image.fromarray(region).resize((140, 140), Image.NEAREST)
            st.image(zoom_img, caption="Zoomed 5x5 Pixel Region", width=140)

        with col2:
            hex_val = get_hex((R, G, B))
            top_colors = get_closest_colors(R, G, B, top_n=3)
            main_color_name = top_colors.iloc[0]['color_name']

            text_color = 'white' if brightness((R, G, B)) < 130 else '#121212'

            st.markdown(f"""
                <div style="background-color:{hex_val}; color:{text_color}; 
                border-radius: 16px; padding: 30px; text-align: center; font-weight: 900; font-size: 2rem;
                box-shadow: 0 8px 20px rgba(97, 218, 251, 0.6); user-select:none;">
                    {main_color_name}<br>
                    <small>Hex: {hex_val}</small><br>
                    <small>RGB: ({R}, {G}, {B})</small>
                </div>
            """, unsafe_allow_html=True)

            st.write("### Closest Matches:")
            for i, row in top_colors.iterrows():
                c_hex = get_hex((row["R"], row["G"], row["B"]))
                st.markdown(f"- **{row['color_name']}** - {c_hex}")

    else:
        st.info("Click on the image to pick a pixel and detect its color.")

else:
    st.info("Upload an image to start detecting colors.")
