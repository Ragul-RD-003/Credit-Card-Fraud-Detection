import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from skimage import color
from sklearn.cluster import KMeans
from io import BytesIO
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config("Smart Color Detector", layout="wide")

@st.cache_data
def load_colors():
    df = pd.read_csv("colors.csv")  # columns: color_name, R, G, B
    df["LAB"] = df.apply(lambda r: color.rgb2lab(np.array([[r[["R", "G", "B"]]]], dtype=np.uint8)/255.0)[0][0], axis=1)
    return df

colors = load_colors()

def rgb_to_lab(rgb):
    return color.rgb2lab(np.array([[rgb]], dtype=np.uint8)/255.0)[0][0]

def get_closest_color(R, G, B):
    input_lab = rgb_to_lab([R, G, B])
    distances = colors["LAB"].apply(lambda lab: np.linalg.norm(input_lab - lab))
    closest = colors.loc[distances.idxmin()]
    return closest["color_name"]

def get_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def brightness(rgb):
    r, g, b = rgb
    return np.sqrt(0.299*r**2 + 0.587*g**2 + 0.114*b**2)

def extract_palette(img_np, n_colors=5):
    pixels = img_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(pixels)
    palette = kmeans.cluster_centers_.astype(int)
    return palette

def generate_color_info(name, rgb, hex_val):
    content = f"Color Name: {name}\nRGB: {rgb}\nHEX: {hex_val}"
    return BytesIO(content.encode())

st.title("Smart Color Detection App")
st.caption("Tip: Upload an image and click anywhere to detect the color!")

uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    coords = streamlit_image_coordinates(img, key="colorclick")

    if coords:
        x, y = coords["x"], coords["y"]
        x = max(2, min(w - 3, x))
        y = max(2, min(h - 3, y))

        region = img_np[y-2:y+3, x-2:x+3]
        avg_color = region.mean(axis=(0, 1)).astype(int)
        R, G, B = avg_color

        marked = img.copy()
        draw = ImageDraw.Draw(marked)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline='red', width=3)

        zoom_img = Image.fromarray(region).resize((120, 120), Image.NEAREST)
        hex_val = get_hex((R, G, B))
        color_name = get_closest_color(R, G, B)
        text_color = 'black' if brightness((R, G, B)) > 160 else 'white'

        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(marked, use_column_width=True, caption=f"Selected pixel at ({x}, {y})")
            st.image(zoom_img, caption="Zoomed 5×5 Region", width=120)
        with col2:
            st.markdown(f"""
                <div style="background-color:{hex_val}; padding:25px; border-radius:12px; text-align:center;
                            color:{text_color}; font-weight:bold; font-size:24px;">
                    {color_name}<br>
                    <span style='font-size:16px;'>RGB: ({R}, {G}, {B})</span><br>
                    <span style='font-size:16px;'>HEX: {hex_val}</span>
                </div>
            """, unsafe_allow_html=True)

            st.download_button("Download Color Info", data=generate_color_info(color_name, (R, G, B), hex_val),
                               file_name="color_info.txt")

    else:
        st.image(img, caption="Click on the image to detect color", use_column_width=True)

    # Show color palette
    st.subheader("Top 5 Dominant Colors in Image")
    palette = extract_palette(img_np)
    palette_html = "".join([
        f"<div style='background-color:{get_hex(c)}; height:50px; flex:1; text-align:center; color:white;'>{get_hex(c)}</div>"
        for c in palette
    ])
    st.markdown(f"<div style='display:flex;'>{palette_html}</div>", unsafe_allow_html=True)
else:
    st.info("Upload an image file to begin detecting colors.")
