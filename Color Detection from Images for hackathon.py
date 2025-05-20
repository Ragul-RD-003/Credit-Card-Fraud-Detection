import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from skimage import color

# Load colors dataset and precompute LAB
@st.cache_data
def load_colors_lab():
    df = pd.read_csv("colors.csv")
    df["LAB"] = df.apply(lambda row: color.rgb2lab(np.array([[row[["R","G","B"]].values]], dtype=np.uint8)/255.0)[0][0], axis=1)
    return df

colors = load_colors_lab()

def rgb_to_lab(rgb):
    return color.rgb2lab(np.array([[rgb]], dtype=np.uint8) / 255.0)[0][0]

def get_closest_color_lab(R, G, B):
    input_lab = rgb_to_lab([R, G, B])
    dists = colors["LAB"].apply(lambda lab: np.linalg.norm(input_lab - lab))
    idx = dists.idxmin()
    delta_e = dists.min()
    return colors.loc[idx, "color_name"], delta_e

def get_text_color(rgb):
    # Compute luminance for deciding text color (white or black)
    r, g, b = rgb
    luminance = (0.299*r + 0.587*g + 0.114*b)/255
    return "black" if luminance > 0.6 else "white"

st.set_page_config(page_title="Accurate Color Detection", layout="wide")

st.markdown("<h1 style='text-align:center; font-family:sans-serif; color:#007acc;'>Interactive Accurate Color Detection</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    st.write(f"**Image Size:** {w} x {h} (Width x Height)")

    x = st.slider("X position (width)", 2, w - 3, w // 2)
    y = st.slider("Y position (height)", 2, h - 3, h // 2)

    region = img_np[y-2:y+3, x-2:x+3]
    avg_color = region.mean(axis=(0,1)).astype(int)
    R, G, B = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])

    color_name, delta_e = get_closest_color_lab(R, G, B)
    text_color = get_text_color((R, G, B))

    # Draw pointer with crosshair
    marked_img = img.copy()
    draw = ImageDraw.Draw(marked_img)
    pointer_radius = 5
    draw.ellipse(
        (x - pointer_radius, y - pointer_radius, x + pointer_radius, y + pointer_radius),
        fill='red', outline='white'
    )
    crosshair_length = 7
    draw.line((x - crosshair_length, y, x + crosshair_length, y), fill='white', width=2)
    draw.line((x, y - crosshair_length, x, y + crosshair_length), fill='white', width=2)

    zoom_patch = Image.fromarray(region).resize((120, 120), resample=Image.NEAREST)

    # Layout: Info on left, images on right
    col_info, col_imgs = st.columns([1, 2])

    with col_info:
        st.markdown(f"""
            <div style="
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 8px 16px rgba(0, 122, 204, 0.2);
                background: linear-gradient(135deg, #e6f0ff, #ffffff);
                max-width: 350px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #333333;
                margin-bottom: 30px;
            ">
                <h2 style='color:#007acc; margin-bottom: 10px;'>Detected Color Details</h2>
                <p style='font-size:18px; margin: 5px 0;'><strong>Name:</strong> {color_name}</p>
                <p style='font-size:18px; margin: 5px 0;'><strong>RGB (5x5 Avg):</strong> ({R}, {G}, {B})</p>
                <p style='font-size:16px; margin: 5px 0; color:#555;'>Delta E (color diff): {delta_e:.2f} (lower is better)</p>
                <div style='
                    margin-top: 15px;
                    width: 150px;
                    height: 80px;
                    border-radius: 12px;
                    background-color: rgb({R},{G},{B});
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    color: {text_color};
                    font-weight: 700;
                    font-size: 22px;
                    text-shadow: 0 0 5px rgba(0,0,0,0.3);
                    border: 2px solid #007acc;
                '>Sample Color</div>
            </div>
            """, unsafe_allow_html=True)

    with col_imgs:
        st.image(img, caption="Original Image", use_column_width=True)
        st.image(marked_img, caption="Marked Image with Pointer", use_column_width=True)
        st.markdown("<style>img {margin-bottom: 20px;}</style>", unsafe_allow_html=True)

        # Zoomed patch with subtle hover zoom effect via CSS injected
        st.markdown(
            """
            <style>
            .zoomed-patch:hover {
                transform: scale(1.3);
                transition: transform 0.3s ease;
                z-index: 10;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<img class="zoomed-patch" src="data:image/png;base64,{st.image_to_bytes(zoom_patch)}" alt="Zoomed Patch" width="120">',
            unsafe_allow_html=True,
        )
