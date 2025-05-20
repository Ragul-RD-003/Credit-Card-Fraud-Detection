import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from skimage import color
import base64
from io import BytesIO

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
    r, g, b = rgb
    luminance = (0.299*r + 0.587*g + 0.114*b)/255
    return "black" if luminance > 0.6 else "white"

def pil_image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode()
    return img_b64

st.set_page_config(page_title="Accurate Color Detection", layout="wide")

st.markdown("<h1 style='text-align:center; font-family:sans-serif; color:#007acc;'>Interactive Accurate Color Detection</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # Set initial slider values to center pixel
    x_init = w // 2
    y_init = h // 2

    # Layout: Info on left, images + sliders on right
    col_info, col_right = st.columns([1, 2])

    with col_info:
        st.write(f"**Image Size:** {w} x {h} (Width x Height)")

    with col_right:
        # Two columns inside right side: image(s) and sliders side by side
        img_col, control_col = st.columns([3, 1])

        with img_col:
            # Sliders must be outside img_col to keep them sticky and interactive, so store values temporarily
            pass

        with control_col:
            st.markdown("### Select Pixel Coordinates")
            x = st.slider("X position (width)", 2, w - 3, x_init)
            y = st.slider("Y position (height)", 2, h - 3, y_init)

        # After we have x, y, proceed to update images accordingly

        # Compute avg color on 5x5 region around (x,y)
        region = img_np[y-2:y+3, x-2:x+3]
        avg_color = region.mean(axis=(0,1)).astype(int)
        R, G, B = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])

        color_name, delta_e = get_closest_color_lab(R, G, B)
        text_color = get_text_color((R, G, B))

        # Draw pointer with crosshair on a copy of the original image
        marked_img = img.copy()
        draw = ImageDraw.Draw(marked_img)

        pointer_radius = 6
        # Draw white outline first for visibility
        draw.ellipse(
            (x - pointer_radius - 1, y - pointer_radius - 1, x + pointer_radius + 1, y + pointer_radius + 1),
            fill='white'
        )
        # Then red circle on top
        draw.ellipse(
            (x - pointer_radius, y - pointer_radius, x + pointer_radius, y + pointer_radius),
            fill='red'
        )
        # Draw crosshair lines with white outline for visibility
        crosshair_length = 8
        line_width = 3
        # horizontal white thicker line as outline
        draw.line((x - crosshair_length, y, x + crosshair_length, y), fill='white', width=line_width+2)
        # vertical white thicker line as outline
        draw.line((x, y - crosshair_length, x, y + crosshair_length), fill='white', width=line_width+2)
        # thinner red lines on top
        draw.line((x - crosshair_length, y, x + crosshair_length, y), fill='red', width=line_width)
        draw.line((x, y - crosshair_length, x, y + crosshair_length), fill='red', width=line_width)

        zoom_patch = Image.fromarray(region).resize((120, 120), resample=Image.NEAREST)
        zoom_patch_b64 = pil_image_to_base64(zoom_patch)

        with img_col:
            st.image(img, caption="Original Image", use_column_width=True)
            st.image(marked_img, caption="Marked Image with Pointer", use_column_width=True)

            # Zoom patch with hover zoom effect
            st.markdown(
                """
                <style>
                .zoomed-patch:hover {
                    transform: scale(1.3);
                    transition: transform 0.3s ease;
                    z-index: 10;
                    cursor: zoom-in;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<img class="zoomed-patch" src="data:image/png;base64,{zoom_patch_b64}" alt="Zoomed Patch" width="120">',
                unsafe_allow_html=True,
            )

    # Show detected color details below everything
    st.markdown(f"""
        <div style="
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0, 122, 204, 0.2);
            background: linear-gradient(135deg, #e6f0ff, #ffffff);
            max-width: 450px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333333;
            margin: 30px auto;
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
