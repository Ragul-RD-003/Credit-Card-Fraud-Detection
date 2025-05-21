import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

st.set_page_config("All Colors Extractor", layout="wide")

def get_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def extract_all_colors(img, resize_width=150):
    # Resize image for processing
    w_percent = resize_width / float(img.size[0])
    h_size = int((float(img.size[1]) * float(w_percent)))
    resized_img = img.resize((resize_width, h_size))

    # Convert to NumPy array
    img_np = np.array(resized_img)
    pixels = img_np.reshape(-1, img_np.shape[-1])
    
    # Count frequency of each unique color
    pixel_list = [tuple(pixel) for pixel in pixels]
    color_counts = Counter(pixel_list)
    
    return color_counts.most_common()  # returns [(rgb_tuple, count), ...]

st.title("All Color Extractor from Image")
st.caption("Upload an image to extract all unique colors, sorted by usage.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    color_data = extract_all_colors(img)
    total = sum(count for _, count in color_data)

    # Prepare DataFrame
    data = []
    for rgb, count in color_data:
        hex_val = get_hex(rgb)
        percent = round((count / total) * 100, 2)
        data.append({
            "Color": hex_val,
            "RGB": rgb,
            "Count": count,
            "Percentage": percent
        })

    df = pd.DataFrame(data)
    
    st.subheader(f"Found {len(df)} Unique Colors")

    # Color Swatch Grid
    st.markdown("### Color Palette (Top 100)")
    top_colors = df.head(100)
    grid_html = "<div style='display:flex;flex-wrap:wrap;'>"
    for _, row in top_colors.iterrows():
        grid_html += f"""
            <div style='width:80px; height:80px; margin:5px; border:1px solid #ccc; 
                        background-color:{row['Color']}; text-align:center; font-size:10px;
                        color:black; display:flex; align-items:center; justify-content:center;'>
                {row['Color']}
            </div>
        """
    grid_html += "</div>"
    st.markdown(grid_html, unsafe_allow_html=True)

    # Full Table
    with st.expander("Show Full Color Table"):
        st.dataframe(df)

    # Download CSV
    csv = df.to_csv(index=False).encode()
    st.download_button("Download Color Data as CSV", csv, file_name="all_colors.csv", mime="text/csv")
else:
    st.info("Please upload an image to begin.")
