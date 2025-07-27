import streamlit as st
from PIL import Image
import numpy as np
from u2net_model import BackgroundRemovalModel
import math

# ✅ Load Model
st.title("📏 Background Removal + Object Distance Calculator")

@st.cache_resource
def load_model():
    return BackgroundRemovalModel(model_path="u2net_best.pth")

model = load_model()

# ✅ File Upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# ✅ Camera Parameters
st.sidebar.header("Camera Parameters")
img_width_px = st.sidebar.number_input("Image Width (px)", value=4096)
img_height_px = st.sidebar.number_input("Image Height (px)", value=3073)
sensor_width_mm = st.sidebar.number_input("Sensor Width (mm)", value=7.40)
sensor_height_mm = st.sidebar.number_input("Sensor Height (mm)", value=5.55)
focal_length_mm = st.sidebar.number_input("Focal Length (mm)", value=5.50)
object_distance_mm = st.sidebar.number_input("Object Distance (mm)", value=300)

# ✅ Process
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    st.write("Removing background...")
    result_img = model.remove_background(image)
    st.image(result_img, caption="Background Removed", use_column_width=True)

    # ✅ Compute Pixel to Real-world scale
    # formula: real_width / image_width_px = sensor_width / focal_length * (distance)
    real_world_width_mm = (sensor_width_mm * object_distance_mm) / focal_length_mm
    real_world_height_mm = (sensor_height_mm * object_distance_mm) / focal_length_mm

    st.write(f"Estimated real-world size:")
    st.write(f"Width ≈ {real_world_width_mm:.2f} mm")
    st.write(f"Height ≈ {real_world_height_mm:.2f} mm")

    # ✅ Download Button
    st.download_button("Download Background Removed Image", data=result_img.tobytes(),
                       file_name="bg_removed.png", mime="image/png")
