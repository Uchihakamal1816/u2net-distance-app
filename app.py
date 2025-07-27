import streamlit as st
from PIL import Image
from u2net_model import BackgroundRemovalModel
import numpy as np

# Initialize Model
model = BackgroundRemovalModel(model_path="u2net_best.pth")

st.title("📏 Background Removal + Object Distance Calculator")

# Upload Image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Camera parameters
    st.subheader("Camera Parameters")
    img_width_px = st.number_input("Image Width (px)", value=image.width)
    img_height_px = st.number_input("Image Height (px)", value=image.height)
    sensor_width_mm = st.number_input("Sensor Width (mm)", value=7.4)
    sensor_height_mm = st.number_input("Sensor Height (mm)", value=5.55)
    focal_length_mm = st.number_input("Focal Length (mm)", value=5.5)
    object_distance_mm = st.number_input("Object Distance (mm)", value=300.0)

    if st.button("Process"):
        result_img = model.remove_background(image)
        st.image(result_img, caption="Background Removed", use_column_width=True)

        # Distance calculation
        px_size_mm = sensor_width_mm / img_width_px
        real_width_mm = img_width_px * px_size_mm * object_distance_mm / focal_length_mm
        real_height_mm = img_height_px * (sensor_height_mm / img_height_px) * object_distance_mm / focal_length_mm

        st.success(f"Estimated Real Dimensions: {real_width_mm:.2f} mm × {real_height_mm:.2f} mm")
