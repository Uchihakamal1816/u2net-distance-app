import streamlit as st
from PIL import Image
import numpy as np
from u2net_model import BackgroundRemovalModel
import math
import torch
import torchvision.transforms as transforms

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

# ✅ Fixed Image Transformation
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),  # Force U2Net's required input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# ✅ Process
if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=True)

        st.write("Removing background...")
        
        # Debug shape before processing
        st.write(f"Original size: {image.size} (WxH)")
        
        # Process and verify tensor shape
        input_tensor = preprocess_image(image)
        st.write(f"Model input shape: {input_tensor.shape} (should be [1, 3, 320, 320])")
        
        result_img = model.remove_background(image)  # Note: Typo in method name (should be remove_background)
        
        # Convert result for display/download
        if isinstance(result_img, torch.Tensor):
            result_img = transforms.ToPILImage()(result_img.squeeze(0))
        
        st.image(result_img, caption="Background Removed", use_column_width=True)

        # ✅ Compute Pixel to Real-world scale
        real_world_width_mm = (sensor_width_mm * object_distance_mm) / focal_length_mm
        real_world_height_mm = (sensor_height_mm * object_distance_mm) / focal_length_mm

        st.markdown("### 📐 Estimated Real-world Size")
        st.success(f"**Width:** {real_world_width_mm:.2f} mm")
        st.success(f"**Height:** {real_world_height_mm:.2f} mm")

        # ✅ Download Button
        from io import BytesIO
        buf = BytesIO()
        result_img.save(buf, format="PNG")
        st.download_button(
            label="Download Background Removed Image",
            data=buf.getvalue(),
            file_name="bg_removed.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.error("Please check: 1) Image format 2) Model file exists 3) Input size")

# Add debug info
st.sidebar.markdown("### Debug Info")
st.sidebar.write(f"Torch version: {torch.__version__}")
st.sidebar.write(f"Device: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")
