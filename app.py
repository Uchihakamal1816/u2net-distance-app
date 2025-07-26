import streamlit as st
from PIL import Image
from u2net_model import BackgroundRemovalModel
from distance_calc import calculate_distance
import tempfile

st.title("🖼️ U²-Net Background Removal + Distance Measurement")

# Sidebar inputs
st.sidebar.header("Camera Parameters")
image_resolution = st.sidebar.text_input("Image Resolution (width,height)", "4096,3073")
sensor_size = st.sidebar.text_input("Sensor Size in mm (width,height)", "7.4,5.55")
focal_length = st.sidebar.number_input("Focal Length (mm)", 5.5)
object_distance = st.sidebar.number_input("Object Distance (mm)", 300)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Load Model
    st.write("Running U²-Net background removal...")
    model = BackgroundRemovalModel(model_path="u2net_best.pth")
    result = model.remove_background(uploaded_file)

    st.image(result, caption="Background Removed", use_column_width=True)

    # Save to temp and calculate distance
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        result.save(tmp.name)
        res = tuple(map(int, image_resolution.split(",")))
        sensor = tuple(map(float, sensor_size.split(",")))

        height_mm, height_cm = calculate_distance(
            image=result,
            image_resolution=res,
            sensor_size_mm=sensor,
            focal_length_mm=focal_length,
            object_distance_mm=object_distance
        )

        st.success(f"Estimated Object Height: {height_cm:.2f} cm")
