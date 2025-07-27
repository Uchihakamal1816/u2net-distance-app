import streamlit as st
from PIL import Image
from u2net_model import BackgroundRemovalModel
from distance_calc import calculate_distance
import io

# ---------------------------
# Load U²-Net model
# ---------------------------
st.set_page_config(page_title="U²-Net Distance Estimation", layout="centered")
st.title("📏 Background Removal + Object Distance Calculator")

# Load model
@st.cache_resource
def load_model():
    return BackgroundRemovalModel(model_path="u2net_best.pth")

model = load_model()

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

# ---------------------------
# Camera Parameters
# ---------------------------
st.subheader("Camera Parameters")
col1, col2 = st.columns(2)

with col1:
    image_width = st.number_input("Image Width (px)", value=4096)
    sensor_width = st.number_input("Sensor Width (mm)", value=7.4)
    focal_length = st.number_input("Focal Length (mm)", value=5.5)

with col2:
    image_height = st.number_input("Image Height (px)", value=3073)
    sensor_height = st.number_input("Sensor Height (mm)", value=5.55)
    object_distance = st.number_input("Object Distance (mm)", value=300)

# ---------------------------
# Process Button
# ---------------------------
if uploaded_file and st.button("Process Image"):
    with st.spinner("Processing image..."):
        # Convert to PIL image
        image = Image.open(uploaded_file).convert("RGB")

        # Background removal
        result_img = model.remove_background(image)

        # Calculate real-world height
        real_height_mm, adjusted_height_mm, height_px = calculate_distance(
            image=image,
            image_resolution=(image_width, image_height),
            sensor_size_mm=(sensor_width, sensor_height),
            focal_length_mm=focal_length,
            object_distance_mm=object_distance
        )

    # Show results
    st.success("✅ Processing Complete!")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(result_img, caption="Background Removed (RGBA)", use_column_width=True)

    st.markdown(f"### Results:")
    st.write(f"- Object Height (pixels): **{height_px:.2f} px**")
    st.write(f"- Real Height (before tilt): **{real_height_mm:.2f} mm**")
    st.write(f"- Real Height (adjusted): **{adjusted_height_mm:.2f} mm ({adjusted_height_mm/10:.2f} cm)**")

    # Download button
    img_byte_arr = io.BytesIO()
    result_img.save(img_byte_arr, format='PNG')
    st.download_button("Download Processed Image", img_byte_arr.getvalue(), file_name="output.png", mime="image/png")
