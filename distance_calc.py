import numpy as np
from PIL import Image, ImageDraw
from skimage import color, measure

def calculate_distance(image: Image.Image,
                       image_resolution=(4096, 3073),
                       sensor_size_mm=(7.4, 5.55),
                       focal_length_mm=5.5,
                       object_distance_mm=300):

    image_np = np.array(image)
    gray = color.rgb2gray(image_np[..., :3]) if image_np.ndim == 3 else image_np
    mask = gray > 0.1

    contours = measure.find_contours(mask, level=0.5)
    if not contours:
        return None, image

    largest_contour = max(contours, key=len)
    y_coords = largest_contour[:, 0]

    top_vertex_y = np.min(y_coords)
    bottom_vertex_y = np.max(y_coords)
    height_px = bottom_vertex_y - top_vertex_y

    image_width_px, image_height_px = image_resolution
    sensor_width_mm, sensor_height_mm = sensor_size_mm

    fov_width_mm = (sensor_width_mm * object_distance_mm) / focal_length_mm
    fov_height_mm = (sensor_height_mm * object_distance_mm) / focal_length_mm

    mm_per_pixel_y = fov_height_mm / image_height_px
    real_height_mm = height_px * mm_per_pixel_y
    real_height_cm = real_height_mm / 10.0

    annotated_img = image.convert("RGB")
    draw = ImageDraw.Draw(annotated_img)
    draw.line([(largest_contour[:, 1].min(), top_vertex_y),
               (largest_contour[:, 1].min(), bottom_vertex_y)],
              fill="red", width=5)
    draw.text((10, 10), f"Height: {real_height_cm:.2f} cm", fill="yellow")

    return real_height_cm, annotated_img
