import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import color, measure
from typing import Tuple, Optional

def calculate_distance(
    image: Image.Image,
    image_resolution: Tuple[int, int] = (4096, 3073),
    sensor_size_mm: Tuple[float, float] = (7.4, 5.55),
    focal_length_mm: float = 5.5,
    object_distance_mm: float = 300
) -> Tuple[Optional[float], Image.Image]:
    """
    Calculate object height from image and annotate the measurement.
    
    Args:
        image: Input PIL Image
        image_resolution: Camera resolution (width, height) in pixels
        sensor_size_mm: Camera sensor size (width, height) in mm
        focal_length_mm: Lens focal length in mm
        object_distance_mm: Distance to object in mm
        
    Returns:
        Tuple of (calculated height in cm, annotated image) or (None, original image) if no contour found
    """
    try:
        # Convert to numpy array and create mask
        image_np = np.array(image)
        gray = color.rgb2gray(image_np[..., :3]) if image_np.ndim == 3 else image_np
        mask = gray > 0.1  # Adjust threshold if needed

        # Find contours
        contours = measure.find_contours(mask, level=0.5)
        if not contours:
            return None, image

        # Process largest contour
        largest_contour = max(contours, key=len)
        y_coords = largest_contour[:, 0]
        top_vertex_y, bottom_vertex_y = np.min(y_coords), np.max(y_coords)
        height_px = bottom_vertex_y - top_vertex_y

        # Calculate real-world dimensions
        image_width_px, image_height_px = image_resolution
        sensor_width_mm, sensor_height_mm = sensor_size_mm

        fov_height_mm = (sensor_height_mm * object_distance_mm) / focal_length_mm
        mm_per_pixel_y = fov_height_mm / image_height_px
        real_height_mm = height_px * mm_per_pixel_y
        real_height_cm = real_height_mm / 10.0

        # Create annotated image
        annotated_img = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated_img)
        
        # Draw measurement line
        left_x = largest_contour[:, 1].min()
        draw.line(
            [(left_x, top_vertex_y), (left_x, bottom_vertex_y)],
            fill="red", 
            width=5
        )
        
        # Add text with better visibility
        try:
            font = ImageFont.load_default(size=20)
        except:
            font = ImageFont.load_default()
            
        draw.text(
            (10, 10), 
            f"Height: {real_height_cm:.2f} cm", 
            fill="yellow",
            font=font,
            stroke_width=2,
            stroke_fill="black"
        )

        return real_height_cm, annotated_img

    except Exception as e:
        print(f"Error in distance calculation: {str(e)}")
        return None, image
