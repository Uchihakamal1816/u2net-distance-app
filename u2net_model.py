import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from u2net_arch import U2NET

class BackgroundRemovalModel:
    def __init__(self, model_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # ✅ Enhanced model loading with error handling
        try:
            self.model = U2NET(3, 1).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Handle potential key mismatches (e.g., from multi-GPU training)
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

        # ✅ Fixed preprocessing to match U2Net requirements
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),  # U2Net expects 320x320 input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def remove_background(self, image: Image.Image) -> Image.Image:
        """Takes a PIL image and returns an RGBA image with background removed.
        
        Args:
            image: PIL Image in RGB mode
            
        Returns:
            PIL Image in RGBA mode with background removed
        """
        # ✅ Preserve original image for final output
        original_size = image.size
        image = image.convert("RGB")
        
        try:
            # ✅ Preprocess with proper size handling
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # ✅ Model inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                pred = outputs[0]  # Get main output (d0)
                pred = torch.sigmoid(pred)
                
                # ✅ Interpolate back to original size
                pred = F.interpolate(pred, 
                                   size=original_size[::-1],  # (H, W)
                                   mode='bilinear', 
                                   align_corners=False)
                mask = pred.squeeze().cpu().numpy()

            # ✅ Improved mask processing
            mask = (mask * 255).clip(0, 255).astype(np.uint8)
            
            # ✅ Create transparent image
            result = image.convert("RGBA")
            result.putalpha(Image.fromarray(mask))
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Background removal failed: {str(e)}")

    @staticmethod
    def validate_model(model_path: str):
        """Helper method to verify model file integrity"""
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            return True
        except Exception as e:
            print(f"Model validation failed: {str(e)}")
            return False
