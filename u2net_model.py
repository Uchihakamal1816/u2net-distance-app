import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from u2net_arch import U2NET  # Import your full U²-Net architecture

class BackgroundRemovalModel:
    def __init__(self, model_path=None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        self.model = U2NET(3, 1).to(self.device)

        
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)  # Allow minor mismatches
                print(f"✅ Loaded model weights from: {model_path}")
            except FileNotFoundError:
                print(f"❌ Model file not found at {model_path}")
            except RuntimeError as e:
                print(f"⚠️ RuntimeError while loading weights: {e}")

        
        self.model.eval()

       
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def remove_background(self, image: Image.Image):
        """
        Removes background from input PIL image and returns an RGBA image.
        """
        original_size = image.size  # Save original size
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy()  # Apply sigmoid

        
        mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(original_size)

        
        mask_array = np.array(mask) / 255.0
        image_array = np.array(image)

        result = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
        result[:, :, :3] = image_array
        result[:, :, 3] = (mask_array * 255).astype(np.uint8)

        return Image.fromarray(result, 'RGBA')
