import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from u2net_arch import U2NET

class BackgroundRemovalModel:
    def __init__(self, model_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ Load same architecture as trained
        self.model = U2NET(3, 1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # ✅ Preprocessing: match training
        self.transform = transforms.Compose([
            transforms.Resize((352, 352)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def remove_background(self, image: Image.Image):
        """Takes a PIL image and returns an RGBA image with background removed."""
        # ✅ Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            pred = outputs[0]  # d0
            pred = torch.sigmoid(pred)
            pred = F.interpolate(pred, size=image.size[::-1], mode="bilinear", align_corners=False)
            mask = pred.squeeze().cpu().numpy()

        # ✅ Normalize mask
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask = (mask * 255).astype(np.uint8)

        # ✅ Apply mask
        image = image.convert("RGBA")
        np_img = np.array(image)
        np_img[:, :, 3] = mask  # alpha channel = mask
        return Image.fromarray(np_img)
