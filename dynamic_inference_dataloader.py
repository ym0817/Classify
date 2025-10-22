import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime
from tqdm import tqdm

# Define the model architecture (same as DefectCNN class in the training script)
class DefectCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(DefectCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(512), nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(512), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Dynamic model selection
script_dir = Path(__file__).parent
models_dir = script_dir / 'models'
model_files = [f for f in models_dir.glob('*.pth') if f.is_file()]

if not model_files:
    print("No model files found in the models directory.")
    exit(1)

# Prompt for latest model and if not, begin index selection
use_latest = input("Use the latest model? (y/n): ").lower()
if use_latest in ['y', 'yes']:
    selected_model = max(model_files, key=os.path.getctime) # getctime uses the file creation time. If you need to use the most recently modified, use getmtime
else:
    print("Available models:")
    for i, model_file in enumerate(model_files):
        print(f"{i}: {model_file.name}")

    while True:
        try:
            choice = int(input("Enter the number of the model to load (or -1 to exit): "))
            if choice == -1:
                exit(0)
            if 0 <= choice < len(model_files):
                selected_model = model_files[choice]
                break
            print("Invalid choice. Please enter a number between 0 and", len(model_files) - 1)
        except ValueError:
            print("Please enter a valid number.")

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DefectCNN(num_classes=6).to(device)
model.load_state_dict(torch.load(selected_model, weights_only=True))
model.eval()
print(f"Loaded model: {selected_model}")

# Define class labels (or clarity). These are hard-coded according to the classes from the train dataset.
classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# Define image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Image inference
image_folder_path = ""
results = []

# Define Custom Dataset class for loading images from a folder
class ImageDataset(Dataset):
    def __init__(self, image_folder_path, transform):
        """Initialize with folder path and transformation pipeline."""
        self.image_paths = list(Path(image_folder_path).glob('*.jpg')) # Adjust extension if needed. Ex: '*.png'
        if not self.image_paths:
            raise ValueError(f"No .jpg images found in {image_folder_path}")
        self.transform = transform

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """Load and transform a single image."""
        try:
            image = Image.open(self.image_paths[index]).convert('L')
            return self.transform(image)
        except Exception as e:
            print(f"Error loading: {self.image_paths[index].name}: {e}")
            return torch.zeros((1, 256, 256)) # Return zero tendor for failed loads
        

# Prompt for directory usage and set image folder path
default_folder = script_dir / 'targets'
if default_folder.exists() and any(default_folder.glob('*.jpg')):  # Check if targets folder has images
    use_default = input(f"Use images from {default_folder} for inference? (y/n): ").lower()
    if use_default in ['y', 'yes']:
        image_folder_path = default_folder
    else:
        image_folder_path = input("Enter the path to an alternative image folder: ").strip()
else:
    image_folder_path = input("Enter the path to an image folder (no valid images in targets): ").strip()

# Create dataset and dataloader for batch processing
try:
    dataset = ImageDataset(image_folder_path, transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
except ValueError as e:
    print(e)
    exit(1)

# Perform inference and collect results
results = []
total_images = len(dataset)
with torch.no_grad():
    for batch_index, images in enumerate(tqdm(loader, desc="Processing images", total=len(loader))):
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)  # Confidence scores for all classes
        batch_size = images.size(0)
        start_index = batch_index * loader.batch_size # Starting index of the current batch
        for i in range(batch_size):
            image_index = start_index + i  # Index for current image
            if image_index < total_images:  # Ensure index is valid
                image_name = dataset.image_paths[image_index].name
                predicted_class = classes[preds[i].item()]
                confidence = probs[i][preds[i]].item()
                results.append({
                    'image_name': image_name,
                    'predicted_class': predicted_class,
                    'confidence': confidence
                })
                print(f"Processed {image_name}: {predicted_class} [Confidence: {confidence:.4f}]")

# Save results to a timestamped CSV file
if results:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv = os.path.join(script_dir, f'inference_results_{timestamp}.csv')
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")
else:
    print("No images were successfully processed.")

