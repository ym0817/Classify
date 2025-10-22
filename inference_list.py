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
from model.model import DefectCNN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])


def model_load(selected_model, names, device):
    # Load the model
    model = DefectCNN(num_classes=len(names)).to(device)
    model.load_state_dict(torch.load(selected_model, weights_only=True))
    model.eval()
    print(f"Loaded model: {selected_model}")
    return model

def inference_files(weight_path,label_names, image_folder_path ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_load(weight_path, label_names, device )
    suff_exet = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')
    matches = []
    all_preds = []
    all_labels = []
    for dirpath, _, filenames in os.walk(image_folder_path):
        for fname in filenames:
            if fname.lower().endswith(suff_exet):
                file_path = os.path.join(dirpath, fname)
                label_name = os.path.basename(os.path.split(file_path)[0])
                if label_name in label_names:
                    label_index = label_names.index(label_name)
                    img = Image.open(file_path)  # .convert('L')  # Grayscale
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    # print("img_tensor , ", img_tensor.shape)
                    outputs = model(img_tensor)
                    _, preds = torch.max(outputs, 1)
                    pred_index = preds.cpu().numpy()
                    all_preds.append(pred_index)
                    all_labels.append(label_index)
                else:
                    print("{label_name} is not in {label_names}")
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.show()
    plt.savefig('./Test_Acc.png')



if __name__ == '__main__':

    checkpoint_path = "checkpoints/defect_classifier_(val_loss)_20251016_160725.pth"
    class_names = ["H_8_划伤", "I_9_刻蚀图缺", "J_10_栅氧damage"]
    inference_fold = "F:\\AlgoData\\SN003\\images_clsdata_renamed\\val"

    inference_files(checkpoint_path, class_names, inference_fold)


