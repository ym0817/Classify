import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
# from torch.amp import GradScaler, autocast
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from pathlib import Path
import random
import json
from sklearn.model_selection import KFold
from datetime import datetime
from dataset import MyDataset
from model.model import DefectCNN
from model.resnet34 import ResNet34model

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Enhanced Data Preprocessing and Augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])
val_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])


def plot_confusion_matrix(y_true, y_pred, classes, result_dir,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if normalize:
        plt.savefig(os.path.join(result_dir,'normalized-result.png'))
    else:
        plt.savefig(os.path.join(result_dir,'non-normalized-result.png'))
    return ax

# Evaluation Function
def evaluate_model(model, test_loader, classes, result_dir, device):
    model.eval()
    all_preds = []
    all_labels = []
    sample_images = []
    sample_preds = []
    sample_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if len(sample_images) < 5:
                sample_images.extend(images.cpu()[:5])
                sample_preds.extend(preds.cpu()[:5])
                sample_labels.extend(labels.cpu()[:5])
    # Plot non-normalized confusion matrix
    ax1 = plot_confusion_matrix(all_labels, all_preds, classes, result_dir,
                                title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    ax2 = plot_confusion_matrix(all_labels, all_preds, classes, result_dir, normalize=True,
                                title='Normalized confusion matrix')

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

     # cm = confusion_matrix(all_labels, all_preds)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # # plt.show()
    # plt.savefig('./result.png')
    #
    #
    # # plt.figure(figsize=(15, 5))
    # # for i in range(min(5, len(sample_images))):
    # #     plt.subplot(1, 5, i+1)
    # #     # img = sample_images[i].squeeze().numpy() * 0.5 + 0.5
    # #     img = sample_images[i].numpy() * 0.5 + 0.5
    # #     plt.imshow(img, cmap='gray')
    # #     # plt.imshow(img, cmap='jet')
    # #     plt.title(f'Pred: {classes[sample_preds[i]]}\nTrue: {classes[sample_labels[i]]}')
    # #     plt.axis('off')
    # # # plt.show()
    # # plt.savefig('./result_normalized.png')
    #
    # # Output confusion matrix in requested format
    # print("Confusion Matrix:")
    # print("[")
    # for row in cm:
    #     print(" [" + ", ".join(map(str, row)) + "],")
    # print("]")
    # print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    #

def train_val_process(full_dataset, test_dataset, model,criterion,batch,init_lr,max_epoch,models_dir, device):
    k_folds = 2
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Define paths for saving models
    os.makedirs(models_dir, exist_ok=True)  # Ensure models directory exists
    training_cache_dir = models_dir + '/training_cache'
    os.makedirs(training_cache_dir, exist_ok=True)  # Ensure training_cache folder exists
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_save_path = os.path.join(models_dir, f'defect_classifier_(val_loss)_{timestamp}.pth')  # Define final save path for the rolled-up model. Will be used after k-fold and eval


    best_models = []
    fold_results = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f'\nFold {fold + 1}/{k_folds}')
        train_subsampler = Subset(full_dataset, train_ids)
        val_subsampler = Subset(full_dataset, val_ids)
        train_loader = DataLoader(train_subsampler, batch_size=batch, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=batch, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=init_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8,
                                                         patience=5, eps=1e-08)  # This will reduce the learning rate by 15% after {patience} consecutive epochs without validation loss improvement
        # scaler = GradScaler('cuda')
        scaler = GradScaler()

        # Training Loop with Early Stopping and TensorBoard
        writer = SummaryWriter(f'runs/fold_{fold + 1}')
        best_val_loss = float('inf')  # Initializes at infinity so loss can decrease
        train_patience = 25
        patience_counter = 0
        for epoch in range(max_epoch):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                # with autocast('cuda'):
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    with autocast():
                        outputs = model(images)
                        val_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning Rate', lr, epoch)
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            metrics = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc,
                       "val_acc": val_acc, "lr": lr}
            metrics_path = os.path.join(training_cache_dir, f"training_metrics_fold_{fold + 1}.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)

            print(
                f'Epoch {epoch + 1}/{max_epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {lr:.6f}')
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_path = os.path.join(training_cache_dir, f'best_model_fold_{fold + 1}_(val_loss).pth')
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= train_patience:
                    print("Early stopping triggered")
                    break
        writer.close()
        best_models.append(best_model_path)  # Use the actual path
        fold_results.append({'fold': fold + 1, 'val_loss': best_val_loss, 'val_acc': val_acc})
    best_model_path = min(best_models, key=lambda p: fold_results[best_models.index(p)]['val_loss'])  # Use min val_loss for consistency with early stopping behavior. If needed, this can be adjusted to val_acc
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    # Save Final Model
    torch.save(model.state_dict(), final_save_path)
    print(f"Model saved successfully at: {final_save_path}")
    print("Performing inference on sample images...")
    dummy_input = torch.randn(1, 3, 256, 256).to('cpu')
    model.to('cpu')
    torch.onnx.export(
        model,  # PyTorch 模型
        dummy_input,  # 示例输入
        os.path.splitext(final_save_path)[0] + '.onnx',
        # export_params=True,  # 是否导出模型参数
        opset_version=11,  # ONNX 的 opset 版本
        # do_constant_folding=True,  # 是否进行常量折叠优化
        input_names=['input'],  # 输入的名称
        output_names=['output'],  # 多个输出的名称
    )

    model.eval()
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        evaluate_model(model, test_loader, test_dataset.classes, models_dir,device)


def main(data_root,classes,batch, init_lr,max_epoch,checkpoint_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "val")

    train_dataset = MyDataset(train_root, classes,transform=train_transform)
    val_dataset = MyDataset(val_root, classes,transform=val_test_transform)
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])  # Combine for k-fold
    test_dataset = MyDataset(val_root, classes, transform=val_test_transform)

    suff_exet = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')
    class_counts = []
    for cls in train_dataset.classes:
        train_flies = [file for file in os.listdir(os.path.join(train_root, cls)) if
                       file.lower().endswith(suff_exet)]
        val_flies = [file for file in os.listdir(os.path.join(val_root, cls)) if
                     file.lower().endswith(suff_exet)]
        class_counts.append(len(train_flies) + len(val_flies))
    class_weights = torch.tensor([1.0 / np.sqrt(count) if count > 0 else 0 for count in class_counts],
                                 dtype=torch.float)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)


    # model = DefectCNN(num_classes=len(class_counts)).to(device)
    model = ResNet34model(num_classes=len(class_counts)).to(device)

    train_val_process(full_dataset, test_dataset, model, criterion, batch, init_lr, max_epoch, checkpoint_dir,device)



if __name__ == '__main__':
    data_path = "F:\\AlgoData\\SN003\\images_clsdata"
    # class_names = ["8_划伤", "9_刻蚀图缺","10_栅氧damage"]
    class_names = ['1_残留物', '2_外延缺陷', '3_光刻图缺', '4_颗粒', '5_掉柱子',
'6_圈状异常', '7_残缺管芯', "8_划伤", "9_刻蚀图缺", "10_栅氧damage"]


    bs = 16
    learningrate = 0.01
    epoches = 120
    checkpoint_dir = 'checkpoints_r34_256_1020'
    main(data_path,class_names,bs, learningrate,epoches,checkpoint_dir)

