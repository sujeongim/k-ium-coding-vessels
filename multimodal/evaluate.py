# train
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import wandb
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, classification_report
import numpy as np
from model import MultiModalAneurysmClassifier  # 위에서 만든 모델 클래스
from dataset import AneurysmDataset             # 위에서 만든 Dataset 클래스
import argparse
import pandas as pd

from utils import CombinedLoss, FocalLoss  
from dataset import COLUMNS_TO_DROP

def compute_pos_weights(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')  # COLUMNS_TO_DROP이 존재하지 않을 수도 있으므로 errors='ignore'
    label_counts = df.iloc[:, 1:].sum(axis=0)  # 'Index' 제외
    total_samples = len(df)
    
    pos_weights = (total_samples - label_counts) / (label_counts + 1e-6)  # 방어적 +epsilon
    return torch.tensor(pos_weights.values, dtype=torch.float)


def custom_collate_fn(batch):
    """
    batch: list of length B, each item is (images, texts, label)
    images: Tensor [8, 3, H, W]
    texts: List[str] (length 8)
    label: Tensor [22]
    """
    images, texts, labels = zip(*batch)
    
    # [B, 8, 3, H, W]
    images = torch.stack(images)

    # texts: List of B samples, each is List[str]
    texts = list(texts)  # stays as List[List[str]]

    # [B, 22]
    labels = torch.stack(labels)

    return images, texts, labels

@torch.no_grad()
def evaluate(device, model, val_loader, criterion):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for images, texts, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, texts)

        # Loss는 raw logits에 대해 계산
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()

        # 확률로 변환 → binary threshold
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        total_correct += (preds == labels).float().sum().item()
        total += labels.numel()

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()


    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    accuracy = total_correct / total
    avg_loss = total_loss / len(val_loader)

    #print("🔍 Classification Report:")
    #print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    return avg_loss, accuracy, macro_f1, micro_f1, macro_recall


# -------------------- Training Loop -------------------- #
def test(args, device, model, test_loader, criterion):
    model.eval()
    total_loss, total_acc, total_macro_f1, total_micro_f1, total_macro_recall = 0, 0, 0, 0, 0
    
    for images, texts, labels in tqdm(test_loader, desc="Testing", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, texts)
        loss = criterion(outputs, labels)

        val_loss, accuracy, macro_f1, micro_f1, macro_recall = evaluate(device, model, test_loader, criterion)
        total_loss += val_loss
        total_acc += accuracy
        total_macro_f1 += macro_f1
        total_micro_f1 += micro_f1
        total_macro_recall += macro_recall
    num_batches = len(test_loader)
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    avg_macro_f1 = total_macro_f1 / num_batches
    avg_micro_f1 = total_micro_f1 / num_batches
    avg_macro_recall = total_macro_recall / num_batches
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}, "
          f"Test Macro F1: {avg_macro_f1:.4f}, Test Micro F1: {avg_micro_f1:.4f}, "
          f"Test Macro Recall: {avg_macro_recall:.4f}")
        


def parse_args():
    parser = argparse.ArgumentParser(description="Aneurysm Multimodal Training")
    
    # Data paths
    parser.add_argument("--csv_path", type=str, 
                       default="/home/edlab/sjim/k-ium-coding-vessels/test_set/test.csv",
                       help="Path to the CSV file")
    parser.add_argument("--image_dir", type=str,
                       default="/home/edlab/sjim/k-ium-coding-vessels/test_set/images",
                       help="Path to the images directory")
    
    # Model configuration
    parser.add_argument("--text_model_name", type=str, default="bert-base-uncased",
                       help="Name of the text model to use")
    parser.add_argument("--image_model_name", type=str, default="resnet18",
                       help="Name of the image model to use")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--loss", type=str, default="focal",
                       choices=["bce", "focal", "combined"],
                       help="Loss function to use: 'bce' for BCEWithLogitsLoss, 'focal' for Focal Loss, 'combined' for Combined Loss")
    
    # Device configuration
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use for training (auto, cuda, or cpu)")
    
    # Wandb configuration
    parser.add_argument("--wandb_project", type=str, default="aneurysm-multimodal",
                       help="Wandb project name")
    parser.add_argument("--disable_wandb", action="store_true",
                       help="Disable wandb logging")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Set random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variables
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    
    print(f"Configuration:")
    print(f"  CSV Path: {args.csv_path}")
    print(f"  Image Directory: {args.image_dir}")
    print(f"  Text Model: {args.text_model_name}")
    print(f"  Image Model: {args.image_model_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    #print(f"  Validation Interval: {args.val_interval}")
    print(f"  Device: {device}")
    
    # Your training code would go here
    # You can access all parameters via args.parameter_name
    global label_names
    df = pd.read_csv(args.csv_path)
    label_names = list(df.columns[1:])  # Assuming the first column is 'Index' and the rest are labels
    # -------------------- Load Data -------------------- #
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = AneurysmDataset(args.csv_path, args.image_dir, tokenizer, transform)


    test_dataset = full_dataset
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    # -------------------- Init Model -------------------- #
    model = MultiModalAneurysmClassifier(args.text_model_name, args.image_model_name).to(device)
    # load pre-trained weights if available
    if os.path.exists("best_aneurysm_model.pth"):
        model.load_state_dict(torch.load("best_aneurysm_model.pth", map_location=device))
        print("Loaded pre-trained model weights.")
    
    pos_weights = compute_pos_weights(args.csv_path).to(device)
    print(f"Positive weights: {pos_weights}")
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    elif args.loss == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean', pos_weight=pos_weights)
    elif args.loss == "combined":
        criterion = CombinedLoss(pos_weight=pos_weights)

    test(args, device, model, test_loader, criterion)


    
if __name__ == "__main__":
    main()
