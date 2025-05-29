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
from dataset import AneurysmDataset             # 위에서 만든 Dataset 클래스
import argparse
import pandas as pd

from utils import CombinedLoss
from dataset import COLUMNS_TO_DROP
from model import MultiModalAneurysmClassifier, ImageOnlyAneurysmClassifier


def compute_pos_weights(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')  # COLUMNS_TO_DROP이 존재하지 않을 수도 있으므로 errors='ignore'
    label_counts = df.iloc[:, 1:].sum(axis=0)  # 'Index' 제외
    total_samples = len(df)
    
    pos_weights = (total_samples - label_counts) / (label_counts + 1e-6)  # 방어적 +epsilon
    return torch.tensor(pos_weights.values, dtype=torch.float)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1, reduction='mean', pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, weight=self.pos_weight, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



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
def train(args, device, model, train_loader, val_loader, criterion, optimizer):
    model.train()
    global_step = 0
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, texts, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, texts)
            loss = criterion(outputs, labels)

            # print gradient
            if global_step % 100 == 0:
                print(f"Step {global_step}: Loss = {loss.item()}")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"  {name}: grad norm = {param.grad.norm().item()}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # logging
            wandb.log({"train/loss": loss.item(), "step": global_step})
            pbar.set_postfix(loss=loss.item())

            # validation
            if global_step % args.val_interval == 0:
                val_loss, accuracy, macro_f1, micro_f1, macro_recall = evaluate(device, model, val_loader, criterion)
                wandb.log({
                    "val/loss": val_loss,
                    "val/accuracy": accuracy,
                    "val/macro_f1": macro_f1,
                    "val/micro_f1": micro_f1,
                    "val/macro_recall": macro_recall,
                    "step": global_step
                })
                
                # Save the model if it has the best validation loss so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "best_aneurysm_model.pth")
                    print(f"New best model saved with val_loss: {val_loss:.4f}")
            global_step += 1

    # Save model
    wandb.log({"best_val_loss": best_val_loss})
    torch.save(model.state_dict(), "aneurysm_model.pth")
    wandb.save("aneurysm_model.pth")



def parse_args():
    parser = argparse.ArgumentParser(description="Aneurysm Multimodal Training")
    
    # Data paths
    parser.add_argument("--csv_path", type=str, 
                       default="/home/edlab/sjim/k-ium-coding-vessels/train_set/train.csv",
                       help="Path to the CSV file")
    parser.add_argument("--image_dir", type=str,
                       default="/home/edlab/sjim/k-ium-coding-vessels/train_set/images",
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
    parser.add_argument("--val_interval", type=int, default=1,
                       help="Validate every n steps")
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
    parser.add_argument("--run_name", type=str, default=None,
                       help="Name of the wandb run")
    parser.add_argument("--disable_wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--modality", type=str, choices=["multimodal", "image_only"], default="multimodal",
                    help="Choose input modality: multimodal (image+text) or image_only")

    
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
    
    # Initialize wandb
    if not args.disable_wandb:
        wandb.init(
            project=args.wandb_project, 
            name=args.run_name,  # 원하는 run name을 여기에 입력
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "image_model": args.image_model_name,
                "text_model": args.text_model_name,
                "csv_path": args.csv_path,
                "image_dir": args.image_dir,
                "device": str(device)
            }
        )
    
    print(f"Configuration:")
    print(f"  CSV Path: {args.csv_path}")
    print(f"  Image Directory: {args.image_dir}")
    print(f"  Text Model: {args.text_model_name}")
    print(f"  Image Model: {args.image_model_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Validation Interval: {args.val_interval}")
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

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    g = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    # -------------------- Init Model -------------------- #
    if args.modality == "multimodal":
        model = MultiModalAneurysmClassifier(args.text_model_name, args.image_model_name).to(device)
    else:
        model = ImageOnlyAneurysmClassifier(args.image_model_name).to(device)

    
    pos_weights = compute_pos_weights(args.csv_path).to(device)
    print(f"Positive weights: {pos_weights}")
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    elif args.loss == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean', pos_weight=pos_weights)
    elif args.loss == "combined":
        criterion = CombinedLoss(pos_weight=pos_weights)
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    train(args, device, model, train_loader, val_loader, criterion, optimizer)

    print("Training complete. Model saved as aneurysm_model.pth")

    
if __name__ == "__main__":
    main()
