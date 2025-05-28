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
from sklearn.metrics import f1_score
import numpy as np
from model import MultiModalAneurysmClassifier  # 위에서 만든 모델 클래스
from dataset import AneurysmDataset             # 위에서 만든 Dataset 클래스
import argparse


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

# -------------------- Settings -------------------- #
# # csv_path = "/home/edlab/sjim/k-ium-coding-vessels/train_set/train.csv"
# # image_dir = "/home/edlab/sjim/k-ium-coding-vessels/train_set/images"
# CSV_PATH = "/home/edlab/sjim/k-ium-coding-vessels/train_set/train.csv"
# IMAGE_DIR = "/home/edlab/sjim/k-ium-coding-vessels/train_set/images"
# args.text_model_name = "bert-base-uncased"
# args.image_model_name = "resnet18"
# args.epochs = 3
# args.batch_size = 128
# args.lr = 1e-4
# args.val_interval = 1  # validate every n steps
# args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# wandb.init(project="aneurysm-multimodal", config={
#     "epochs": args.epochs,
#     "batch_size": args.batch_size,
#     "lr": args.lr,
#     "image_model": args.image_model_name,
#     "text_model": args.text_model_name
# })



# -------------------- Validation Loop -------------------- #
@torch.no_grad()
def evaluate(device, model, val_loader, criterion):
    model.eval()
    total_loss, total_correct = 0, 0
    total = 0
    all_preds = []
    all_labels = []
    for images, texts, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        total_correct += (preds == labels).float().sum().item()
        total += labels.numel()
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    # 전체 예측 결과 정리
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    print(y_pred)
    print(y_true)
    print("recall"  , (y_pred * y_true).sum(axis=0))
    print("precision", (y_pred * y_true).sum(axis=1))
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    # class-wise f1 (21개 각 위치)
    class_wise_f1 = f1_score(y_true, y_pred, average=None)
    accuracy = total_correct / total
    print("Class-wise F1:", class_wise_f1.round(3))
    return total_loss / len(val_loader), accuracy, macro_f1, micro_f1

# -------------------- Training Loop -------------------- #
def train(args, device, model, train_loader, val_loader, criterion, optimizer):
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, texts, labels in pbar:
            #print(images.shape, texts[0], labels.shape)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, texts)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            wandb.log({"train/loss": loss.item(), "step": global_step})
            pbar.set_postfix(loss=loss.item())

            # validation
            if global_step % args.val_interval == 0:
                val_loss, accuracy, macro_f1, micro_f1 = evaluate(device, model, val_loader, criterion)
                wandb.log({
                    "val/loss": val_loss,
                    "val/accuracy": accuracy,
                    "val/macro_f1": macro_f1,
                    "val/micro_f1": micro_f1,
                    "step": global_step
                })

            global_step += 1

    # Save model
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
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Initialize wandb
    if not args.disable_wandb:
        wandb.init(project=args.wandb_project, config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "image_model": args.image_model_name,
            "text_model": args.text_model_name,
            "csv_path": args.csv_path,
            "image_dir": args.image_dir,
            "device": str(device)
        })
    
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
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    # -------------------- Init Model -------------------- #
    model = MultiModalAneurysmClassifier(args.text_model_name, args.image_model_name).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    train(args, device, model, train_loader, val_loader, criterion, optimizer)
    print("Training complete. Model saved as aneurysm_model.pth")

    
if __name__ == "__main__":
    main()
