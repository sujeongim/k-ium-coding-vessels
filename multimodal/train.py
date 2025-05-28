# train
import os
import wandb
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from torchvision import transforms
from tqdm import tqdm

from model import MultiModalAneurysmClassifier  # 위에서 만든 모델 클래스
from dataset import AneurysmDataset             # 위에서 만든 Dataset 클래스

# -------------------- Settings -------------------- #
# csv_path = "/home/edlab/sjim/k-ium-coding-vessels/train_set/train.csv"
# image_dir = "/home/edlab/sjim/k-ium-coding-vessels/train_set/images"
CSV_PATH = "/home/edlab/sjim/k-ium-coding-vessels/train_set/train.csv"
IMAGE_DIR = "/home/edlab/sjim/k-ium-coding-vessels/train_set/images"
TEXT_MODEL_NAME = "bert-base-uncased"
IMAGE_MODEL_NAME = "resnet18"
EPOCHS = 3
BATCH_SIZE = 4
LR = 1e-4
VAL_INTERVAL = 1  # validate every n steps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(project="aneurysm-multimodal", config={
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "image_model": IMAGE_MODEL_NAME,
    "text_model": TEXT_MODEL_NAME
})

# -------------------- Load Data -------------------- #
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

full_dataset = AneurysmDataset(CSV_PATH, IMAGE_DIR, tokenizer, transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# -------------------- Init Model -------------------- #
model = MultiModalAneurysmClassifier(TEXT_MODEL_NAME, IMAGE_MODEL_NAME).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# -------------------- Validation Loop -------------------- #
@torch.no_grad()
def evaluate():
    model.eval()
    total_loss, total_correct = 0, 0
    total = 0
    for images, texts, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        preds = (outputs > 0.5).float()
        total_correct += (preds == labels).float().sum().item()
        total += labels.numel()
    accuracy = total_correct / total
    return total_loss / len(val_loader), accuracy

# -------------------- Training Loop -------------------- #
model.train()
global_step = 0
for epoch in range(EPOCHS):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for images, texts, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images, texts)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        wandb.log({"train/loss": loss.item(), "step": global_step})
        pbar.set_postfix(loss=loss.item())

        # validation
        if global_step % VAL_INTERVAL == 0:
            val_loss, val_acc = evaluate()
            wandb.log({
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "step": global_step
            })

        global_step += 1

# Save model
torch.save(model.state_dict(), "aneurysm_model.pth")
wandb.save("aneurysm_model.pth")
