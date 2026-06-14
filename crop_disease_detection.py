"""
Task 6 — Crop Disease Detection
Model: EfficientNetB0 Transfer Learning
Dataset: PlantVillage Dataset (Kaggle)
End-to-end CNN Pipeline: Image → Preprocess → CNN → Disease Class
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 15
LR         = 0.001
VAL_SPLIT  = 0.2
DATA_DIR   = Path("./plant_disease_data")

print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
NUM_CLASSES  = len(full_dataset.classes)
CLASS_NAMES  = full_dataset.classes

print(f"Total images : {len(full_dataset):,}")
print(f"Total classes: {NUM_CLASSES}")
print(f"Classes: {CLASS_NAMES[:5]}...")

val_size   = int(VAL_SPLIT * len(full_dataset))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
val_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train: {train_size:,} | Val: {val_size:,}")

# ─────────────────────────────────────────────
# EfficientNetB0 Model
# ─────────────────────────────────────────────
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
in_features  = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, NUM_CLASSES),
)

model = model.to(DEVICE)
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def run_epoch(model, loader, train=True):
    model.train() if train else model.eval()
    loss_sum, correct, total = 0, 0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct  += out.max(1)[1].eq(labels).sum().item()
            total    += labels.size(0)
    return loss_sum / total, 100. * correct / total


history      = {"tl": [], "ta": [], "vl": [], "va": []}
best_val_acc = 0.0

print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>8}")
print("─" * 55)

# Phase 1 — Train classifier only
for epoch in range(1, EPOCHS + 1):
    tl, ta = run_epoch(model, train_loader, train=True)
    vl, va = run_epoch(model, val_loader,   train=False)
    scheduler.step()

    history["tl"].append(tl); history["ta"].append(ta)
    history["vl"].append(vl); history["va"].append(va)

    print(f"{epoch:>6} {tl:>11.4f} {ta:>9.2f}% {vl:>9.4f} {va:>7.2f}%")

    if va > best_val_acc:
        best_val_acc = va
        torch.save(model.state_dict(), "best_crop_disease_model.pth")

# Phase 2 — Fine-tune full network
print("\n── Fine-tuning full network ──")
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam([
    {"params": model.classifier.parameters(), "lr": LR},
    {"params": [p for n, p in model.named_parameters() if "classifier" not in n], "lr": LR * 0.1},
], weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

for epoch in range(EPOCHS + 1, EPOCHS + 6):
    tl, ta = run_epoch(model, train_loader, train=True)
    vl, va = run_epoch(model, val_loader,   train=False)
    scheduler.step()

    history["tl"].append(tl); history["ta"].append(ta)
    history["vl"].append(vl); history["va"].append(va)

    print(f"{epoch:>6} {tl:>11.4f} {ta:>9.2f}% {vl:>9.4f} {va:>7.2f}%")

    if va > best_val_acc:
        best_val_acc = va
        torch.save(model.state_dict(), "best_crop_disease_model.pth")

print(f"\n✓ Best Val Accuracy: {best_val_acc:.2f}%")

# ─────────────────────────────────────────────
# Training Curves
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["tl"], label="Train"); ax1.plot(history["vl"], label="Val")
ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(history["ta"], label="Train"); ax2.plot(history["va"], label="Val")
ax2.set_title("Accuracy (%)"); ax2.legend(); ax2.grid(alpha=0.3)
plt.suptitle("Crop Disease Detection — EfficientNetB0", fontsize=12)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# Sample Predictions
# ─────────────────────────────────────────────
model.load_state_dict(torch.load("best_crop_disease_model.pth", map_location=DEVICE))
model.eval()

imgs, labels = next(iter(val_loader))
with torch.no_grad():
    preds = model(imgs.to(DEVICE)).max(1)[1]

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

fig, axes = plt.subplots(2, 8, figsize=(20, 6))
for i, ax in enumerate(axes.flat):
    img = imgs[i].permute(1, 2, 0).numpy()
    img = np.clip(std * img + mean, 0, 1)
    ax.imshow(img)
    t = CLASS_NAMES[labels[i]].replace("_", " ")[:15]
    p = CLASS_NAMES[preds[i].cpu()].replace("_", " ")[:15]
    color = "green" if labels[i] == preds[i].cpu() else "red"
    ax.set_title(f"T:{t}\nP:{p}", fontsize=5, color=color)
    ax.axis("off")

plt.suptitle("Crop Disease Predictions (green=correct, red=wrong)", fontsize=11)
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.show()
print("All done! ✓")
