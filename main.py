"""
Action Recognition on Ho-3D with an Action Transformer
-------------------------------------------------------

This script:
 1. Loads and annotates Ho-3D per-frame object poses and action labels.
 2. Balances classes via oversampling.
 3. Scales features and encodes labels.
 4. Defines an ActionTransformer classifier (based on nn.TransformerEncoder).
 5. Trains and evaluates on train/test split.

Inputs:
 - `root` should point to the Ho-3D folder containing subfolders with:
      • object_pose.txt
      • action_class.txt
      • grasp_class.txt (optional)

Outputs:
 - Prints per-epoch training/validation loss & accuracy.
 - Final test accuracy.
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# -----------------------------------------------------------------------------
# LOAD & ANNOTATE DATA
# -----------------------------------------------------------------------------
root = '/media/song/新加卷/Action_Recognition/data'
labels, values, actions = [], [], []

for tf in glob.glob(root + '/**/*.txt', recursive=True):
    fname = os.path.basename(tf)
    with open(tf, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if fname == 'object_pose.txt':
                values.append([float(x) for x in line.split()])
            elif fname == 'action_class.txt':
                # format "frame_idx:action_label"
                idx, lbl = line.split(':')
                actions.append(lbl)
            # grasp_class.txt could be parsed similarly if needed

# Build DataFrame: each row = one frame
df = pd.DataFrame(values)
df['ActionClass'] = actions
df.columns = [f"x{i}" for i in range(df.shape[1]-1)] + ['ActionClass']

# Map labels to integers
df['ActionClass'] = df['ActionClass'].str.strip()
le = LabelEncoder()
df['y'] = le.fit_transform(df['ActionClass'])
label_map = {i: lbl for i, lbl in enumerate(le.classes_)}
print("Annotated label mapping:", label_map)

X = df.drop(['ActionClass', 'y'], axis=1).values
y = df['y'].values

# -----------------------------------------------------------------------------
# BALANCE & SCALE
# -----------------------------------------------------------------------------
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# -----------------------------------------------------------------------------
# TRAIN/TEST SPLIT
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------------------------
# PYTORCH DATASET & DATALOADER
# -----------------------------------------------------------------------------
class Ho3DDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = Ho3DDataset(X_train, y_train)
test_ds  = Ho3DDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=2)

# -----------------------------------------------------------------------------
# ACTION TRANSFORMER MODEL
# -----------------------------------------------------------------------------
class ActionTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=128, nhead=4, num_layers=2, num_classes=10):
        super().__init__()
        # project input features to d_model
        self.input_proj = nn.Linear(feature_dim, d_model)
        # positional embeddings for a single token
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: (B, F) -> (B, 1, d_model)
        x = self.input_proj(x).unsqueeze(1) + self.pos_emb
        # transformer expects (S, B, E)
        x = self.transformer(x.permute(1,0,2))  # -> (1, B, d_model)
        x = x.squeeze(0)                        # -> (B, d_model)
        return self.classifier(x)

# instantiate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ActionTransformer(
    feature_dim=X_train.shape[1],
    d_model=128,
    nhead=4,
    num_layers=2,
    num_classes=len(label_map)
).to(device)

# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)
num_epochs = 20

for epoch in range(1, num_epochs+1):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        running_loss += loss.item()*xb.size(0)
        _, preds = pred.max(1)
        correct += (preds==yb).sum().item()
        total += yb.size(0)

    train_loss = running_loss/total
    train_acc  = correct/total

    # validation
    model.eval()
    val_loss, val_corr, val_tot = 0, 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            l = criterion(pred, yb)
            val_loss += l.item()*xb.size(0)
            _, preds = pred.max(1)
            val_corr += (preds==yb).sum().item()
            val_tot += yb.size(0)
    val_loss /= val_tot
    val_acc  = val_corr/val_tot

    print(f"Epoch {epoch:02d} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
          f"Val Loss:   {val_loss:.4f}, Acc: {val_acc:.3f}")

# -----------------------------------------------------------------------------
# FINAL TEST METRICS
# -----------------------------------------------------------------------------
model.eval()
test_corr, test_tot = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        _, preds = pred.max(1)
        test_corr += (preds==yb).sum().item()
        test_tot += yb.size(0)

print(f"\nFinal Test Accuracy: {test_corr/test_tot:.3f}")
