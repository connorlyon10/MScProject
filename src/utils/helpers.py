import numpy as np
import random
import torch
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import src.model as m



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# Load an entire dataset
def get_loaders(dataset, train_frac=0.8, bs=32, seed=42):
    n = len(dataset)
    t = int(train_frac * n)
    train_ds, val_ds = random_split(dataset, [t, n-t], generator=torch.Generator().manual_seed(seed))
    return (DataLoader(train_ds, batch_size=bs, shuffle=True),
            DataLoader(val_ds, batch_size=bs, shuffle=False))



# Load a subset; makes testing quicker (needs to be stratified to prevent class imbalance)
def get_stratified_loaders(
    dataset,
    subset_frac: float = 0.3,
    train_frac: float = 0.8,
    batch_size: int = 32,
    random_state: int = 42,
    shuffle_train: bool = True
):

    # 1. Stratified sampling of the full dataset
    full_indices = list(range(len(dataset)))
    labels = dataset.labels
    subset_idx, _ = train_test_split(
        full_indices,
        train_size=subset_frac,
        stratify=labels,
        random_state=random_state
    )
    reduced_ds = Subset(dataset, subset_idx)

    # 2. Stratified train/val split of the reduced dataset
    reduced_labels = [labels[i] for i in subset_idx]
    train_idx, val_idx = train_test_split(
        list(range(len(reduced_ds))),
        train_size=train_frac,
        stratify=reduced_labels,
        random_state=random_state
    )

    # 3. Build DataLoaders
    train_loader = DataLoader(
        Subset(reduced_ds, train_idx),
        batch_size=batch_size,
        shuffle=shuffle_train
    )
    val_loader = DataLoader(
        Subset(reduced_ds, val_idx),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader



# Model builder
def build_model(cfg, input_h=64, input_w=96):
    return m.ConvCount(
        input_height=input_h, input_width=input_w,
        conv1_out=cfg['conv1_out'], conv2_out=cfg['conv2_out'],
        conv3_out=cfg['conv3_out'], conv4_out=cfg['conv4_out'], fc_hidden=cfg['fc_hidden'],
        dropout_prob=cfg['dropout_prob']
    )



# One‐epoch training with tqdm - this can be looped.
# Can return per-class loss, good for testing
def train_one_epoch(model, loader, opt, device, per_class=False):
    model.train()
    total = 0.0

    if per_class:
        # Try to infer number of classes from model
        if hasattr(model, 'fc2'):
            num_classes = int(model.fc2.out_features)
        else:
            num_classes = 5  # fallback default

        sums = [0.0 for _ in range(num_classes)]
        counts = [0 for _ in range(num_classes)]

    for x, y in tqdm(loader, desc="Training", unit="batch"):
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        logits = model(x)

        losses = F.cross_entropy(logits, y, reduction='none')
        loss = losses.mean()

        loss.backward()
        opt.step()

        total += loss.item()

        if per_class:
            losses_cpu = losses.detach().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()
            for lbl, lval in zip(y_cpu, losses_cpu):
                lbl = int(lbl)
                if 0 <= lbl < num_classes:
                    sums[lbl] += float(lval)
                    counts[lbl] += 1

    avg_loss = total / len(loader)

    if per_class:
        per_class_loss = {
            i: (sums[i] / counts[i] if counts[i] > 0 else None)
            for i in range(num_classes)
        }
        return avg_loss, per_class_loss, counts
    else:
        return avg_loss



# Evaluate model (for after training)
def validate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds==y).sum().item()
    return correct / len(loader.dataset)
# endregion