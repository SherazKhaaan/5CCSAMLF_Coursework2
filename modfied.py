import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# ================ GLOBALS ================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ========================================
# 1. DATASETS & TRANSFORMS
# ========================================

def get_cifar10_datasets():
    """
    Returns train_dataset, test_dataset for CIFAR-10.
    Adjust transforms as needed.
    """
    # e.g. minimal transforms:
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    return train_dataset, test_dataset


# ========================================
# 2. RANDOM SELECTION vs. TPC-RP
# ========================================
def select_random_samples(dataset_size, budget, seed=None):
    """
    Returns a random subset of indices of length 'budget'.
    dataset_size = len(train_dataset) = 50000 typically.
    """
    if seed is not None:
        random.seed(seed)
    all_indices = list(range(dataset_size))
    selected_indices = random.sample(all_indices, budget)
    return selected_indices


# ---------- TPC-RP (TypiClust) -----------
class SimCLRResNet18(nn.Module):
    """
    A minimal ResNet-18 for SimCLR. (Encoder only, no final FC.)
    """
    def __init__(self, feature_dim=128):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(pretrained=False)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        # In typical SimCLR training, there's a projection head:
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
    def forward(self, x):
        h = self.encoder(x)   # (batch, 512, 1, 1)
        h = h.view(h.size(0), -1)  # (batch, 512)
        return h

def compute_embeddings(encoder, dataset, batch_size=128):
    """
    Returns (embeddings, indices). embeddings shape: (N, 512).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    encoder.eval()
    all_embs = []
    all_idxs = []
    start_idx = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            feats = encoder(images)
            all_embs.append(feats.cpu().numpy())

            # track indices
            bsz = images.size(0)
            idxs = list(range(start_idx, start_idx + bsz))
            start_idx += bsz
            all_idxs.extend(idxs)

    all_embs = np.concatenate(all_embs, axis=0)
    return all_embs, all_idxs

def select_typiclust_samples(encoder, train_dataset, budget=100, k_nn=20):
    """
    TPC-RP approach: K-means, pick the highest typicality in each cluster.
    """
    # 1) Compute embeddings
    all_embs, all_idxs = compute_embeddings(encoder, train_dataset)
    # 2) K-means => cluster assignments
    kmeans = KMeans(n_clusters=budget, random_state=42)
    labels = kmeans.fit_predict(all_embs)

    selected_indices = []
    for cluster_id in range(budget):
        cluster_idxs = np.where(labels == cluster_id)[0]
        if len(cluster_idxs) == 0:
            continue
        cluster_embeds = all_embs[cluster_idxs]
        # pairwise distances
        dist = pairwise_distances(cluster_embeds)
        typicalities = []
        for i in range(dist.shape[0]):
            sorted_dists = np.sort(dist[i])[1:k_nn+1]
            avg_dist = np.mean(sorted_dists)
            typicalities.append(1.0 / (avg_dist + 1e-8))
        best_local_idx = np.argmax(typicalities)
        best_global_idx = cluster_idxs[best_local_idx]
        # map from local idx in cluster to dataset index
        selected_indices.append(best_global_idx)
    return selected_indices


# ========================================
# 3. TRAINING METHODS
# ========================================

# ---- 3A) FULLY SUPERVISED (train CNN from scratch) ----
class SimpleCNN(nn.Module):
    """
    A small CNN for CIFAR-10 (fully supervised).
    You can replace with any architecture or a smaller ResNet.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_cnn_fully_supervised(train_dataset, selected_indices, epochs=10):
    """
    Train a CNN from scratch on the selected subset.
    """
    subset = Subset(train_dataset, selected_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=True)

    model = SimpleCNN(num_classes=10).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"[FullySup] Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")
    return model


# ---- 3B) FULLY SUPERVISED w/ SELF-SUPERVISED EMBEDDINGS (linear head) ----
def train_linear_head(encoder, train_dataset, selected_indices, epochs=10):
    """
    Freeze the encoder, train a linear layer on top of the 512-d features for the selected subset.
    """
    subset = Subset(train_dataset, selected_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=True)

    # Freeze encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Build a linear head
    linear_head = nn.Linear(512, 10).to(DEVICE)
    optimizer = optim.Adam(linear_head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    linear_head.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                feats = encoder(images)  # shape (batch, 512)
            logits = linear_head(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"[LinHead] Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")
    return linear_head


# ---- 3C) SEMI-SUPERVISED (placeholder) ----
def train_semi_supervised(train_dataset, selected_indices, unlabeled_indices, epochs=10):
    """
    Placeholder function for a semi-supervised pipeline (e.g., FlexMatch).
    - 'train_dataset' is the entire CIFAR-10 train set.
    - 'selected_indices' are labeled; 'unlabeled_indices' are unlabeled.
    You'd integrate your chosen library or code here.
    Return a trained model.
    """
    # This is left abstract since frameworks differ.
    # In practice you might:
    #  1. Create a DataLoader for labeled subset
    #  2. Another for unlabeled subset
    #  3. Implement or call e.g. FixMatch with these two loaders
    #  4. Train & return the final model
    print("Semi-supervised training not implemented (placeholder).")
    return None


# ========================================
# 4. EVALUATION
# ========================================
def evaluate_accuracy(model, test_dataset, is_encoder_plus_linear=False, encoder=None):
    """
    Evaluate on the CIFAR-10 test set.
    If is_encoder_plus_linear=True, 'model' is the linear head, and 'encoder' is the pretrained encoder.
    Otherwise, 'model' is a normal CNN that directly outputs logits.
    """
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    correct = 0
    total = 0

    model.eval()
    if is_encoder_plus_linear and encoder is not None:
        encoder.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_encoder_plus_linear and encoder is not None:
                feats = encoder(images)
                logits = model(feats)
            else:
                logits = model(images)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return acc

import matplotlib.pyplot as plt
from collections import defaultdict
import statistics

def plot_results(results):
    """
    Takes a list of (method, framework, budget, seed, accuracy) records,
    averages accuracy across seeds, and creates one separate matplotlib
    line chart per framework.
    """
    # Step 1: Organize data by (framework -> (method -> dict of budget -> list of accuracies))
    # Example structure:
    # data["FullySup"]["Random"][budget] = [acc1, acc2, ... from multiple seeds]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for (method, framework, budget, seed, acc) in results:
        data[framework][method][budget].append(acc)

    # Step 2: For each framework, create a plot
    for framework, methods_dict in data.items():
        # Create a new figure for each framework
        plt.figure()
        plt.title(f"Budget vs. Accuracy - {framework}")

        # We'll iterate over "Random" and "TPC-RP" (or whatever methods you have)
        for method, budget_dict in methods_dict.items():
            # budget_dict is {budget: [acc1, acc2, ...], ...}
            # Compute average accuracy per budget across seeds
            sorted_budgets = sorted(budget_dict.keys())
            avg_accs = []
            for b in sorted_budgets:
                avg_acc = statistics.mean(budget_dict[b])
                avg_accs.append(avg_acc)

            # Plot a line from budget to average accuracy
            plt.plot(sorted_budgets, avg_accs, label=method)

        plt.xlabel("Budget (Number of Labeled Examples)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()


# ========================================
# 5. MAIN SCRIPT: COMPARE SELECTION & FRAMEWORKS
# ========================================
def main():
    # A. Load CIFAR-10
    train_dataset, test_dataset = get_cifar10_datasets()
    n_train = len(train_dataset)  # 50000

    # B. Prepare the SimCLR encoder (used for TPC-RP & the linear-head approach)
    encoder = SimCLRResNet18().to(DEVICE)
    # Load your custom checkpoint if available
    ckpt_path = "model/simclr_cifar_10.pth.tar"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        encoder.load_state_dict(ckpt, strict=False)
        print("Loaded SimCLR checkpoint.")
    else:
        print("SimCLR checkpoint not found; you can still run TPC-RP if the random init is used, but it won't be meaningful.")

    # C. Evaluate across multiple budgets
    budgets = [10, 20, 50]  # for a quick test, adjust as you like
    seeds = [0, 1]          # multiple seeds for each budget
    results = []            # store (method, framework, budget, seed, accuracy)

    for budget in budgets:
        for seed in seeds:
            # 1) Random selection
            rand_indices = select_random_samples(n_train, budget, seed=seed)

            # 2) TPC-RP selection
            #   (Note: If the checkpoint isn't meaningfully trained, TPC-RP won't be good. Just demonstrating logic.)
            typiclust_indices = select_typiclust_samples(encoder, train_dataset, budget=budget)

            # =========== FULLY SUPERVISED ===========
            print(f"\n=== [Fully Supervised] Budget={budget}, Seed={seed}, Random Selection ===")
            model_fs_rand = train_cnn_fully_supervised(train_dataset, rand_indices, epochs=5)  
            # Evaluate
            fs_rand_acc = evaluate_accuracy(model_fs_rand, test_dataset)
            print(f"FullySup (Random) - Test Acc = {fs_rand_acc*100:.2f}%")
            results.append(("Random", "FullySup", budget, seed, fs_rand_acc))

            print(f"\n=== [Fully Supervised] Budget={budget}, Seed={seed}, TPC-RP Selection ===")
            model_fs_tpc = train_cnn_fully_supervised(train_dataset, typiclust_indices, epochs=5)
            fs_tpc_acc = evaluate_accuracy(model_fs_tpc, test_dataset)
            print(f"FullySup (TPC-RP) - Test Acc = {fs_tpc_acc*100:.2f}%")
            results.append(("TPC-RP", "FullySup", budget, seed, fs_tpc_acc))

            # =========== SELF-SUPERVISED EMBEDDINGS (Linear Head) ===========
            print(f"\n=== [Linear Head] Budget={budget}, Seed={seed}, Random Selection ===")
            linear_rand = train_linear_head(encoder, train_dataset, rand_indices, epochs=5)
            lin_rand_acc = evaluate_accuracy(linear_rand, test_dataset,
                                             is_encoder_plus_linear=True, encoder=encoder)
            print(f"LinHead (Random) - Test Acc = {lin_rand_acc*100:.2f}%")
            results.append(("Random", "LinHead", budget, seed, lin_rand_acc))

            print(f"\n=== [Linear Head] Budget={budget}, Seed={seed}, TPC-RP Selection ===")
            linear_tpc = train_linear_head(encoder, train_dataset, typiclust_indices, epochs=5)
            lin_tpc_acc = evaluate_accuracy(linear_tpc, test_dataset,
                                            is_encoder_plus_linear=True, encoder=encoder)
            print(f"LinHead (TPC-RP) - Test Acc = {lin_tpc_acc*100:.2f}%")
            results.append(("TPC-RP", "LinHead", budget, seed, lin_tpc_acc))

            # =========== SEMI-SUPERVISED ===========
            # Here is a placeholder. In reality, you'd integrate your method (like FlexMatch).
            # For example:
            # unlabeled_indices = list(set(range(n_train)) - set(rand_indices))
            # model_semi_rand = train_semi_supervised(train_dataset, rand_indices, unlabeled_indices, epochs=5)
            # ...
            # Evaluate similarly.

    # D. Summarize results
    print("\n================== FINAL RESULTS ==================")
    plot_results(results)
    for row in results:
        method, framework, budget, seed, acc = row
        print(f"{framework} / {method} | B={budget} | Seed={seed} => Acc={acc*100:.2f}%")

if __name__ == "__main__":
    main()
