# src/visualisation.py

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def set_seed(seed=30):
    """
    Sets the seed for reproducibility across numpy, torch, and CUDA.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def plot_tsne(embeddings, cluster_assignments, selected_indices=None,
                       title="t-SNE Visualization: Clusters + Selected Points",
                       n_samples=2000, save_path=None):
    """
    Plots a t-SNE of 'embeddings' color-coded by cluster_assignments.
    Then marks 'selected_indices' with an 'x' marker (if provided).
    
    Parameters:
        embeddings (np.array): shape (N, D) array of embeddings.
        cluster_assignments (np.array): shape (N,) integer array of cluster IDs.
        selected_indices (list, optional): indices of selected "typical" points.
        title (str): figure title.
        n_samples (int): if embeddings > n_samples, randomly subsample for faster t-SNE.
        save_path (str, optional): path to save the figure if not None.
    """
    N = embeddings.shape[0]
    if n_samples < N:
        # Randomly sample
        rand_idxs = np.random.choice(N, size=n_samples, replace=False)
        # Ensure selected points appear if they are in the sample
        if selected_indices is not None:
            rand_idxs = np.union1d(rand_idxs, selected_indices)
        emb_subset = embeddings[rand_idxs]
        cluster_subset = cluster_assignments[rand_idxs]
        subset_indices = rand_idxs
    else:
        emb_subset = embeddings
        cluster_subset = cluster_assignments
        subset_indices = np.arange(N)
    
    # t-SNE on the subset
    tsne = TSNE(n_components=2, random_state=42, init='pca')
    emb_2d = tsne.fit_transform(emb_subset)
    
    plt.figure(figsize=(8, 8))
    
    # Plot all points color-coded by their cluster.
    # The cluster_assignments array might be large, so for a max of 30 clusters, we just cycle colors.
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1],
                          c=cluster_subset, cmap='tab20', alpha=0.7, s=20,
                          label='Data (by cluster)')
    
    plt.colorbar(scatter, label="Cluster ID")
    
    # Mark selected indices with an 'x' marker in black
    if selected_indices is not None:
        mask = np.isin(subset_indices, selected_indices)
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                    c='black', marker='x', s=60, label='Selected')
    
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE cluster plot to {save_path}")
    plt.show()


def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth.tar"):
    """
    Saves the model and optimizer states along with the current epoch.
    
    Parameters:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): Current epoch number.
        filename (str): File name for the checkpoint.
    """
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename):
    """
    Loads model and optimizer states from a checkpoint.
    
    Parameters:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        filename (str): Path to the checkpoint file.
    
    Returns:
        int or None: The epoch number if a checkpoint is loaded; otherwise, None.
    """
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        print(f"Loaded checkpoint '{filename}' (epoch {epoch})")
        return epoch
    else:
        print(f"No checkpoint found at '{filename}'")
        return None


def plot_selected_images_by_label(dataset, selected_indices, label_array,
                                  title="",
                                  save_path=None):
    """
    Displays selected images in a grid format, grouped by their labels.
    Each column represents a specific label, and images are arranged accordingly.
    
    Parameters:
        dataset (torch.utils.data.Dataset): The dataset containing the images
        selected_indices (list): Indices of selected images
        label_array (np.array or list): An array of ground-truth labels corresponding to dataset indices
        title (str, optional): Title of the figure. Default is an empty string
        save_path (str, optional): Path to save the figure if provided
    """
    # CIFAR-10 class names
    cifar10_labels = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }

    # Group selected indices by label
    selected_by_label = {}
    for idx in selected_indices:
        label = label_array[idx]
        if label not in selected_by_label:
            selected_by_label[label] = []
        selected_by_label[label].append(idx)
    
    # Sort labels to ensure a consistent column order.
    all_labels = sorted(selected_by_label.keys())
    num_labels = len(all_labels)
    
    # Determine the number of rows needed - maximum number of selected images for any label.
    rows_needed = max(len(selected_by_label[lbl]) for lbl in all_labels)
    
    fig, axes = plt.subplots(nrows=rows_needed, ncols=num_labels, figsize=(2*num_labels, 2*rows_needed))
    
    # If only one row then ensure axes is 2D for easier indexing.
    if rows_needed == 1:
        axes = np.array([axes])
    
    for col, lbl in enumerate(all_labels):
        these_indices = selected_by_label[lbl]  # all indices for this label
        for row in range(rows_needed):
            ax = axes[row, col]
            if row < len(these_indices):
                idx = these_indices[row]
                img, _ = dataset[idx]
                # Unnormalize the image using CIFAR-10 statistics 
                mean = torch.tensor([0.4914, 0.4822, 0.4465])
                std = torch.tensor([0.247, 0.243, 0.261])
                unnorm_img = img * std[:, None, None] + mean[:, None, None]
                img_np = unnorm_img.permute(1, 2, 0).cpu().numpy().clip(0, 1)
                ax.imshow(img_np)
            else:
                ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f" {lbl} {cifar10_labels[lbl]}")
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved selected images grid to {save_path}")
    plt.show()
