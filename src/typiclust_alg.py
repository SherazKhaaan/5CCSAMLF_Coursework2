import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.cluster import KMeans
import sys
from sklearn.metrics import pairwise_distances

# Add the current file's directory to the system path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Define the device for computation (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimCLRResNet18(nn.Module):
    """
    SimCLR Encoder Definition for Self-Supervised Representation Learning.
    - Uses a ResNet-18-based encoder for self-supervised learning.
    - The final fully connected layer is removed from ResNet-18 to extract a 512-dimensional feature vector.
    - A projection head is added to further refine the representation.
    
    This self-supervised approach is used to learn a semantic feature space on unlabeled data before clustering.
    """
    def __init__(self, feature_dim=128):
        super(SimCLRResNet18, self).__init__()
        
        # Load ResNet-18 backbone without pre-trained weights
        backbone = resnet18(pretrained=False)  # We load our own checkpoint later.
        
        # Remove the final fully connected layer to get feature extraction layers
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        
        # Define the projection head (MLP for feature refinement)
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),  # First linear layer reduces dimension
            nn.ReLU(),  # Non-linearity
            nn.Linear(512, feature_dim)  # Second linear layer maps to feature_dim
        )
    
    def forward(self, x):
        """
        Forward pass through the encoder to extract features.
        
        Parameters:
        x (Tensor): Input image batch.
        
        Returns:
        Tensor: Extracted features of shape (batch_size, 512).
        """
        h = self.encoder(x)  # Extract feature maps (batch_size, 512, 1, 1)
        h = h.view(h.size(0), -1)  # Flatten to (batch_size, 512)
        return h


def compute_embeddings(model, dataset, batch_size=512, num_workers=4):
    """
    Computes embeddings for all samples in the dataset using the given model.
    
    Steps:
    1. Create a DataLoader to iterate over the dataset.
    2. Pass each batch through the model to extract feature embeddings.
    3. Store the computed embeddings and corresponding indices.
    
    Parameters:
    model (nn.Module): Trained model for feature extraction.
    dataset (Dataset): Input dataset.
    batch_size (int): Number of samples per batch.
    num_workers (int): Number of worker threads for data loading.
    
    Returns:
    np.ndarray: Array of extracted embeddings.
    list: List of corresponding indices.
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()  # Set the model to evaluation mode
    embeddings_list = []
    indices_list = []
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch_idx, (images, _) in enumerate(data_loader):
            images = images.to(DEVICE)
            
            # Forward pass to extract features
            feats = model(images)
            embeddings_list.append(feats.cpu().numpy())
            
            # Track the dataset indices corresponding to the embeddings
            start_idx = batch_idx * data_loader.batch_size
            end_idx = start_idx + images.size(0)
            indices_list.extend(range(start_idx, end_idx))
    
    # Combine all batch embeddings into a single NumPy array
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    return all_embeddings, indices_list


def typical_clustering_selection(all_embeddings, budget=100, k_nn=20, random_state=42):
    """
    Clusters embeddings with KMeans and selects one typical sample per cluster.
        
    Steps:
    1. Cluster the embedded data points into `budget` clusters using KMeans to ensure diversity.
    2. Compute a "typicality" score for each point in a cluster based on the inverse of the average distance
       to its `k_nn` nearest neighbors.
    3. Select the most typical point from each cluster (i.e., the point in the densest region).
    
    Parameters:
    all_embeddings (np.ndarray): Embeddings for all samples.
    budget (int): Number of clusters to form (i.e., number of samples to select).
    k_nn (int): Number of nearest neighbors to consider for typicality calculation.
    random_state (int): Seed for reproducibility.
    
    Returns:
    list: Indices of selected typical samples.
    np.ndarray: Cluster labels for all samples.
    """
    
    # Apply KMeans clustering to divide data into 'budget' clusters
    kmeans = KMeans(n_clusters=budget, random_state=random_state)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    selected_indices = []
    
    # Iterate over each cluster and find the most typical sample
    for cluster_id in range(budget):
        cluster_idxs = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_idxs) == 0:
            continue  # Skip empty clusters
        
        cluster_embeds = all_embeddings[cluster_idxs]
        
        # Compute pairwise distances within the cluster
        distances = pairwise_distances(cluster_embeds, metric="euclidean")
        typicalities = []
        
        for i in range(distances.shape[0]):
            sorted_dists = np.sort(distances[i])[1:k_nn+1]  # Skip self-distance
            avg_dist = np.mean(sorted_dists)  # Compute mean distance to k nearest neighbors
            typicalities.append(1.0 / (avg_dist + 1e-8))  # Compute typicality score
        
        # Find the most typical sample (max typicality score)
        best_local_idx = np.argmax(typicalities)
        selected_indices.append(cluster_idxs[best_local_idx])
    
    return selected_indices, cluster_labels
