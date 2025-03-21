import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.cluster import KMeans
import sys
from sklearn.metrics import pairwise_distances
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# SimCLR Encoder Definition
# ---------------------------
class SimCLRResNet18(nn.Module):
    """
    SimCLR Encoder Definition for Self-Supervised Representation 
        - ResNet-18-based encoder for self-supervised learning 
        - Final fully connected layer is removed from ResNet-18 to extract a 512-dimensional feature
    A self-supervised approach is used to learn a semantic feature space on unlabeled data before clustering.

    Parameters:
        - 
    """
    def __init__(self, feature_dim=128):
        super(SimCLRResNet18, self).__init__()
        backbone = resnet18(pretrained=False)  # We load our own checkpoint later.
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
    
    def forward(self, x):
        """
        Pass images through the encoder (all layer except the final fully connected layer)
        """
        h = self.encoder(x)            # Shape: (batch, 512, 1, 1)
        h = h.view(h.size(0), -1)        # Flattens to (batch, 512)
        return h


def compute_embeddings(model, dataset, batch_size=512, num_workers=4):
    """
    Computes embeddings for all samples in the dataset using the given model.
    Returns a numpy array of embeddings and a list of corresponding indices.
    Given a trained model (self-supervised encoder) and a dataset
        - 1) Create a DataLoader to iterate over the CIFAR-10 dataset
        - 2) Run a forward pass to extract embeddings for each sample
        - 3) Return the embeddings (as a NumPy array) and the sample indices 
    
    Once a self-supervised representation has been learned, embed all unlabeled data points into that space
    ... where distances are semantically meaningful. 
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    embeddings_list = []
    indices_list = []
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            images = images.to(DEVICE)
            
            # Forward pass to get features 
            feats = model(images)
            embeddings_list.append(feats.cpu().numpy())

            # Keeps track of which data indices correspond to these embeddings
            start_idx = batch_idx * data_loader.batch_size
            end_idx = start_idx + images.size(0)
            indices_list.extend(range(start_idx, end_idx))
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    return all_embeddings, indices_list




def typical_clustering_selection(all_embeddings, budget=100, k_nn=20, random_state=42):
    """
    Clusters embeddings with KMeans and selects one typical sample per cluster.
    Typicality is defined as the inverse of the average distance to the k_nn nearest neighbors.
    Returns the list of selected indices and the cluster labels.
    Implements the "TypiClust" - Typical clustering logic ahdering to the method of Hacohen et al. (2022)
    
    1) Cluster the embedded data points into "budget" clusters to ensure diversity.
        - Each cluster tries to capture some meaningful region in the feature space.
    
    2) Within each cluster, compute a "typicality" score for each point based on average distance to k nearest neighbours.
        - Smaller average distance implies point lies in a denser part of the cluster - more typical
        - typicality(x) = 1 / (average_distance_to_k_neighbours + )

    3) Pick the most "typical" sample from each cluster, return the indices.
        - Chosen example is representative of cluster - hence data distribution - addresssing the "cold start" problem for low-budget regimes

    Parameters:
        - 
    """
    kmeans = KMeans(n_clusters=budget, random_state=random_state)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    selected_indices = []
    for cluster_id in range(budget):
        cluster_idxs = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_idxs) == 0:
            continue
        cluster_embeds = all_embeddings[cluster_idxs]
        distances = pairwise_distances(cluster_embeds, metric="euclidean")
        typicalities = []
        for i in range(distances.shape[0]):
            sorted_dists = np.sort(distances[i])[1:k_nn+1]  # skip self-distance
            avg_dist = np.mean(sorted_dists)
            typicalities.append(1.0 / (avg_dist + 1e-8))
        best_local_idx = np.argmax(typicalities)
        selected_indices.append(cluster_idxs[best_local_idx])
    return selected_indices, cluster_labels 
