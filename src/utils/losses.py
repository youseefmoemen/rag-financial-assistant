import torch
import torch.nn.functional as F


def info_nce_loss(query_embedding, doc_embedding, tempetature=0.05):
    """
    query_embedding: shape (B, D)
    doc_embedding: shape (B, D)
    """
    query_embedding = F.normalize(query_embedding, dim=1)
    doc_embedding = F.normalize(doc_embedding, dim=1)

    sim_matrix = torch.matmul(query_embedding, doc_embedding.T) / tempetature
    labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
    return F.cross_entropy(sim_matrix, labels)


