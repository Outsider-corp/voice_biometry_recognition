import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConstrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super(ConstrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, cos, label):
        distance = 1 - cos
        loss_contrastive = torch.mean(0.5 * (1 - label) * torch.pow(distance, 2) +
                                      0.5 * label * torch.pow(F.relu(self.margin - distance), 2))
        return loss_contrastive


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = batch_cosine_similarity(anchor, positive)
        negative_distance = batch_cosine_similarity(anchor, negative)
        losses = F.relu(self.margin - (positive_distance - negative_distance))
        return losses.mean()


def cosine_similarity_triplet(vec1, vec2):
    return torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0), dim=1)


def cosine_similarity(x, y):
    # Вычисляем числитель: скалярное произведение векторов x и y
    x = x.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()
    dot_product = np.dot(x, y)

    # Вычисляем знаменатель: произведение евклидовых норм векторов x и y
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    # Убедимся, что знаменатель не равен нулю
    if norm_x == 0 or norm_y == 0:
        return 0
    else:
        # Вычисляем косинусное сходство
        cosine_similarity = dot_product / (norm_x * norm_y)
        return cosine_similarity


def batch_cosine_similarity(x, y):
    """
    Вычисляет косинусное сходство между двумя батчами векторов.
    """
    dot_product = (x * y).sum(dim=1)
    norm_x = x.norm(p=2, dim=1)
    norm_y = y.norm(p=2, dim=1)
    cosine_similarity = dot_product / (norm_x * norm_y)
    return cosine_similarity


def cosine_similarity_pair(vec1, vec2):
    return torch.nn.functional.cosine_similarity(vec1, vec2, dim=1)


def triplet_accuracy(anchor, positive, negative, margin=0.1):
    # pos_dist = cosine_similarity_triplet(anchor, positive)
    pos_dist = batch_cosine_similarity(anchor, positive)
    neg_dist = batch_cosine_similarity(anchor, negative)
    return ((pos_dist - neg_dist) > margin).float().mean()


def pair_accuracy(cos, labels, threshold: float = 0.5):
    predictions = (cos >= threshold).int()
    is_correct = (predictions == labels)
    accuracy = is_correct.float().mean()
    return accuracy
