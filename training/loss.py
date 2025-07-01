
# ================================
# 7. LOSS FUNCTIONS
# ================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for self-supervised pre-training
    """

    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        """
        Args:
            z_i, z_j: Embeddings of augmented views from the same signal
        """
        batch_size = z_i.size(0)
        device = z_i.device

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Positive pairs
        pos_sim = self.cosine_similarity(z_i, z_j) / self.temperature

        # Negative pairs
        neg_sim_i = torch.mm(z_i, z_j.T) / self.temperature
        neg_sim_j = torch.mm(z_j, z_i.T) / self.temperature

        # Create labels and compute loss
        labels = torch.arange(batch_size).to(device)
        loss_i = F.cross_entropy(neg_sim_i, labels)
        loss_j = F.cross_entropy(neg_sim_j, labels)

        loss = (loss_i + loss_j) / 2
        return loss


class DualNetLoss(nn.Module):
    """
    Multi-task loss function for DualNet
    """

    def __init__(self, alpha=0.45, beta=0.45, lambda_reg=0.01):
        super(DualNetLoss, self).__init__()
        self.alpha = alpha  # Pain loss weight
        self.beta = beta    # Stress loss weight
        self.lambda_reg = lambda_reg  # Regularization weight
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions, targets, task_types):
        """
        Args:
            predictions: Dict with 'pain' and/or 'stress' predictions
            targets: Ground truth labels
            task_types: List of task types in the batch
        """
        total_loss = 0.0
        loss_components = {}

        # Pain loss
        if 'pain' in predictions:
            pain_mask = torch.tensor(
                [t == 'pain' for t in task_types], device=targets.device)
            if pain_mask.sum() > 0:
                pain_pred = predictions['pain'][pain_mask]
                pain_target = targets[pain_mask]
                pain_loss = self.bce_loss(pain_pred, pain_target)
                total_loss += self.alpha * pain_loss
                loss_components['pain_loss'] = pain_loss.item()

        # Stress loss
        if 'stress' in predictions:
            stress_mask = torch.tensor(
                [t == 'stress' for t in task_types], device=targets.device)
            if stress_mask.sum() > 0:
                stress_pred = predictions['stress'][stress_mask]
                stress_target = targets[stress_mask]
                stress_loss = self.bce_loss(stress_pred, stress_target)
                total_loss += self.beta * stress_loss
                loss_components['stress_loss'] = stress_loss.item()

        # L2 regularization
        reg_loss = 0.0
        for param in list(predictions.parameters()):
            reg_loss += torch.norm(param, 2)
        total_loss += self.lambda_reg * reg_loss
        loss_components['reg_loss'] = reg_loss.item()

        loss_components['total_loss'] = total_loss.item()
        return total_loss, loss_components
