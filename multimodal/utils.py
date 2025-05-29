import torch 
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1, reduction='mean', pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, weight=self.pos_weight, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        # logits: [batch_size, num_labels], targets: [batch_size, num_labels]
        probs = torch.sigmoid(logits)  # 예측 확률 (0~1)
        batch_size, num_labels = probs.shape

        dice_loss = 0
        for i in range(num_labels):
            p = probs[:, i]  # i번째 레이블의 예측 확률
            t = targets[:, i]  # i번째 레이블의 실제 값
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice_score = (2 * intersection + self.epsilon) / (union + self.epsilon)
            dice_loss += 1 - dice_score

        return dice_loss / num_labels  # 레이블별 평균 손실

# class CombinedLoss(nn.Module):
#     def __init__(self, pos_weight, dice_weight=0.5, epsilon=1e-8):
#         super().__init__()
#         self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#         self.dice_loss = DiceLoss(epsilon=epsilon)
#         self.dice_weight = dice_weight

#     def forward(self, logits, targets):
#         bce_loss = self.bce(logits, targets)
#         dice_loss = self.dice_loss(logits, targets)
#         return (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss

class CombinedLoss(nn.Module):
    def __init__(self, pos_weight, alpha=0.99, gamma=2, dice_weight=0.3, focal_weight=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        probs = torch.sigmoid(logits)
        focal_loss = -self.alpha * (1 - probs) ** self.gamma * targets * torch.log(probs + 1e-8) \
                     - (1 - self.alpha) * probs ** self.gamma * (1 - targets) * torch.log(1 - probs + 1e-8)
        focal_loss = focal_loss.mean()
        return (1 - self.dice_weight - self.focal_weight) * bce_loss + self.dice_weight * dice_loss + self.focal_weight * focal_loss