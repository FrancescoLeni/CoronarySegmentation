import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# look at it, simply copied from chatgpt
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Apply softmax to get probabilities
        inputs = F.softmax(inputs, dim=1)
        targets = targets
        loss = 0.0

        for c in range(inputs.size(1)):
            # Get the mask for the current class
            mask = (targets == c)
            # Calculate the focal loss for this class
            p_t = inputs[:, c, :, :]
            loss -= self.alpha * mask * (1 - p_t) ** self.gamma * torch.log(p_t)

        return loss.mean()


class SmoothL2Loss(nn.Module):  # hubert loss
    def __init__(self, delta=1.0):
        super(SmoothL2Loss, self).__init__()
        self.delta = delta

    def forward(self, y_true, y_pred):
        diff = torch.abs(y_true - y_pred)
        loss = torch.where(diff < self.delta,
                           0.5 * diff**2,
                           self.delta * (diff - 0.5 * self.delta))
        return loss.mean()


def _reshape_mask(mask):
    return mask.reshape(mask.shape[0] * mask.shape[1], mask.shape[2] * mask.shape[3])


def dice_loss_gpt(inputs, targets, valid=None, target_logit=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = _reshape_mask(inputs)

    if target_logit:
        targets = targets.sigmoid()
    targets = _reshape_mask(targets)

    if valid is not None:
        valid = _reshape_mask(valid)
        inputs = inputs * valid
        targets = targets * valid

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

        self.running_dict = {'loss': 0.}

    def forward(self, x, y):
        loss = self.loss_fn(x, y)
        self.accumulate(loss)
        return loss

    def accumulate(self, batch_loss):
        self.running_dict['loss'] += batch_loss.item()

    def get_current_value(self, batch_n):
        n = batch_n + 1  # batch_n is [0, N-1]
        return self.running_dict['loss'] / n

    def reset(self):
        self.running_dict = {'loss': 0.}


class CELoss(nn.Module):
    def __init__(self, reduction='mean', weights=None):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction, weight=weights)

        self.running_dict = {'loss': 0.}

    def forward(self, x, y):
        loss = self.loss_fn(x, y)
        self.accumulate(loss)
        return loss

    def accumulate(self, batch_loss):
        self.running_dict['loss'] += batch_loss.item()

    def get_current_value(self, batch_n):
        n = batch_n + 1  # batch_n is [0, N-1]
        return self.running_dict['loss'] / n

    def reset(self):
        self.running_dict = {'loss': 0.}



class SemanticFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, gain=1.):
        """
            Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            - alpha: Weighting factor in range (0,1). (I think is useless for multiclass)
            - gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
            - weight: torch.Tensor of weights per class (expected normalized)
            - gain: additional gain (default = 1)
        """

        super().__init__()

        self.gain = gain
        self.alpha = alpha
        self.gamma = gamma

        self.weight = weight

        self.running_dict = {'loss': 0.}

    def forward(self, x, y):
        """
        Args:
            x: Logits as output of model with shape BxCxHxW.
            y: Masks tensor as integers with shae BxHxW.
        Returns:
            Loss: loss averaged
        """

        ce_loss = F.cross_entropy(x, y, reduction='none')  # shape BxHxW
        pt = torch.exp(-ce_loss)
        if isinstance(self.weight, torch.Tensor):
            alpha = torch.where(y == 1, self.weight[1], self.weight[0])
            alpha = alpha / alpha.mean()
        else:
            alpha = 1
        loss = alpha * (1 - pt) ** self.gamma * ce_loss  # shape BxHxW
        self.accumulate(loss.mean())

        return self.gain * loss.mean()

    def accumulate(self, batch_loss):
        self.running_dict['loss'] += batch_loss.item()

    def get_current_value(self, batch_n):
        n = batch_n + 1  # batch_n is [0, N-1]
        return self.running_dict['loss'] / n

    def reset(self):
        self.running_dict['loss'] = 0.


class SemanticDiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-6, gain=1.):
        """
            Dice Loss for Semantic Segmentation.

        Args:
            - weight: torch.Tensor of weights per class (expected normalized)
            - smooth: (float, optional): Smoothing term to avoid division by zero.
            - gain: additional gain (default = 1)
        """
        super().__init__()

        self.gain = gain
        self.smooth = smooth
        self.weight = weight  # shape C

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Predicted logits of shape [N, C, H, W].
            y (torch.Tensor): Ground truth labels of shape [N, H, W].

        Returns:
            loss: torch.Tensor Dice loss.
            dice: torch.Tensor with per class Dice inside batch

        """

        N, C = x.size()[:2]
        # Flatten spatial dimensions
        x = F.softmax(x, dim=1).view(N, C, -1)  # [N, C, *]

        y = y.view(N, -1)  # [N, *]

        # One-hot encoding for the target
        y_onehot = F.one_hot(y, num_classes=C).permute(0, 2, 1)  # [N, C, *]

        # Compute intersection and union
        intersection = torch.sum(x * y_onehot, dim=2)  # [N, C]
        union = torch.sum(x.pow(2), dim=2) + torch.sum(y_onehot, dim=2)  # [N, C]

        # Compute Dice coefficient
        dice = (2 * intersection + self.smooth) / (union + self.smooth)  # [N, C]

        # # Apply class weights if available
        # if isinstance(self.weight, torch.Tensor):
        #     weight = self.weight.to(x.device)
        #     dice = dice * weight  # [N, C]

        # Average over classes and batches
        loss = 1 - torch.mean(dice[1])  # [1] to ONLY foreground

        return self.gain * loss    # dice.mean(dim=0).to('cpu').squeeze().detach().numpy().astype(np.float16)[1]  # no bkg


class SemanticLosses(nn.Module):
    def __init__(self, alpha=1, gamma=2, lambdas: tuple = (0.5, 0.5), weight=None):
        """
            handler for the losses to be used Semantic Segmentation scenario

        args:
            - alpha: focal loss alpha parameter (I think is useless for multiclass usage, use weights instead)
            - gamma: focal loss gamma parameter
            - lambdas: tuple with relative losses weights (1st is focal, 2nd Dice)
            - weight: list of class weights of len == N (num_classes)
        """
        super().__init__()

        assert sum([*lambdas]) == 1, 'not normalized lambdas'

        self.loss1 = SemanticFocalLoss(alpha, gamma, weight, gain=1)
        self.loss2 = SemanticDiceLoss(weight, gain=0.85)

        self.lambda1, self.lambda2 = lambdas

        self.running_dict = {'loss': 0., 'focal_loss': 0., 'dice_loss': 0.}

    def forward(self, x, y):
        focal = self.loss1(x, y)
        dice_loss = self.loss2(x, y)
        if torch.isnan(focal).any() or torch.isinf(focal).any():
            raise ValueError("focal contains NaN or Inf!")
        if torch.isnan(dice_loss).any() or torch.isinf(dice_loss).any():
            raise ValueError("dice_loss contains NaN or Inf!")

        batch_loss = self.lambda1 * focal + self.lambda2 * dice_loss
        if torch.isnan(batch_loss).any() or torch.isinf(batch_loss).any():
            raise ValueError("batch_loss contains NaN or Inf!")

        self.accumulate(batch_loss, focal, dice_loss)

        return batch_loss

    def accumulate(self, batch_loss, focal_loss, dice_loss):
        self.running_dict['loss'] += batch_loss.item()
        self.running_dict['focal_loss'] += focal_loss.item()
        self.running_dict['dice_loss'] += dice_loss.item()
        # if not isinstance(self.running_dict['dice'], np.ndarray):
        #     self.running_dict['dice'] = dice
        # else:
        #     self.running_dict['dice'] += dice

    def get_current_value(self, batch_n, only_loss=True):
        n = batch_n + 1  # batch_n is [0, N-1]

        if only_loss:
            return self.running_dict['loss'] / n
        else:
            return {self.running_dict[k] / n for k in self.running_dict.keys()}

    def reset(self):
        self.running_dict = {'loss': 0., 'focal_loss': 0., 'dice_loss': 0.}


class DistillationLoss(nn.Module):
    def __init__(self, loss_fn='KL', temp=2.5):
        super().__init__()

        assert loss_fn in ['KL'], f'"{loss_fn}" not recognised as loss function. use "KL" or "CE" '
        if loss_fn == 'KL':
            self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        else:
            raise AttributeError(f'"{loss_fn}" not recognised as loss. use "KL" or "CE" ')

        self.softmax_s = nn.LogSoftmax(dim=1)
        self.softmax_t = nn.Softmax(dim=1)
        self.temp = temp

    def forward(self, s, t):
        # s = student, t = teacher
        return self.loss_fn(self.softmax_s(s / self.temp), self.softmax(t / self.temp))


class FullLossKD(nn.Module):
    def __init__(self, lamda_dist=0.25, alpha=1, gamma=2, lambdas_focal=(0.5, 0.5), weights=None):
        super().__init__()

        self.l_gt = SemanticLosses(alpha, gamma, lambdas_focal, weights)
        self.l_KD = DistillationLoss('KL', 2.5)

        self.lambda_dist = lamda_dist

        self.running_dict = {'loss': 0., 'gt_loss': 0., 'kd_loss': 0.} | \
                            {k: self.l_gt.running_dict[k] for k in self.l_gt.running_dict.keys() if k != 'loss'}

    def forward(self, x_s, gt, y_t):

        gt_loss = self.l_gt(x_s, gt)
        kd_loss = self.l_KD(x_s, y_t)

        loss = (1 - self.lambda_dist) * gt_loss + self.lambda_dist * kd_loss

        self.accumulate(loss, gt_loss, kd_loss)

        return loss

    def accumulate(self, batch_loss, loss_gt, loss_kd):
        self.running_dict['loss'] += batch_loss.item()
        self.running_dict['gt_loss'] += loss_gt.item()
        self.running_dict['kd_loss'] += loss_kd.item()

    def get_current_value(self, batch_n, only_loss=True):
        gt_dict = self.l_gt.get_current_value(batch_n, False)

        del gt_dict['loss']

        self.running_dict.update(gt_dict)
        n = batch_n + 1  # batch_n is [0, N-1]
        if only_loss:
            return self.running_dict['loss'] / n
        else:
            return {self.running_dict[k] / n for k in self.running_dict.keys()}

    def reset(self):
        self.l_gt.reset()
        self.running_dict = {'loss': 0., 'gt_loss': 0., 'kd_loss': 0.} | \
                            {k: self.l_gt.running_dict[k] for k in self.l_gt.running_dict.keys() if k != 'loss'}








