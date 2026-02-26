import torch
import torch.nn as nn
import torch.nn.functional as F

def cb_focal_loss(logits: torch.Tensor, targets: torch.Tensor, class_counts: dict, gamma: float = 1.5, beta: float = 0.999, device: str = 'cuda') -> torch.Tensor:
    """Class-Balanced Focal Loss"""
    num_classes = logits.size(1)
    
    n = torch.tensor([class_counts.get(c, 1) for c in range(num_classes)], 
                     device=device, dtype=torch.float)
    weights = (1 - beta) / (1 - torch.pow(beta, n) + 1e-8)
    weights = weights / weights.mean()
    
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    pt = p[torch.arange(len(targets)), targets]
    focal = (1 - pt).pow(gamma) * (-logp[torch.arange(len(targets)), targets])
    
    w = weights[targets]
    return (w * focal).mean()


@torch.cuda.amp.autocast(enabled=False)
def coral_loss(feat_s: torch.Tensor, feat_t: torch.Tensor, shrink: float = 1e-3) -> torch.Tensor:
    """Correlation alignment loss."""
    xs = feat_s.float() - feat_s.float().mean(dim=0, keepdim=True)
    xt = feat_t.float() - feat_t.float().mean(dim=0, keepdim=True)
    ns, nt = max(xs.shape[0]-1, 1), max(xt.shape[0]-1, 1)
    Cs = (xs.T @ xs) / ns
    Ct = (xt.T @ xt) / nt
    I = torch.eye(Cs.shape[0], device=Cs.device, dtype=Cs.dtype)
    Cs = (1 - shrink) * Cs + shrink * I
    Ct = (1 - shrink) * Ct + shrink * I
    d = Cs.shape[0]
    return torch.mean((Cs - Ct)**2) / (4.0 * d * d)


@torch.cuda.amp.autocast(enabled=False) 
def mmd_loss(feat_s: torch.Tensor, feat_t: torch.Tensor, kernel: str = 'rbf', bandwidth: float = None) -> torch.Tensor:
    """Maximum Mean Discrepancy loss for domain alignment"""
    feat_s = feat_s.float()
    feat_t = feat_t.float()
    
    def rbf_kernel(x, y, bandwidth):
        xx = torch.sum(x**2, dim=1, keepdim=True)
        yy = torch.sum(y**2, dim=1, keepdim=True)
        xy = torch.mm(x, y.t())
        dist = xx + yy.t() - 2 * xy
        return torch.exp(-dist / (2 * bandwidth**2 + 1e-8))
    
    ns, nt = feat_s.size(0), feat_t.size(0)
    if ns < 2 or nt < 2:
        return torch.tensor(0.0, device=feat_s.device)
    
    if bandwidth is None:
        with torch.no_grad():
            diff = feat_s.unsqueeze(1) - feat_t.unsqueeze(0) 
            dists = torch.sqrt((diff ** 2).sum(dim=2) + 1e-8) 
            bandwidth = torch.median(dists) + 1e-6
    
    Kss = rbf_kernel(feat_s, feat_s, bandwidth)
    Ktt = rbf_kernel(feat_t, feat_t, bandwidth)
    Kst = rbf_kernel(feat_s, feat_t, bandwidth)
    
    mmd = (Kss.sum() / (ns * ns) + Ktt.sum() / (nt * nt) - 2 * Kst.sum() / (ns * nt))
    return mmd.clamp(min=0)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss for few-shot learning"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    @torch.cuda.amp.autocast(enabled=False) 
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        features = features.float()
        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature
        
        labels_view = labels.view(-1, 1)
        mask_positive = (labels_view == labels_view.T).float()
        mask_positive.fill_diagonal_(0) 
        
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        exp_logits = torch.exp(logits)
        
        mask_self = torch.eye(batch_size, device=device)
        exp_logits = exp_logits * (1 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)
        
        num_positives = mask_positive.sum(dim=1)
        mean_log_prob_pos = (mask_positive * log_prob).sum(dim=1) / (num_positives + 1e-6)
        
        valid_mask = num_positives > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        loss = -mean_log_prob_pos[valid_mask].mean()
        return loss

def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor = None, T: float = 4.0, alpha: float = 0.7) -> torch.Tensor:
    """Knowledge Distillation loss"""
    p_student = F.log_softmax(student_logits / T, dim=1)
    p_teacher = F.softmax(teacher_logits / T, dim=1)
    
    loss_kd = F.kl_div(p_student, p_teacher, reduction='batchmean') * (T * T)
    
    if labels is not None:
        loss_ce = F.cross_entropy(student_logits, labels)
        loss = alpha * loss_kd + (1 - alpha) * loss_ce
    else:
        loss = loss_kd
    return loss


class BalancedFocalLoss(nn.Module):
    """Focal Loss with per-class alpha"""
    def __init__(self, alpha: list = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha))
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, classes: int, smoothing: float = 0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        num = 2 * (probs * targets).sum(dims) + self.eps
        den = (probs * probs).sum(dims) + (targets * targets).sum(dims) + self.eps
        dice = num / den
        return 1 - dice.mean()

class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss"""
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.wb = bce_weight
        self.wd = dice_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.wb * self.bce(logits, targets) + self.wd * self.dice(logits, targets)
