import torch
import torch.nn as nn
from torch.distributions import normal
import torch.nn.functional as F
import numpy as np


def focal_loss_new(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)   
    loss = (1 - p.detach()) ** gamma * input_values
    return loss.mean()


class ASLSingleLabel(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []  # prevent gpu repeated memory allocation
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target, reduction=None):
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        

        # ASL weights
        with torch.no_grad():
            self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)
            targets = self.targets_classes
            anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = torch.ones_like(xs_pos) - xs_pos
        xs_pos_new = torch.mul(xs_pos.clone(), targets.clone().detach()) #* targets.detach()
        xs_neg_new = torch.mul(xs_neg.clone(), anti_targets.clone().detach()) #* anti_targets.detach()
        asymmetric_w = torch.pow(1 - xs_pos_new - xs_neg_new,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes.mul_(1 - self.eps).add_(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss
    

class AGCL(nn.Module):

    def __init__(self, cls_num_list, m=0.5, weight=None, s=30, train_cls=False, noise_mul = 1., gamma=0., gamma_pos=0., gamma_neg=4., device=None):
        super(AGCL, self).__init__()
        cls_list = torch.FloatTensor(cls_num_list).to(device)
        m_list = torch.log(cls_list)
        m_list = m_list.max()-m_list
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.m = m
        self.simpler = normal.Normal(0, 1/3)
        self.train_cls = train_cls
        self.noise_mul = noise_mul
        self.gamma = gamma
        self.loss_func = ASLSingleLabel(gamma_pos=gamma_pos, gamma_neg=gamma_neg)


    def forward(self, cosine, target):
        index = torch.zeros_like(cosine, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        noise = self.simpler.sample(cosine.shape).clamp(-1, 1).to(cosine.device)

        cosine = cosine - self.noise_mul * noise.abs()/self.m_list.max() *self.m_list
        output = torch.where(index, cosine-self.m, cosine)
        if self.train_cls:
            return focal_loss_new(F.cross_entropy(self.s*output, target, reduction='none', weight=self.weight), self.gamma)
        else:
            return self.loss_func(self.s*output, target)


class DiverseExpertLoss(nn.Module):
    def __init__(self, cls_num_list=None,  max_m=0.5, s=30, tau=2, device=None):
        super().__init__()
        self.base_loss = F.cross_entropy
     
        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().to(device)
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = tau 
        self.m1 = 0.1 * 20
        self.m2 = 0.1 * 20
        self.m3 = 0.1 * 20

        # self.uniform = torch.tensor([1. / len(cls_num_list) for _ in range(len(cls_num_list))] ).to(device)
        # self.base_loss = ASLSingleLabel(gamma_pos=0.5, gamma_neg=8.0)

    def inverse_prior(self, prior): 
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        
        return inverse_prior

    def forward(self, target, extra_info=None):
        loss = 0

        # Obtain logits from each expert  
        expert1_logits = extra_info['head_logits']
        expert2_logits = extra_info['medium_logits']
        expert3_logits = extra_info['few_logits']  

        # index = torch.zeros_like(expert1_logits, dtype=torch.uint8)
        # index.scatter_(1, target.data.view(-1, 1), 1)

        # expert1_logits = torch.where(index, expert1_logits-self.m1, expert1_logits)
        # expert2_logits = torch.where(index, expert2_logits-self.m2, expert2_logits)
        # expert3_logits = torch.where(index, expert3_logits-self.m3, expert3_logits)

        # Softmax loss for expert 1 
        loss = loss + self.base_loss(expert1_logits, target)
        
        # Balanced Softmax loss for expert 2 
        expert2_logits = expert2_logits + torch.log(self.prior + 1e-9) 
        loss = loss + self.base_loss(expert2_logits, target)
        
        # Inverse Softmax loss for expert 3
        inverse_prior = self.inverse_prior(self.prior)
        expert3_logits = expert3_logits + torch.log(self.prior + 1e-9) - self.tau * torch.log(inverse_prior+ 1e-9) 
        loss = loss + self.base_loss(expert3_logits, target)

        reg_loss = -1 *  extra_info['reg']

        loss = loss + reg_loss #* 0.0001
        return loss 
    

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)