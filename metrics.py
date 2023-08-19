import torch
import torch.nn as nn
import torch.nn.functional as F

def true_positive(y_true,y_pred):
    return ((y_true == 1) & (y_pred == 1)).sum().item()

def true_negative(y_true,y_pred):
    return ((y_true == 0) & (y_pred == 0)).sum().item()

def false_positive(y_true,y_pred):
    return ((y_true == 0) & (y_pred == 1)).sum().item()

def false_negative(y_true,y_pred):
    return ((y_true == 1) & (y_pred == 0)).sum().item()


def precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return tp / (tp + fp + 1e-7)

def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return tp / (tp + fn + 1e-7) 

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec + 1e-7)

def accuracy(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    total_samples = y_true.shape[0]
    return (tp + tn) / total_samples



class F1ScoreCrossEntropyLoss(nn.Module):
    def __init__(self,weight=None,alpha=1.0, beta=0.5, epsilon=1e-7):
        super(F1ScoreCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)

    def forward(self, y_pred, y_true):
        _, y_pred_final = torch.max(y_pred, 1)
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred_final.cpu().numpy()
        f1 = f1_score(y_true_np, y_pred_np)
        ce_loss = self.cross_entropy(y_pred, y_true)
        
        loss = self.alpha * ce_loss - self.beta * torch.log(torch.tensor(f1) + self.epsilon) + 1
        return loss.mean()
    


# this loss caculate new weight base on f1score each mini batch 
class F1_Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        y_true_one_hot = F.one_hot(y_true.to(torch.int64), 2).to(torch.float32)
        
        tp = (y_true_one_hot * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true_one_hot) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true_one_hot) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true_one_hot * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        f1=f1.detach()
        CE =torch.nn.CrossEntropyLoss(weight=( 1 - f1))(y_pred, y_true_one_hot)
        return  CE.mean()