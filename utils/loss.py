import torch
import torch.nn.functional as F

    
def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, sig: bool=True):
    if sig == True:
        inputs = inputs.sigmoid().flatten(1) 
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)  
    denominator = inputs.sum(-1) + targets.sum(-1)  

    loss = 1 - (numerator + 1) / (denominator + 1) 
    
    return loss.mean()  


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    loss = F.binary_cross_entropy_with_logits(inputs.flatten(1), targets.flatten(1), reduction="none") 
    return loss.mean() 
