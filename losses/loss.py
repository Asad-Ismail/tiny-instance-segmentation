import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


def prepare_data(gts,preds,stride=8,device="cpu"):    
    gt_pts=gts["pts"]
    gt_offs=gts["offs"]
    gt_boxes=gts["bboxs"]
    gt_msks=gts["msks"]
    gt_centers=gts["center"]
    
    pred_offs=preds["offs"]
    pred_szs=preds["boxes"]
    
    fil_gt_box=[]
    fil_tgt_box=[] # For keeping the processed distance from centers
    fil_gt_msks=[]
    fil_gt_cts=[]
    fil_gt_offs=[]
    
    fil_pred_szs=[]
    fil_pred_offs=[]
    
    for i,pts in enumerate(gt_pts):
        fil_gt_box.append(gt_boxes[i,pts[:,0],pts[:,1]])
        fil_pred_szs.append(pred_szs[i,pts[:,0],pts[:,1]])
        
        mskidx=pts[:,0]*64+pts[:,1]      
        fil_gt_msks.append(gt_msks[i,mskidx])
        # convert y,x to x,y
        fil_gt_offs.append(torch.fliplr(gt_offs[i]))
        fil_pred_offs.append(pred_offs[i,pts[:,0],pts[:,1]])
        
        ## convert y,x to x,y for box adjustments 
        fil_gt_cts.append(torch.fliplr(pts))
        
        pboxesmin=(fil_gt_cts[-1]+fil_gt_offs[-1])-(fil_gt_box[-1][...,:2]/stride)
        pboxesmax=(fil_gt_box[-1][...,2:]/stride)-(fil_gt_cts[-1]+fil_gt_offs[-1])
        
        pboxes=torch.hstack((pboxesmin,pboxesmax))
        fil_tgt_box.append(pboxes)

    tgt_box=torch.concat(fil_tgt_box).to(device)
    tgt_msks=torch.concat(fil_gt_msks).to(device)
    
    pred_szs=torch.concat(fil_pred_szs).to(device)
    pred_offs=torch.concat(fil_pred_offs).to(device)
    
    tgt_offs=torch.concat(fil_gt_offs).to(device)
    
    return pred_szs,tgt_box,pred_offs,tgt_offs,tgt_msks,fil_gt_box

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

def calc_loss(gts,preds,device="cuda"):
    #n_positive= torch.sum(gt_cats>0) 
    #print(n_positive)
    pred_szs,tgt_szs,pred_offs,tgt_offs,tgt_msks,boxlist=prepare_data(gts,preds,stride=8.0,device=device)
    box_loss1=F.l1_loss(pred_szs,tgt_szs,reduction="mean")
    box_loss2=F.mse_loss(pred_szs,tgt_szs,reduction="mean")
    box_loss=box_loss1+box_loss2
    off_loss=F.l1_loss(pred_offs,tgt_offs,reduction="mean")
    gt_cats=gts["center"]
    pred_cats=preds["center"]
    ## Centerness Focal Loss
    gt_cats=gt_cats.unsqueeze(1)
    cat_loss=sigmoid_focal_loss(pred_cats,gt_cats,reduction="mean")
    cat_loss=cat_loss
    return {"cat_loss":cat_loss,"box_loss":box_loss,"off_loss":off_loss},tgt_msks,boxlist

DLoss=BinaryDiceLoss()

def calc_mskLoss(pred_msks,tgt_msks,loss_dict):
    msk_loss1=F.binary_cross_entropy_with_logits(pred_msks.squeeze(1), tgt_msks, reduction="mean")
    msk_loss2=DLoss(pred_msks.squeeze(1), tgt_msks)
    msk_loss=msk_loss1+msk_loss2
    loss_dict.update({"Mask_loss":msk_loss})
    return loss_dict