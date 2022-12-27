import timm
from einops import  repeat
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops import roi_align
import sys,os
sys.path.append(os.path.abspath('../losses'))
from loss import *

def min_max_norm(x,a,b):
    min_x=x.min()
    max_x=x.max()
    return a+((x-min_x)*(b-a)/(max_x-min_x))

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def decode_boxes(cx,cy,pred_szs,pred_offs,stride):
    cx = (cx+pred_offs[:,0]) * stride 
    cy = (cy+pred_offs[:,1]) * stride
    pred_szs*=stride
    x_min = cx - (pred_szs[:,0])
    y_min = cy - (pred_szs[:,1])
    x_max = cx + (pred_szs[:,2])
    y_max = cy + (pred_szs[:,3])
    result= torch.stack([x_min, y_min, x_max, y_max], axis=-1)
    return result.float()

def process_preds(preds,grid_sz=64,cat_th=0.4): 
    pred_cats=preds["center"].cpu()
    pred_cats=pred_cats.squeeze(0).sigmoid()
    pred_cats[pred_cats<cat_th]=0   
    nzidx=torch.nonzero(pred_cats)
    bidx=nzidx[:,0]    
    idx=nzidx[:,1]
    jdx=nzidx[:,2]
    mskidx=nzidx[:,1]*grid_sz+nzidx[:,2]
    ## Select boxes
    pred_boxes=preds["boxes"].cpu()
    pred_offs=preds["offs"].cpu()
    pred_boxes=pred_boxes[bidx,idx,jdx,:].detach().numpy()
    pred_offs=pred_offs[bidx,idx,jdx,:].detach().numpy()
    pred_boxes=decode_boxes(jdx,idx,pred_boxes,pred_offs,8.0)
    scores=pred_cats[bidx,idx,jdx]
    labels=torch.ones_like(scores).to(torch.long) 
    preds={"centers":pred_cats,"boxes":pred_boxes,"scores":scores,"labels":labels}
    return preds

def posEncoding(x):
    b=x.shape[0]
    xpos=torch.arange(x.shape[-1]).to(x)
    xpos=repeat(xpos, 'i -> i newaxis', newaxis=x.shape[-1])
    xpos=min_max_norm(xpos,-1,1).reshape(1,x.shape[-1],x.shape[-1])

    ypos=torch.arange(x.shape[-2]).to(x)
    ypos=repeat(ypos, 'i -> newaxis i', newaxis=x.shape[-2])
    ypos=min_max_norm(ypos,-1,1).reshape(1,x.shape[-2],x.shape[-2])

    xpos=repeat(xpos,'i j k -> b i j k',b=b)
    ypos=repeat(ypos,'i j k -> b i j k',b=b)
    ## Append 
    x=torch.cat([x,xpos,ypos],axis=1)
    return x


class tinyModel(nn.Module):
    def __init__(self,grid_sz=64,posEncoding=True) -> None:
        super(tinyModel, self).__init__()
        interchn=256
        self.grid_sz=grid_sz
        self.posEncoding=posEncoding
        self.backbone=timm.create_model('resnet18', pretrained=True, in_chans=3,num_classes=0, global_pool='',output_stride=8)
        #bb_channel=self.backbone.layer4[1].conv3.out_channels
        bb_channels=self.backbone.layer4[1].conv2.out_channels
        if posEncoding:
            bb_channels+=2
        ## Classification Head
        self.class_head=nn.Sequential(nn.Conv2d(bb_channels, interchn, kernel_size=3,padding=1),nn.ReLU())
        for i in range(5):
            self.class_head.add_module(f"conv_{i}",nn.Conv2d(interchn, interchn, kernel_size=3,padding=1))
            self.class_head.add_module(f"relu_{i}",nn.ReLU())
        self.class_head.add_module(f"cls",nn.Conv2d(interchn, 1, kernel_size=1))
        # classification size should be grid_sz x grid_sz
        ## Detection Head only h and width calculations
        self.box_head=nn.Sequential(nn.Conv2d(bb_channels, interchn, kernel_size=3,padding=1),nn.ReLU())
        for i in range(6):
            self.box_head.add_module(f"conv_{i}",nn.Conv2d(interchn, interchn, kernel_size=3,padding=1))
            self.box_head.add_module(f"relu_{i}",nn.ReLU())
        self.box=nn.Conv2d(interchn,4,3,padding=1)    
        # Scaling factor should be number of feature pyramid network one in the simple case
        self.regression_scales = torch.nn.Parameter(torch.ones((1), dtype=torch.float32))
        ## Offset Head 
        self.offset_head=nn.Sequential(nn.Conv2d(bb_channels, interchn, kernel_size=3,padding=1),nn.ReLU())
        for i in range(6):
            self.offset_head.add_module(f"conv_{i}",nn.Conv2d(interchn, interchn, kernel_size=3,padding=1))
            self.offset_head.add_module(f"relu_{i}",nn.ReLU())
        self.offset=nn.Conv2d(interchn,2,1,padding=0)   
        img_channels=3
        if posEncoding:
            img_channels+=2
        ## Segmentation Head
        self.seg_head=nn.Sequential(nn.Conv2d(img_channels, interchn, kernel_size=3,padding=1),nn.ReLU())
        for i in range(6):
            self.seg_head.add_module(f"conv_{i}",nn.Conv2d(interchn, interchn, kernel_size=3,padding=1))
            self.seg_head.add_module(f"relu_{i}",nn.ReLU())
        #self.upsample=Upsample(256)
        self.seg=nn.Conv2d(interchn,1,1,padding=0)    
        
    def forward(self,inp) -> torch.Tensor:
        ## For training expects the labels to be present also
        #img=inp["img"]
        img=inp
        x=self.backbone(img)
        if self.posEncoding:
            x=posEncoding(x)

        centers=self.class_head(x)

        box=self.box_head(x)
        box=self.box(box)
        box=box.permute(0,2,3,1)
        box=box*self.regression_scales
        box=F.relu(box)
        
        #box=torch.exp(box)
        off=self.offset_head(x)
        off=self.offset(off)
        off=off.permute(0,2,3,1)
        
        preds={"center":centers,"offs":off,"boxes":box}

        if self.training:
            assert "labels" in inp, "Labels must exist for training"
            loss_dict,tgt_msks,boxlist=calc_loss(inp["labels"],preds)
            pred_msks=roi_align(img,boxlist,output_size=64)
            if self.posEncoding:
                pred_msks=posEncoding(pred_msks)
            pred_msks=self.seg_head(pred_msks)
            pred_msks=self.seg(pred_msks)
            # Mask losses
            loss_dict=calc_mskLoss(pred_msks.squeeze(1), tgt_msks,loss_dict)

            accum_loss=0.0
            for k,v in loss_dict.items():
                if k=="box_loss":
                    accum_loss+=1.0*v
                else:
                    accum_loss+=v
            loss_dict["Total_Loss"]=accum_loss
            return {"preds":preds,"loss":loss_dict}
        else:
            preds=process_preds(preds)
            boxlist=[preds["boxes"]]
            pred_msks=roi_align(img,boxlist,output_size=64)
            if self.posEncoding:
                pred_msks=posEncoding(pred_msks)
            pred_msks=self.seg_head(pred_msks)
            pred_msks=self.seg(pred_msks)
            pred_msks=pred_msks.sigmoid()
            preds["masks"]=pred_msks
            return pred_msks
