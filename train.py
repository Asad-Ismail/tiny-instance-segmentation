import deeplake as hub
import numpy.random as random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import torch
import torchvision
import logging
import argparse
import torch.optim as optim
from datasets.data_loader import DataLoader as dLoader
from datasets.data_loader import collate_batch
import os, datetime
#from models.tinyism import tinyModel
from torch.utils.data import DataLoader
datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="hub://aismail2/cucumber_OD",help="Activeloop data path")
parser.add_argument("--size",default=512,type=int,help="Image size used for training model")
parser.add_argument("--epochs",default=1000,type=int,help="Image size used for training model")
parser.add_argument("--device", default="cuda",help="Device to Train Model")
parser.add_argument("--batch_sz", default=2,type=int,help="Device to Train Model")
parser.add_argument("--pretrain",default="./weights/RepVGGplus-L2pse-train256-acc84.06.pth",type=str,help="Pretrained weights")
parser.add_argument("--model_arch",default="repvggplus", choices=['resnet', 'repvgg','repvggplus'],type=str,help="Model Architecture")
parser.add_argument("--posencoding",default=True,type=bool,help="Positional Encoding")
parser.add_argument("--outdir",default="./output",type=str,help="Output of weights")

args = parser.parse_args()
data_path = args.dataset
img_sz = args.size
dname = args.device
pretrain= args.pretrain
outdir=args.outdir
epochs=args.epochs
modelarch=args.model_arch
posencoding=args.posencoding
BATCH_SIZE=args.batch_sz

outdir=f"{outdir}/{datestring}"
print(f"Trained Weight are stored in {outdir}")
os.makedirs(outdir)
assert os.path.exists(outdir), "Output dir does not exist"

# Model
if modelarch=="resnet":
    from models.tinyism import tinyModel
    model=tinyModel(posEncoding=posencoding)
    if pretrain:
        print(f"Loading Pretrained Weights!!")
        model.load_state_dict(torch.load(pretrain,map_location=torch.device('cpu')))
elif modelarch=="repvgg":
    from models.repvgg_tinyism import tinyModel
    model=tinyModel(posEncoding=posencoding,deploy=False)
    if pretrain:
        print(f"Loading Pretrained Weights!!")
        model.load_state_dict(torch.load(pretrain,map_location=torch.device('cpu')))
elif modelarch=="repvggplus":
    from models.repvggplus_tinyism import tinyModel
    model=tinyModel(posEncoding=posencoding,deploy=False)
    if pretrain:
        print(f"Loading Pretrained Weights!!")
        model.backbone.load_state_dict(torch.load(pretrain,map_location=torch.device('cpu')),strict=False)


device=torch.device(dname)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
model.train()

if __name__=="__main__":
    ds = hub.load(data_path)
    data=dLoader(ds=ds)
    data_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE,num_workers=1,collate_fn=collate_batch,shuffle=True)
    visidx=10
    saveepoch=10
    for epoch in range(0,epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i,batch in tqdm(enumerate(data_loader)):
            imgs=batch["img"].to(device)
            gt_centers=batch["pts"]
            gt_offsets=batch["offs"]
            gt_boxes=batch["bboxs"].to(device)
            gt_cats=batch["center"].to(device)
            gt_msks=batch ["msks"].to(device)
            for j,item in enumerate(gt_centers):
                gt_centers[j]=item.to(device)
            for j,item in enumerate(gt_offsets):
                gt_offsets[j]=item.to(device)
                
            optimizer.zero_grad()
            inp={"img":imgs}
            labels={"pts":gt_centers,"offs":gt_offsets,"bboxs":gt_boxes,"center":gt_cats,"msks":gt_msks}
            inp["labels"]=labels
            # Prediction of Model

            pred_dict=model(inp)

            losses=pred_dict["loss"]
            loss=losses["Total_Loss"]
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % visidx == visidx-1:
                print(f'[{epoch + 1}, {i + 1:5d}] Running Loss: {running_loss / visidx:.3f}')
                print(f"LR,{scheduler._last_lr}")
                for k,v in losses.items():
                    print(f"{k}: {v.item()}")
                running_loss = 0.0
                
        scheduler.step()
        if epoch % saveepoch == 1:
            PATH=f"{outdir}/{epoch}.pth"
            torch.save(model.state_dict(), PATH)