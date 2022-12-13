import deeplake as hub
import numpy.random as random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import torch
import torchvision

import logging


model=tinyModel()
model.to("cuda")

import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-4)


model.train()
pidx=10
allres=[]
device="cuda"
for epoch in range(0,500):  # loop over the dataset multiple times
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
        
        pred_dict=model(inp)
        
        losses=pred_dict["loss"]
        loss=losses["Total_Loss"]
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % pidx == pidx-1:
            print(f'[{epoch + 1}, {i + 1:5d}] Running Loss: {running_loss / pidx:.3f}')
            #print(f"LR,{scheduler._last_lr}")
            for k,v in losses.items():
                print(f"{k}: {v.item()}")
            running_loss = 0.0
            
        #scheduler.step()
    if epoch % 2 == 1:
        PATH=f"./cnet_pos_ckpts/{epoch}.pth"
        torch.save(model.state_dict(), PATH)