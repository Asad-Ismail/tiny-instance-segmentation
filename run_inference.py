import torch.utils.data as data
import torch

model=tinyModel()
model.load_state_dict(torch.load("./cnet_pos_ckpts/437.pth"))

model.cpu()
model.eval()

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
test_loader = DataLoader(dataset=data,  batch_size=1,num_workers=4, shuffle=True)


def vis_results(img,masks):
    img=postprocess(img)
    for i in range(len(masks)):
        color=[np.random.randint(0,255) for _ in range(3)]
        img[masks[i]!=0]=color
    return img

def vis_boxes(img,boxes):
    img=postprocess(img)
    img=img.copy()
    for box in boxes:
        cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), (255,0,0), 2)
    return img

def vis_masks(img,msks,boxes,msk_th=0.4):
    imgh,imgw,_=img.shape
    for i in range(msks.shape[0]):
        box=boxes[i]
        xmin=box[0].long().item()
        xmax=box[2].long().item()
        ymin=box[1].long().item()
        ymax=box[3].long().item()
        
        xmin=max(0,xmin)
        xmax=max(0,xmax)
        ymin=max(0,ymin)
        ymax=max(0,ymax)
        ## To Take care of max mask
        xmax=min(xmax+1,imgw)
        ymax=min(ymax+1,imgh)
        w=xmax-xmin
        h=ymax-ymin
        
        msk=msks[i].unsqueeze(0)
        msk=F.interpolate(msk, size=(h,w), mode='bicubic',align_corners=True)
        msk[msk<msk_th]=0
        msk=msk.squeeze(0).squeeze(0)
        
        bmsk=msk>0
        color=[np.random.randint(0,255) for _ in range(3)]
        img[ymin:ymax,xmin:xmax][bmsk]=color
    return img


for i,batch in tqdm(enumerate(test_loader)):
    preds=model(batch)
    imgs=batch["img"]
    lbl=batch["center"]
    lbl_boxes=batch["bboxs"]
    boxes=preds["boxes"]
    pred_cats=preds["centers"]
    pred_msks=preds["masks"]
    bidx=0
    ## Draw Preds
    img=vis_boxes(imgs[bidx],boxes)
    img=vis_masks(img.copy(),pred_msks,boxes)
    ## Draw GT Boxes
    nz=torch.nonzero(lbl)
    for idx in range(len(nz)):
        lbl_box=lbl_boxes[nz[idx][0],nz[idx][1],nz[idx][2]].numpy()
        cv2.rectangle(img, (int(lbl_box[0]),int(lbl_box[1])), (int(lbl_box[2]),int(lbl_box[3])), (0,255,0), 2)
    #plt.figure(2)
    #plt.imshow(pred_cats.detach().squeeze(0).numpy())
    #plt.figure(3)
    #plt.imshow(img)
    #break
    cv2.imwrite(f"{res_dir}{i}_seg.png",img)
    cv2.imwrite(f"{res_dir}{i}_cent.png",pred_cats.detach().squeeze(0).numpy()*20)
    