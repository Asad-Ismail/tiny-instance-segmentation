import torch
import numpy as np
import cv2
import torch.nn.functional as F


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()
    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)

        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        throughput = 30 * batch_size / (tic2 - tic1)
        logger.info(f"batch_size {batch_size} throughput {throughput}")
        return

def vis_results(img,masks):
    img=postprocess(img)
    for i in range(len(masks)):
        color=[np.random.randint(0,255) for _ in range(3)]
        img[masks[i]!=0]=color
    return img

def vis_boxes(img,boxes,postprocess):
    img=postprocess(img)
    img=img.copy()
    for box in boxes:
        box=box.type(torch.int).numpy()
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


def find_center_cv(mask):
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    h,w=mask.shape
    c = max(contours, key = cv2.contourArea)
    M = cv2.moments(c)
    #calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cY,cX)


