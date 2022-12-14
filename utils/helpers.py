import torch
import numpy as np
import cv2
import torch.nn.functional as F

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

def fixed_points(x,y,mask):
    """Fixed Mask if centroid is not in the middle go to left to find one valid point"""
    h,w=mask.shape
    while x<w and mask[y][x]==0:
        x+=1
    return (y,x)

def get_quantized_center(x,y,mask,dst_size=64,p_sofar=None):
    h,w=mask.shape
    cy=y*(dst_size/h)
    cx=x*(dst_size/w) 
    # Quantized centers
    cyq=int(cy)
    cxq=int(cx)
    # Offsets of centers
    offy=cy-cyq
    offx=cx-cxq
    # Make sure two objects are not assigned same center
    if (cyq,cxq) in p_sofar:
        assert False
    p_sofar.add((cyq,cxq))
    return cyq,cxq,offy,offx,p_sofar

def find_center(mask):
    (ys,xs)=np.nonzero(mask)
    points=list(zip(xs,ys))
    h,w=mask.shape
    assert max(ys)<h and max(xs)<w
    #horizontal cucumber
    midx=None
    midy=None
    if abs(max(ys)-min(ys))<abs(max(xs)-min(xs)):
        xs=sorted(xs)
        midx=xs[len(xs)//2]
        yrel=[y for  x,y in points if x==midx]
        yrel=sorted(yrel)
        midy=yrel[len(yrel)//2]
    else:
        ys=sorted(ys)
        midy=ys[len(ys)//2]
        xrel=[x for  x,y in points if y==midy]
        xrel=sorted(xrel)
        midx=xrel[len(xrel)//2]
    return (midy,midx)    

def find_center_cv(mask):
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    h,w=mask.shape
    c = max(contours, key = cv2.contourArea)
    M = cv2.moments(c)
    #calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cY,cX)


