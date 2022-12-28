from torchvision.transforms import Compose, Resize,Normalize,Lambda
import torch
import torch.utils.data as data
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import cv2



mean=torch.tensor([0.4850, 0.4560, 0.4060])
std=torch.tensor([0.2290, 0.2240, 0.2250])

preprocess = Compose([
            Lambda(lambda t: t/255.),
            Lambda(lambda t: t.permute(2, 0, 1)), # HWC to CHW
            Normalize(mean, std)
])

postprocess = Compose([
     Lambda(lambda t: (t.cpu() * std.reshape(3,1,1))+mean.reshape(3,1,1)),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
])

def collate_batch(batch):
    inputs={}
    for k in batch[0].keys():
        inps=[item[k] for item in batch]
        if k!="pts" and k!="offs":
            inps=torch.stack(inps)
        inputs[k]=inps
    return inputs

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

class DataLoader(data.Dataset):
    def __init__(self,ds,img_sz=512,grid_sz=64,seg_sz=64,data=None):
        super(DataLoader, self).__init__()
        self.img_sz=img_sz
        self.grid_sz=grid_sz
        self.seg_sz=seg_sz
        self.imgs=[]
        self.label_pts=[]
        self.label_off=[]
        self.label_masks=[]
        self.label_boxes=[]
        self.getData(ds)
        self.ds=ds
        
    def getData(self,ds):
        for i,d in tqdm(enumerate(ds)):
            image=d.images.numpy()
            if i==100:
                cv2.imwrite("test2.png",image)
            if i==280:
                cv2.imwrite("test3.png",image)
            if i==240:
                cv2.imwrite("test4.png",image)
            image=cv2.resize(image,(self.img_sz,self.img_sz))
            masks=d.masks.numpy().astype(np.uint8)*255
            img_cs=set(())
            mod_masks=[]
            mod_boxes=[]
            mod_centers=[]
            mod_offsets=[]
            grid=np.zeros((self.grid_sz,self.grid_sz),dtype=np.uint8)
            for j in range(masks.shape[-1]):
                mask=masks[...,j]
                mask=cv2.resize(mask,(self.img_sz,self.img_sz),cv2.INTER_NEAREST)
                cY,cX=find_center(mask)
                if not mask[cY][cX]:
                    cY,cX=fixed_points(cX,cY,mask)
                cY,cX,offy,offx,img_cs=get_quantized_center(cX,cY,mask,dst_size=self.grid_sz,p_sofar=img_cs)
                mod_centers.append([cY,cX])
                mod_offsets.append([offy,offx])
                nzeros=np.nonzero(mask)
                ys=nzeros[0]
                xs=nzeros[1]
                ymin=min(ys)
                ymax=max(ys)
                xmin=min(xs)
                xmax=max(xs)
                croped_mask = mask[ymin : ymax , xmin: xmax]
                ## resize masks to eventual size of masks to be predicted
                croped_mask=cv2.resize(croped_mask,(self.seg_sz,self.seg_sz),cv2.INTER_NEAREST)
                mod_masks.append(croped_mask)            
                mod_boxes.append([xmin,ymin,xmax,ymax])
                # Considering only one calss can add dict with value corresponding to class for multiclass
            self.label_pts.append(mod_centers)
            self.label_off.append(mod_offsets)
            self.imgs.append(image)
            self.label_masks.append(mod_masks)
            self.label_boxes.append(mod_boxes)
        
    def __getitem__(self, index):
        
        tgt_img = self.imgs[index]
        pts = self.label_pts[index]
        offs= self.label_off[index]
        masks = self.label_masks[index]
        boxes = self.label_boxes[index]
        tgt_masks=np.zeros((self.grid_sz*self.grid_sz,self.seg_sz,self.seg_sz),dtype=np.uint8)
        tgt_label=np.zeros((self.grid_sz,self.grid_sz),dtype=np.uint8)
        tgt_boxes=np.zeros((self.grid_sz,self.grid_sz,4),dtype=np.float32)
        for i,pt in enumerate(pts):
            #first point is y second is x
            tgt_label[pt[0]][pt[1]]=1
            idx=pt[0]*self.grid_sz+pt[1]
            tgt_masks[idx]= masks[i]//255
            for j in range(4):
                tgt_boxes[pt[0]][pt[1]][j]= boxes[i][j]
        data={}
        data["img"]=preprocess((torch.from_numpy(tgt_img).float()))
        data["pts"]=torch.tensor(pts)
        data["offs"]=torch.tensor(offs).float()
        data["center"]=torch.from_numpy(tgt_label).float()
        data["bboxs"]=torch.from_numpy(tgt_boxes).float()
        data["msks"]=torch.from_numpy(tgt_masks).float()   
        return data
    
    def __len__(self):
        return len(self.ds)


class valLoader(data.Dataset):
    def __init__(self, img_sz=512,grid_sz=64,seg_sz=64,data=None):
        super(valLoader, self).__init__()
        self.img_sz=img_sz
        self.grid_sz=grid_sz
        self.seg_sz=seg_sz
        self.imgs=[]
        self.label_pts=[]
        self.label_off=[]
        self.label_masks=[]
        self.label_boxes=[]
        self.getData(data)
        self.ds=data
        
    def getData(self,data):
        for i,d in tqdm(enumerate(data)):
            image=d.images.numpy()
            image=cv2.resize(image,(self.img_sz,self.img_sz))
            masks=d.masks.numpy().astype(np.uint8)*255
            img_cs=set(())
            mod_masks=[]
            mod_boxes=[]
            mod_centers=[]
            mod_offsets=[]
            grid=np.zeros((self.grid_sz,self.grid_sz),dtype=np.uint8)
            for j in range(masks.shape[-1]):
                mask=masks[...,j]
                mask=cv2.resize(mask,(self.img_sz,self.img_sz),cv2.INTER_NEAREST)
                cY,cX=find_center(mask)
                if not mask[cY][cX]:
                    cY,cX=fixed_points(cX,cY,mask)
                cY,cX,offy,offx,img_cs=get_quantized_center(cX,cY,mask,dst_size=self.grid_sz,p_sofar=img_cs)
                mod_centers.append([cY,cX])
                mod_offsets.append([offy,offx])
                nzeros=np.nonzero(mask)
                ys=nzeros[0]
                xs=nzeros[1]
                ymin=min(ys)
                ymax=max(ys)
                xmin=min(xs)
                xmax=max(xs)
                croped_mask = mask[ymin : ymax , xmin: xmax]
                ## resize masks to eventual size of masks to be predicted
                croped_mask=cv2.resize(croped_mask,(self.seg_sz,self.seg_sz),cv2.INTER_NEAREST)
                mod_masks.append(croped_mask)            
                mod_boxes.append([xmin,ymin,xmax,ymax])
                # Considering only one calss can add dict with value corresponding to class for multiclass
            self.label_pts.append(mod_centers)
            self.label_off.append(mod_offsets)
            self.imgs.append(image)
            self.label_masks.append(mod_masks)
            self.label_boxes.append(mod_boxes)
        
    def __getitem__(self, index):
        
        tgt_img = self.imgs[index]
        masks = self.label_masks[index]
        boxes = self.label_boxes[index]
        rgb=preprocess((torch.from_numpy(tgt_img).float()))
        # There is only one class for now
        labels= torch.ones((len(masks),), dtype=torch.int64)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        ## convert to tensors
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target = {}
        target["boxes"] = boxes
        target["masks"] = masks//255
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["labels"] = labels
        return rgb, target
    
    def __len__(self):
        return len(self.ds)



if __name__=="__main__":
    src="hub://aismail2/cucumber_OD"
    ds = hub.load(src)
    data=DataLoader(ds=ds)
    ## Check random data 
    d=data[219]
    img=d["img"]
    pts=d["pts"]
    offs=d["offs"]
    boxes=d["bboxs"]
    lbl=d["center"]
    msks=d["msks"]
    print(img.shape,pts.shape,offs.shape,lbl.shape,boxes.shape,msks.shape)
    print("Min and Max values are")
    print(img.min(),img.max(), lbl.min(),lbl.max(),pts.max(),pts.min(),offs.min(),offs.max(),boxes.min(),boxes.max())
    vis_img=postprocess(img)
    plt.imshow(vis_img)
    plt.figure(2)
    plt.imshow(lbl.numpy().astype(np.uint8))
    plt.figure(3)
    # Visualize boxes
    for idx in range(len(pts)):
        box=boxes[pts[idx][0],pts[idx][1]]
        cv2.rectangle(vis_img, (box[0],box[1]), (box[2],box[3]), (255,0,0), 5)
    plt.imshow(vis_img)
    # Visualize masks
    for idx in range(len(pts)):
        msk=msks[pts[idx][0]*64+pts[idx][1]]
        msk=msk.numpy()
        plt.figure(idx+4)
        plt.imshow(msk)

    ## Check Data Loader collate function
    BATCH_SIZE=4
    data_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE,num_workers=4,collate_fn=collate_batch,shuffle=True)
    for batch in data_loader:
        print(batch.keys())
        print(batch["img"].shape)
        print(batch["center"].shape)
        print(batch["bboxs"].shape)
        print(batch["msks"].shape)
        gt_center=batch["pts"]
        print(gt_center)
        #print(batch[0].shape,len(batch[1]),len(batch[2]),batch[3].shape,batch[4].shape,batch[5].shape)
        break