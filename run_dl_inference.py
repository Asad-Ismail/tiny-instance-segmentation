import torch.utils.data as data
import torch
import matplotlib.pyplot as plt
from models.tinyism import tinyModel
import torch.nn.functional as F
import numpy as np
import cv2
from datasets.data_loader import preprocess, postprocess
from utils.helpers import vis_results,vis_masks,vis_boxes
import argparse
from torch.utils.data import DataLoader
import deeplake as hub

# Parse Args
parser = argparse.ArgumentParser()
parser.add_argument("--image",default="test.png",help="Input Image")
parser.add_argument("--size",default=512,type=int,help="Image size used for training model")
parser.add_argument("--vispath", default="vis_results",help="Write visualizations to this location")
parser.add_argument("--weight_path",default="./weights/resnet18_inst.pth",type=str,help="Image size used for training model")
args = parser.parse_args()

img_path = args.image
img_sz = args.size
vispath = args.vispath
weightpath= args.weight_path


device=torch.device('cpu')
model=tinyModel()
model.load_state_dict(torch.load(weightpath,map_location=torch.device('cpu')))
model.cpu()
model.eval()

if __name__=="__main__":
    src="hub://aismail2/cucumber_OD"
    ds = hub.load(src)
    print(f"The size of Test Loader is {len(ds)}")
    test_loader = DataLoader(dataset=data,  batch_size=1,num_workers=4, shuffle=False)
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
        cv2.imwrite(f"{vispath}/{i}_seg.png",img)
        cv2.imwrite(f"{vispath}/{i}_cent.png",pred_cats.detach().squeeze(0).numpy()*20)
    