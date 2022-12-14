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

# Parse Args
parser = argparse.ArgumentParser()
resnet18_inst.pth
parser.add_argument("--image",default="test.png",help="Input Image")
parser.add_argument("--size",default=512,type=int,help="Image size used for training model")
parser.add_argument("--vispath", default="vis_results",help="Write visualizations to this location")
parser.add_argument("--weight_path",default="./weights/resnet18seg.pth",type=str,help="Image size used for training model")
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
    img=cv2.imread("test.png")
    img=cv2.resize(img,(img_sz,img_sz))
    batch={}
    batch["img"]=preprocess((torch.from_numpy(img).float())).unsqueeze(0)
    preds=model(batch)
    imgs=batch["img"]
    boxes=preds["boxes"]
    pred_cats=preds["centers"]
    pred_msks=preds["masks"]
    bidx=0
    ## Draw Preds
    img=vis_boxes(imgs[bidx],boxes,postprocess=postprocess)
    img=vis_masks(img.copy(),pred_msks,boxes)
    cv2.imwrite(f"{vispath}/seg.png",img)
    cv2.imwrite(f"{vispath}/cent.png",pred_cats.detach().squeeze(0).numpy()*20)
    