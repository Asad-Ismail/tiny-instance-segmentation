import torch.utils.data as data
import torch
## For single stage not FPN
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
        rgb=transform((torch.from_numpy(tgt_img).float()/255))
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
        return len(ds)


val_dataset = valLoader(data=ds)
val_loader = torch.utils.data.DataLoader(
                    dataset=val_dataset, num_workers=1, 
                    batch_size=1, shuffle=True)

from coco_utils import get_coco_api_from_dataset, coco_to_excel
from coco_eval import CocoEvaluator
import utils
import time

coco = get_coco_api_from_dataset(val_loader.dataset)
iou_types = ["bbox"]
iou_types.append("segm")

@torch.no_grad() 
def evaluate(coco, model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    model.to(device)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 5, header):
        #print(image.shape)
        image = [image.to(device)]
        #print(image[0].shape)
        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: v.to(device) for k, v in targets.items()}]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model({"img":image[0]})

        #outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        outputs = [{k: v.to(cpu_device) for k, v in outputs.items()}]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


coco_evaluator = evaluate(coco, model, val_loader, device)