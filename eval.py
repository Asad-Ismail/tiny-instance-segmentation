import torch.utils.data as data
import torch
from coco_utils import get_coco_api_from_dataset, coco_to_excel
from coco_eval import CocoEvaluator
import utils
import time
from datasets.data_loader import valLoader 


src="hub://aismail2/cucumber_OD"
ds = hub.load(src)
print(f"The size of Test Loader is {len(ds)}")
val_dataset = valLoader(data=ds)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=1, batch_size=1, shuffle=False)
coco = get_coco_api_from_dataset(val_loader.dataset)

iou_types = ["bbox","segm"]

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