import time
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import accuracy, AverageMeter
from train.config import get_config
from train.lr_scheduler import build_scheduler
from train.logger import create_logger
from utils.helpers import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, save_latest, update_model_ema, unwrap_model
import copy
from train.optimizer import build_optimizer
from models.repvggplus import create_RepVGGplus_by_name
import os
import deeplake as hub
from datasets.data_loader import DataLoader as dLoader
from datasets.data_loader import collate_batch
from torch.utils.data import DataLoader
from models.repvgg_tinyism import tinyModel


def parse_option():
    parser = argparse.ArgumentParser('RepOpt-VGG training script built on the codebase of Swin Transformer', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--arch', default="RepVGG-tinyism", type=str, help='arch name')
    parser.add_argument('--batch-size', default=16, type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', default='hub://aismail2/cucumber_OD', type=str, help='path to dataset')
    parser.add_argument('--scales-path', default=None, type=str, help='path to the trained Hyper-Search model')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],  #TODO Note: use amp if you have it
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='./output/repvggplus', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def main(config):

    logger.info(f"Creating model:{config.MODEL.ARCH}")

    model = tinyModel(posEncoding=True)
    ds = hub.load(config.DATA.DATA_PATH)
    data=dLoader(ds=ds)
    data_loader_train = DataLoader(dataset=data, batch_size=config.DATA.BATCH_SIZE,num_workers=1,collate_fn=collate_batch,shuffle=True)
    optimizer = build_optimizer(config, model)
    logger.info(str(model))
    model.cuda()

    if torch.cuda.device_count() > 1:
        if config.AMP_OPT_LEVEL != "O0":
            model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK],
                                                          broadcast_buffers=False)
        model_without_ddp = model.module
    else:
        if config.AMP_OPT_LEVEL != "O0":
            model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
        model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.TRAIN.EMA_ALPHA > 0 and (not config.EVAL_MODE) and (not config.THROUGHPUT_MODE):
        model_ema = copy.deepcopy(model)
    else:
        model_ema = None

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler, model_ema=model_ema)
        if epoch % config.SAVE_FREQ == 0:
            save_checkpoint(config, epoch, model_without_ddp, optimizer, lr_scheduler, logger, model_ema=model_ema)

        if epoch % config.SAVE_FREQ == 0 or epoch >= (config.TRAIN.EPOCHS - 10):
            if model_ema is not None:
                    latest_ema_path = os.path.join(config.OUTPUT, 'latest_ema.pth')
                    logger.info(f"{latest_ema_path} latest EMA saving......")
                    torch.save(unwrap_model(model_ema).state_dict(), latest_ema_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler,device="cuda",model_ema=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    
    for idx, batch in enumerate(data_loader):

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

        inp={"img":imgs}
        labels={"pts":gt_centers,"offs":gt_offsets,"bboxs":gt_boxes,"center":gt_cats,"msks":gt_msks}
        inp["labels"]=labels

        outputs = model(inp)

        losses=outputs["loss"]
        loss=losses["Total_Loss"]

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)

        else:
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item())
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)

        if model_ema is not None:
            update_model_ema(config, dist.get_world_size(), model=model, model_ema=model_ema, cur_epoch=epoch, cur_iter=idx)

        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


import os

if __name__ == '__main__':
    args, config = parse_option()
    seed = config.SEED 

    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if not config.EVAL_MODE:
        # linear scale the learning rate according to total batch size, may not be optimal
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 256.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE  / 256.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE  / 256.0
        # gradient accumulation also need to scale the learning rate
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()

    print('==========================================')
    print('real base lr: ', config.TRAIN.BASE_LR)
    print('==========================================')

    os.makedirs(config.OUTPUT, exist_ok=True)

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0 if torch.cuda.device_count() == 1 else dist.get_rank(), name=f"{config.MODEL.ARCH}")

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
