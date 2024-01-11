# encoding: utf-8

import logging
from torchvision.transforms import transforms as T
import torch
import os
from PIL import Image
from math import exp
import torch.nn.functional as F
import torchvision
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from skimage.metrics import structural_similarity as compare_ssim

from utils.reid_metric import r1_mAP_mINP
from tools.test import create_supervised_evaluator

global ITER
ITER = 0

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def create_supervised_trainer(noise, model, optimizer, criterion, cetner_loss_weight=0.0, device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (dict - class:`torch.optim.Optimizer`): the optimizer to use
        criterion (dict - class:loss function): the loss function to use
        cetner_loss_weight (float, optional): the weight for cetner_loss_weight
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """

    def _update(engine, batch):
        noise.requires_grad = True
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mean = torch.Tensor(normalize.mean).view(1, 3, 1, 1).cuda()
        std = torch.Tensor(normalize.std).view(1, 3, 1, 1).cuda()
        # model.train()
        optimizer.zero_grad()
        optimizer.rescale()

        # if 'center' in optimizer.keys():
        #     optimizer['center'].zero_grad()

        img, target = batch

        # img = torch.clamp(ori_img + noise, 0, 1)
        # img = (img - mean) / std
        img1 = img.to(device) if torch.cuda.device_count() >= 1 else img
        img = torch.clamp(img1 + noise, 0, 1)
        # loss_ssim = ssim(img, img1)
        # load_torch = torch.load('noise.pt')

        # ssim = compare_ssim(img.cpu().numpy(), img1.cpu().numpy(), win_size=11, data_range=1.0, multichannel=True)
        img = (img - mean) / std
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        feat = model(img)
        optimizer.zero_grad()
        loss = criterion['total'](feat, feat, target)
        loss.backward()
        optimizer.step()

        if engine.state.epoch % 10 == 0:
            filename = os.path.join('../tools/log/market1501/Experiment-AGW-baseline', f"noise_{engine.state.epoch}")
            filename_pt = os.path.join('../tools/log/market1501/Experiment-AGW-baseline', f"noise_{engine.state.epoch}.pt")
            torch.save(noise, filename_pt)
            torchvision.utils.save_image(noise, filename + ".png", normalize=True)

        # compute acc
        acc = torch.tensor(0).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def do_train(
        cfg, noise,
        model,
        data_loader,
        optimizer,
        scheduler,
        criterion,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline")
    logger.info("Start training")

    trainer = create_supervised_trainer(noise, model, optimizer, criterion, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)

    evaluator = create_supervised_evaluator(noise, model, metrics={'r1_mAP_mINP': r1_mAP_mINP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer
                                                                     })
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(data_loader['train']),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(data_loader['train']) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            data_loader['train'].batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(data_loader['eval'])
            cmc, mAP, mINP = evaluator.state.metrics['r1_mAP_mINP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mINP: {:.1%}".format(mINP))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(data_loader['train'], max_epochs=epochs)
