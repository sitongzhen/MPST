# encoding: utf-8

import argparse
import os
import sys
import torch, math
from PIL import Image
from torchvision.transforms import transforms as T
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from modeling import build_model
from torch.optim.optimizer import Optimizer, required
from utils.lr_scheduler import WarmupMultiStepLR
from utils.logger import setup_logger
from tools.train import do_train
from tools.test import do_test

CHECK = 1e-5
SAT_MIN = 0.5
MODE = "bilinear"
def rescale_check(check, sat, sat_change, sat_min):
    return sat_change < check and sat > sat_min

class MI_SGD(Optimizer):
    def __init__(
            self,
            params,
            lr=required,
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False,
            max_eps=10 / 255,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            sign=False,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MI_SGD, self).__init__(params, defaults)
        self.sat = 0
        self.sat_prev = 0
        self.max_eps = max_eps

    def __setstate__(self, state):
        super(MI_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def rescale(self, ):
        for group in self.param_groups:
            if not group["sign"]:
                continue
            for p in group["params"]:
                self.sat_prev = self.sat
                self.sat = (p.data.abs() >= self.max_eps).sum().item() / p.data.numel()
                sat_change = abs(self.sat - self.sat_prev)
                if rescale_check(CHECK, self.sat, sat_change, SAT_MIN):
                    print('rescaled')
                    p.data = p.data / 2

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group["sign"]:
                    d_p = d_p / (d_p.norm(1) + 1e-12)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if group["sign"]:
                    p.data.add_(-group["lr"], d_p.sign())
                    p.data = torch.clamp(p.data, -self.max_eps, self.max_eps)
                else:
                    p.data.add_(-group["lr"], d_p)

        return loss

def main():
    parser = argparse.ArgumentParser(description="AGW Re-ID Baseline")
    parser.add_argument(
        "--config_file", default="../configs/AGW_baseline.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True

    data_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(cfg, num_classes)

    if 'cpu' not in cfg.MODEL.DEVICE:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device=cfg.MODEL.DEVICE)
    model.load_param(cfg.TEST.WEIGHT)
    if cfg.TEST.EVALUATE_ONLY == 'on':

        # noise = Image.open('./log/market1501/Experiment-AGW-baseline/market_noise_120_14.4-4.7.png')
        # noise = noise.convert('RGB')
        # noise_tensor = T.ToTensor()
        # noise = noise_tensor(noise).cuda()

        noise = torch.load('./log/market1501/Experiment-AGW-baseline/noise_120.pt').cuda()
        logger.info("Evaluate Only")
        model.load_param(cfg.TEST.WEIGHT)
        do_test(cfg, noise, model, data_loader, num_query)
        return

    criterion = model.get_creterion(cfg, num_classes)
    optimizer = model.get_optimizer(cfg, criterion)

    # Add for using self trained model
    if cfg.MODEL.PRETRAIN_CHOICE == 'self':
        start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
        print('Start epoch:', start_epoch)
        path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
        print('Path to the checkpoint of optimizer:', path_to_optimizer)
        path_to_center_param = cfg.MODEL.PRETRAIN_PATH.replace('model', 'center_param')
        print('Path to the checkpoint of center_param:', path_to_center_param)
        path_to_optimizer_center = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center')
        print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center)
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
        optimizer['model'].load_state_dict(torch.load(path_to_optimizer))
        criterion['center'].load_state_dict(torch.load(path_to_center_param))
        optimizer['center'].load_state_dict(torch.load(path_to_optimizer_center))
        scheduler = WarmupMultiStepLR(optimizer['model'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
    elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        start_epoch = 0
        scheduler = WarmupMultiStepLR(optimizer['model'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    else:
        print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))
    noise = torch.zeros((3, cfg.INPUT.IMG_SIZE[0], cfg.INPUT.IMG_SIZE[1])).cuda()
    MAX_EPS = 0.03 #8 / 255.0
    noise.requires_grad = True
    optimizer = MI_SGD(
        [
            {"params": [noise], "lr": MAX_EPS, "momentum": cfg.SOLVER.MOMENTUM, "sign": True}
        ],
        #max_eps=MAX_EPS,
    )
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(-0.01))
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    model.eval()
    do_train(cfg, noise,
        model,
        data_loader,
        optimizer,
        scheduler,
        criterion,
        num_query,
        start_epoch
    )

if __name__ == '__main__':
    main()
