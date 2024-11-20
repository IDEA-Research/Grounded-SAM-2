import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.stark import build_starks, build_starkst
from lib.models.stark import build_stark_lightning_x_trt
# forward propagation related
from lib.train.actors import STARKLightningXtrtdistillActor
# for import modules
import importlib


def build_network(script_name, cfg):
    # Create network
    if script_name == "stark_s":
        net = build_starks(cfg)
    elif script_name == "stark_st1" or script_name == "stark_st2":
        net = build_starkst(cfg)
    elif script_name == "stark_lightning_X_trt":
        net = build_stark_lightning_x_trt(cfg, phase="train")
    else:
        raise ValueError("illegal script name")
    return net


def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update the default teacher configs with teacher config file
    if not os.path.exists(settings.cfg_file_teacher):
        raise ValueError("%s doesn't exist." % settings.cfg_file_teacher)
    config_module_teacher = importlib.import_module("lib.config.%s.config" % settings.script_teacher)
    cfg_teacher = config_module_teacher.cfg
    config_module_teacher.update_config_from_file(settings.cfg_file_teacher)
    if settings.local_rank in [-1, 0]:
        print("New teacher configuration is shown below.")
        for key in cfg_teacher.keys():
            print("%s configuration:" % key, cfg_teacher[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir
    """turn on the distillation mode"""
    cfg.TRAIN.DISTILL = True
    cfg_teacher.TRAIN.DISTILL = True
    net = build_network(settings.script_name, cfg)
    net_teacher = build_network(settings.script_teacher, cfg_teacher)

    # wrap networks to distributed one
    net.cuda()
    net_teacher.cuda()
    net_teacher.eval()

    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        net_teacher = DDP(net_teacher, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    # settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    # settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "L1")
    # Loss functions and Actors
    if settings.script_name == "stark_lightning_X_trt":
        objective = {'giou': giou_loss, 'l1': l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = STARKLightningXtrtdistillActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings,
                                               net_teacher=net_teacher)
    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True, distill=True)
