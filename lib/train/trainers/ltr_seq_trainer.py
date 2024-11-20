import os
import datetime
from collections import OrderedDict
from torch.nn.utils import clip_grad_norm_
# from lib.train.data.wandb_logger import WandbWriter
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from memory_profiler import profile
# from lib.train.admin import TensorboardWriter
import torch
import time
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from lib.utils.misc import get_world_size


class LTRSeqTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard and wandb
        # self.wandb_writer = None
        # if settings.local_rank in [-1, 0]:
        #    tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
        #    if not os.path.exists(tensorboard_writer_dir):
        #        os.makedirs(tensorboard_writer_dir)
        #    self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        #    if settings.use_wandb:
        #        world_size = get_world_size()
        #        cur_train_samples = self.loaders[0].dataset.samples_per_epoch * max(0, self.epoch - 1)
        #        interval = (world_size * settings.batchsize)  # * interval
        #        self.wandb_writer = WandbWriter(settings.project_path[6:], {}, tensorboard_writer_dir, cur_train_samples, interval)

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        print("move_data", self.move_data_to_gpu)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

        self.miou_list = []

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""
        torch.autograd.set_detect_anomaly(True)
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        for i, data in enumerate(loader, 1):
            self.actor.eval()
            self.data_read_done_time = time.time()
            with torch.no_grad():
                explore_result = self.actor.explore(data)
            if explore_result == None:
                print("this time i skip")
                # self._update_stats(stats, batch_size, loader)
                continue
            # get inputs
            # print(data)

            self.data_to_gpu_time = time.time()

            data['epoch'] = self.epoch
            data['settings'] = self.settings

            stats = {}
            reward_record = []
            miou_record = []
            e_miou_record = []
            num_seq = len(data['num_frames'])

            # Calculate reward tensor
            # reward_tensor = torch.zeros(explore_result['baseline_iou'].size())
            baseline_iou = explore_result['baseline_iou']
            # explore_iou = explore_result['explore_iou']
            for seq_idx in range(num_seq):
                num_frames = data['num_frames'][seq_idx] - 1
                b_miou = torch.mean(baseline_iou[:num_frames, seq_idx])
                #    e_miou = torch.mean(explore_iou[:num_frames, seq_idx])
                miou_record.append(b_miou.item())
                #    e_miou_record.append(e_miou.item())

                b_reward = b_miou.item()
            #    e_reward = e_miou.item()
            #    iou_gap = e_reward - b_reward
            #    reward_record.append(iou_gap)
            #    reward_tensor[:num_frames, seq_idx] = iou_gap

            # Training mode
            cursor = 0
            bs_backward = 1

            # print(self.actor.net.module.box_head.decoder.layers[2].mlpx.fc1.weight)
            self.optimizer.zero_grad()
            while cursor < num_seq:
                # print("now is ", cursor , "and all is ", num_seq)
                model_inputs = {}
                model_inputs['slt_loss_weight'] = 15
                if cursor < num_seq:
                    model_inputs['template_images'] = explore_result['template_images'][
                                                      cursor:cursor + bs_backward].cuda()
                else:
                    model_inputs['template_images'] = explore_result['template_images_reverse'][
                                                      cursor - num_seq:cursor - num_seq + bs_backward].cuda()
                model_inputs['search_images'] = explore_result['search_images'][:, cursor:cursor + bs_backward].cuda()
                model_inputs['search_anno'] = explore_result['search_anno'][:, cursor:cursor + bs_backward].cuda()
                model_inputs['pre_seq'] = explore_result['pre_seq'][:, cursor:cursor + bs_backward].cuda()
                model_inputs['x_feat'] = explore_result['x_feat'].squeeze(1)[:, cursor:cursor + bs_backward].cuda()
                model_inputs['epoch'] = data['epoch']
                # model_inputs['template_update'] = explore_result['template_update'].squeeze(1)[:,
                #                                  cursor:cursor + bs_backward].cuda()
                # print("this is cursor")
                # print(explore_result['pre_seq'].shape)
                # print(explore_result['x_feat'].squeeze(1).shape)
                # model_inputs['action_tensor'] = explore_result['action_tensor'][:, cursor:cursor + bs_backward].cuda()
                # model_inputs['reward_tensor'] = reward_tensor[:, cursor:cursor + bs_backward].cuda()

                loss, stats_cur = self.actor.compute_sequence_losses(model_inputs)
                # for name, param in self.actor.net.named_parameters():
                #    shape, c = (param.grad.shape, param.grad.sum()) if param.grad is not None else (None, None)
                #    print(f'{name}: {param.shape} \n\t grad: {shape} \n\t {c}')
                # print("i make this!")
                loss.backward()
                # print("i made that?")

                for key, val in stats_cur.items():
                    if key in stats:
                        stats[key] += val * (bs_backward / num_seq)
                    else:
                        stats[key] = val * (bs_backward / num_seq)
                cursor += bs_backward
            grad_norm = clip_grad_norm_(self.actor.net.parameters(), 100)
            stats['grad_norm'] = grad_norm
            # print(self.actor.net.module.backbone.blocks[8].mlp.fc1.weight)
            self.optimizer.step()
            # print(self.optimizer)

            miou = np.mean(miou_record)
            self.miou_list.append(miou)
            # stats['reward'] = np.mean(reward_record)
            # stats['e_mIoU'] = np.mean(e_miou_record)
            stats['mIoU'] = miou
            stats['mIoU10'] = np.mean(self.miou_list[-10:])
            stats['mIoU100'] = np.mean(self.miou_list[-100:])

            batch_size = num_seq * np.max(data['num_frames'])
            self._update_stats(stats, batch_size, loader)
            self._print_stats(i, loader, batch_size)
            torch.cuda.empty_cache()

            # # forward pass
            # if not self.use_amp:
            #     loss, stats = self.actor(data)
            # else:
            #     with autocast():
            #         loss, stats = self.actor(data)
            #
            # # backward pass and update weights
            # if loader.training:
            #     self.optimizer.zero_grad()
            #     if not self.use_amp:
            #         loss.backward()
            #         if self.settings.grad_clip_norm > 0:
            #             torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
            #         self.optimizer.step()
            #     else:
            #         self.scaler.scale(loss).backward()
            #         self.scaler.step(self.optimizer)
            #         self.scaler.update()

            # update statistics
            # batch_size = data['template_images'].shape[loader.stack_dim]
            # self._update_stats(stats, batch_size, loader)

            # print statistics
            # self._print_stats(i, loader, batch_size)

            # update wandb status
            # if self.wandb_writer is not None and i % self.settings.print_interval == 0:
            #    if self.settings.local_rank in [-1, 0]:
            #        self.wandb_writer.write_log(self.stats, self.epoch)

        # calculate ETA after every epoch
        # epoch_time = self.prev_time - self.start_time
        # print("Epoch Time: " + str(datetime.timedelta(seconds=epoch_time)))
        # print("Avg Data Time: %.5f" % (self.avg_date_time / self.num_frames * batch_size))
        # print("Avg GPU Trans Time: %.5f" % (self.avg_gpu_trans_time / self.num_frames * batch_size))
        # print("Avg Forward Time: %.5f" % (self.avg_forward_time / self.num_frames * batch_size))

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        # if self.settings.local_rank in [-1, 0]:
        #    self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        # add lr state
        if loader.training:
            lr_list = self.lr_scheduler.get_last_lr()
            for i, lr in enumerate(lr_list):
                var_name = 'LearningRate/group{}'.format(i)
                if var_name not in self.stats[loader.name].keys():
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        prev_frame_time_backup = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += (self.data_read_done_time - prev_frame_time_backup)
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)

            # 2021.12.14 add data time print
            print_str += 'DataTime: %.3f (%.3f)  ,  ' % (
            self.avg_date_time / self.num_frames * batch_size, self.avg_gpu_trans_time / self.num_frames * batch_size)
            print_str += 'ForwardTime: %.3f  ,  ' % (self.avg_forward_time / self.num_frames * batch_size)
            print_str += 'TotalTime: %.3f  ,  ' % ((current_time - self.start_time) / self.num_frames * batch_size)
            # print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.data_read_done_time - prev_frame_time_backup, self.data_to_gpu_time - self.data_read_done_time)
            # print_str += 'ForwardTime: %.3f  ,  ' % (current_time - self.data_to_gpu_time)
            # print_str += 'TotalTime: %.3f  ,  ' % (current_time - prev_frame_time_backup)

            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    # def _write_tensorboard(self):
    #    if self.epoch == 1:
    #        self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

    #    self.tensorboard_writer.write_epoch(self.stats, self.epoch)
