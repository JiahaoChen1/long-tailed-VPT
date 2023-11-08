# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer

import utils
import matplotlib.pyplot as plt
from loss import AGCL, DiverseExpertLoss, LabelSmoothingCrossEntropy
import seaborn as sns
import torch.nn.functional as F
from transmix import Mixup_transmix, mixup_data, mixup_criterion


# mon = Mixup_transmix()


def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    set_training_mode=True, stage1_criterion = None, args = None,
                    dataloader_bal=None):

    model.train(set_training_mode)
    # model.eval()
    if original_model is not None:
        original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for input, target, c_index in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # c_index = c_index.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        
        # mixed_x, y_a, y_b, lam = mixup_data(input, target)
        # output = model(mixed_x, cls_features=cls_features, target=target, c_index=c_index)
        output = model(input, cls_features=cls_features)
        # logits = output['logits']
        # loss = stage1_criterion(logits, target)
        # loss = F.cross_entropy(logits, target) # base criterion (CrossEntropyLoss)
        loss_expert = criterion(target, output)
        loss = loss_expert #+ output['simi_loss']
        # print(loss)
        logits = output['head_logits'] + output['medium_logits'] + output['few_logits']
        # logits = output['out']
        # input_logits = F.log_softmax(logits, dim=1)
        # target_logits = F.softmax(tem_logits, dim=1)
        # loss = loss + F.kl_div(input_logits, target_logits) * 0.1

        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']
            # loss = loss + args.pull_constraint_coeff * output['reduce_lang']
            # print(output['reduce_lang'])
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()
    if original_model is not None:
        original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, cls_features=cls_features)
            # logits = output['out']
            logits = output['head_logits'] + output['medium_logits']+ output['few_logits']
            loss = criterion(logits, target)
            
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def visualize(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, args=None,):
    criterion = torch.nn.CrossEntropyLoss()
    # original_model = None
    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()
    if original_model is not None:
        original_model.eval()

    attn_all = []

    # head_sample = []
    # few_sample = []
    targets = []
    rights = []
    logits = []
    # layer_xs = []
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, cls_features=cls_features)
            attn_ = output['attn_list']
            attn_all.append(attn_)    # layer
            
            logit = output['logits']

            pred = torch.argmax(logit, dim=1)

            targets.append(target)
            logits.append(logit)
            rights.append(pred == target)
            # layer_xs.append(output['layer_list'])

    attn_all = torch.cat(attn_all, dim=0)
    targets = torch.cat(targets, dim=0)
    rights = torch.cat(rights, dim=0)
    logits = torch.cat(logits, dim=0)
    # layer_xs = torch.cat(layer_xs, dim=0)
    # head_sample = torch.cat(head_sample, dim=0)
    # few_sample = torch.cat(few_sample, dim=0)

    vis1 = []
    vis2 = []
    for i in range(100):
        head_attn_all = attn_all[targets == i]
        # few_attn_all = attn_all[targets == 99]
        head_attn_mean = head_attn_all.mean(dim=0)
        # head_attn_mean = torch.max(head_attn_all, dim=)
        item1 = torch.mean(head_attn_mean[-1, 1:6], dim=0)
        item2 = torch.mean(head_attn_mean[-1, 6:], dim=0)
        vis1.append(float(item1))
        vis2.append(float(item2))
    print(vis1)
    print(vis2)

    # head_attn_all = attn_all[targets == 0]
    # few_attn_all = attn_all[targets == 99]

    
    # for i  in  range(12):
    #     sns.heatmap(head_attn_all[:, i, :].cpu().numpy(), vmin=0, vmax=1)
    #     plt.savefig(f'head{i}.png')
    #     plt.clf()
    
    # for i  in  range(12):
    #     sns.heatmap(few_attn_all[:, i, :].cpu().numpy(), vmin=0, vmax=1)
    #     plt.savefig(f'few{i}.png')
    #     plt.clf()

    # for i in range(100):
    #     head_layer_xs = layer_xs[targets == i]
    #     head_layer_xs = F.normalize(head_layer_xs, dim=-1)
    #     simi = (head_layer_xs[:, :, None, :] * head_layer_xs[:, None, :, :]).sum(dim=-1)
    #     simi = torch.mean(simi, dim=0)
    #     print(simi[0])

    # head_attn_mean = head_attn_all.mean(dim=0)
    # few_attn_mean = few_attn_all.mean(dim=0)
    # 
    
    # # print(torch.mean(head_attn_mean[:, 6:], dim=1))
    # print('&&&&&&&&&&&')
    # print(torch.mean(few_attn_mean[:, 1:6], dim=1))
    # # print(torch.mean(few_attn_mean[:, 6:], dim=1))


@torch.no_grad()
def evaluate_cls(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, args=None,):
    
    pre_sam = torch.tensor([0 for _ in range(100)])
    num_sam = torch.tensor([0 for _ in range(100)])
    # confidence_score = torch.tensor([0. for _ in range(100)])
    attn_all = []

    model.eval()
    if original_model is not None:
        original_model.eval()

    with torch.no_grad():
        for input, target in data_loader:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
                # tem_logits = output['logits']
            else:
                cls_features = None
            
            output = model(input, cls_features=cls_features)
            # logits = output['logits']
            logits = output['head_logits'] + output['medium_logits'] + output['few_logits']
            # confidence = torch.softmax(logits, dim=1)
            # confidence, _ = torch.max(confidence.cpu(), dim=1)

            _, pred = logits.topk(1, 1, True, True)

            for i in range(logits.shape[0]):
                num_sam[target[i]] += 1
                if pred[i][0] == target[i]:
                    pre_sam[target[i]] += 1
                
                # confidence_score[target[i]] += confidence[i]
    
    print(f'head {pre_sam[:36].sum() / num_sam[:36].sum()}')
    print(f'medium {pre_sam[36:71].sum() / num_sam[36:71].sum()}')
    print(f'few {pre_sam[71:].sum() / num_sam[71:].sum()}')
    print(pre_sam / num_sam)
    print(f'All {pre_sam.sum() / num_sam.sum()}')
    # print(confidence_score)
    

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader_train: Iterable, data_loader_val: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    cls_num_list, stage1_criterion = None, args = None, dataloader_bal = None):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for epoch in range(args.epochs):  
        
        if lr_scheduler:
            lr_scheduler.step(epoch+1)        

        train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                    data_loader=data_loader_train, optimizer=optimizer, 
                                    device=device, epoch=epoch, 
                                    set_training_mode=True, stage1_criterion = stage1_criterion, args=args,
                                    dataloader_bal=dataloader_bal)
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader_val, 
                            device=device, args=args)
    
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'model_path')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(args.output_dir, 'log_path')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'model_path/checkpoint.pth')
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)
    
    test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader_val, 
                            device=device, args=args)
