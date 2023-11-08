# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Simple Continual Learning datasets
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from timm.data import create_transform

from longtailed_datasets.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100

import utils
from sampler.class_aware_sampler import ClassAwareSampler


def build_longtailed_dataloader(args):
    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    dataset_train, dataset_val = get_dataset(args.dataset, transform_train, transform_val, args)

    args.nb_classes = len(dataset_val.classes)

    data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            shuffle=True,
            drop_last=True,
        )
    if args.balanced_sampler:
        balanced_sampler = ClassAwareSampler(dataset_train, num_samples_cls=4)
        data_loader_train_bal = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
                sampler=balanced_sampler
            )
        print('use class balanced sampler')

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=128,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=False
    )
    return data_loader_train, data_loader_val, None

def get_dataset(dataset, transform_train, transform_val, args,):
    if dataset == 'CIFAR100':
        dataset_train = IMBALANCECIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_val)
    elif dataset == 'CIFAR10':
        dataset_train = IMBALANCECIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_val)
    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # scale = (0.05, 1.0)
        # ratio = (3. / 4., 4. / 3.)
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(args.input_size),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5,0.5,0.5], [0.5, 0.5, 0.5])
        # ])
        # return transform
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5, 0.5, 0.5])
        ])
        return transform
    # t = []
    # if resize_im:
    #     size = int((256 / 224) * args.input_size)
    #     t.append(
    #         # transforms.Resize(size),  # to maintain same ratio w.r.t. 224 images
    #         transforms.Resize(size),
    #     )
    #     t.append(transforms.CenterCrop(args.input_size))
    # t.append(transforms.ToTensor())
    # t.append(transforms.Normalize([0.5,0.5,0.5], [0.5, 0.5, 0.5]))
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5, 0.5, 0.5])
        ])
    
    return transform