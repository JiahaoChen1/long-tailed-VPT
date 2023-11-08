import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import utils
from engine import *
import models
from datasets import build_longtailed_dataloader

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    data_loader_train, data_loader_val, data_loader_train_bal = build_longtailed_dataloader(args)
    cls_num_list = data_loader_train.dataset.get_cls_num_list()
    head_indices, medium_indices, few_indices = data_loader_train.dataset.head, data_loader_train.dataset.medium, data_loader_train.dataset.few

    original_model = create_model(
        'base_'+args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        checkpoint_path='/data00/jiahao/l2p-pytorch-main/longtailed-l2p/pretrained_model/ViT-B_16.npz',
    )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        checkpoint_path=args.ckpt,
        cls_num_list = cls_num_list,
        task=args.task,
        head_indices=head_indices,
        medium_indices=medium_indices,
        few_indices=few_indices,
        layer_prompt_length=args.layer_prompt_length,
        vpt_depth = args.layer_prompt_depth
    
    )
    model.to(device)  
    original_model.to(device)
    print(args)
    # original_model = None
    
    if args.eval:
        evaluate_cls(model, original_model, data_loader_val, device, args)
        # visualize(model, original_model, data_loader_val, device, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('L2P training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_l2p':
        from configs.cifar100_l2p import get_args_parser
        config_parser = subparser.add_parser('cifar100_l2p', help='Split-CIFAR100 L2P configs')
    elif config == 'five_datasets_l2p':
        from configs.five_datasets_l2p import get_args_parser
        config_parser = subparser.add_parser('five_datasets_l2p', help='5-Datasets L2P configs')
    else:
        raise NotImplementedError
    
    get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

    sys.exit(0)