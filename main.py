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
from timm.models.helpers import load_checkpoint

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

os.environ[ "CUDA_VISIBLE_DEVICES" ] = "1"
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

    print(f"Creating original model: {args.model}")
    original_model = create_model(
        'base_'+args.model,
        pretrained=True,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        # checkpoint_path='/data00/jiahao/l2p-pytorch-main/longtailed-l2p/pretrained_model/ViT-B_16.npz',
    )

    print(f"Creating model: {args.model}")
    print(args.prompt_pool)
    model = create_model(
        args.model,
        pretrained=True,
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
        # checkpoint_path='/data00/jiahao/l2p-pytorch-main/longtailed-l2p/pretrained_model/ViT-B_16.npz',
        cls_num_list = cls_num_list,
        task=args.task,
        head_indices=head_indices,
        medium_indices=medium_indices,
        few_indices=few_indices,
        layer_prompt_length=args.layer_prompt_length,
        vpt_depth = args.layer_prompt_depth
    )

    # load_checkpoint(model, checkpoint_path='/data00/jiahao/l2p-pytorch-main/VLM-l2p/vpt_ckpt_88/model_path/checkpoint.pth')
    original_model.to(device)
    model.to(device)  

    print(args.freeze)
    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False
        
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    print(args)

    if args.eval:
        # acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        evaluate(model, original_model, data_loader_val, device, args)
        evaluate_cls(model, original_model, data_loader_val, device, args)
        return

    model_without_ddp = model
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0

    optimizer = create_optimizer(args, model_without_ddp)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    stage1_criterion = torch.nn.CrossEntropyLoss().to(device)
    # stage1_criterion = LabelSmoothingCrossEntropy()
    # stage1_criterion = AGCL(cls_num_list=cls_num_list, m=0.1, s=20, weight=None, train_cls=False, noise_mul=0.5, gamma=4., device=device, gamma_pos=0.5, gamma_neg=8.0)
    criterion = DiverseExpertLoss(cls_num_list, device=device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    original_model = None
    train_and_evaluate(model, model_without_ddp, original_model,
                    criterion, data_loader_train, data_loader_val, optimizer, lr_scheduler,
                    device, cls_num_list, stage1_criterion, args, data_loader_train_bal)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")


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