# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import checkpoint, get_dist_info, init_dist, load_checkpoint

from mmedit.apis import multi_gpu_test, set_random_seed, single_gpu_test
from mmedit.core.distributed_wrapper import DistributedDataParallelWrapper
from mmedit.datasets import build_dataloader, build_dataset
from mmedit.models import build_model
import test_lpips
import test_batch

def parse_args():
    parser = argparse.ArgumentParser(description='mmediting tester')
    # parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic',action='store_true',help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--out', help='output result pickle file')
    parser.add_argument('--gpu-collect',action='store_true',help='whether to use gpu to collect results')
    parser.add_argument('--save-path',default=None,type=str,help='path to store images and if not given, will not save image')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--launcher',choices=['none', 'pytorch', 'slurm', 'mpi'],default='none',help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    config_path = './configs/restorers/GDPN/GDPN_ffhq_16xmod.py'
    scale = 16
    work_path = 'Diffusion_%dx_v2'%scale
    # test_iter='all'
    test_iter=1000
    work_dir_base = '/media/user/Expansion/Data_storage/FaceGAN'
    # work_dir_base = '.'
    if test_iter != 'all':
        args.checkpoint = work_dir_base+'/work_dirs/%s/iter_%s.pth'%(work_path,test_iter)
        checkpoints = [args.checkpoint]
    else:
        files = os.listdir(work_dir_base+'/work_dirs/%s'%work_path)
        checkpoint_names = [f for f in files if f.endswith('.pth')]
        checkpoints = [work_dir_base+'/work_dirs/%s/%s'%(work_path,f) for f in checkpoint_names]
    
    args.save_path = './results/%s/'%work_path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    cfg = mmcv.Config.fromfile(config_path)
    print(cfg.dscale)
    cfg.dscale = scale
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()

    # set random seeds
    if args.seed is not None:
        if rank == 0:
            print('set random seed to', args.seed)
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)

    loader_cfg = {
        **dict((k, cfg.data[k]) for k in ['workers_per_gpu'] if k in cfg.data),
        **dict(
            samples_per_gpu=1,
            drop_last=False,
            shuffle=False,
            dist=distributed),
        **cfg.data.get('test_dataloader', {})
    }

    data_loader = build_dataloader(dataset, **loader_cfg)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    args.save_image = args.save_path is not None
    empty_cache = cfg.get('empty_cache', False)

    for checkpoint in checkpoints:
        args.checkpoint = checkpoint
        print(checkpoint)
        if not distributed:
            _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(
                model,
                data_loader,
                save_path=args.save_path,
                save_image=args.save_image)
        else:
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            model = DistributedDataParallelWrapper(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)

            device_id = torch.cuda.current_device()
            _ = load_checkpoint(
                model,
                args.checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
            outputs = multi_gpu_test(
                model,
                data_loader,
                args.tmpdir,
                args.gpu_collect,
                save_path=args.save_path,
                save_image=args.save_image,
                empty_cache=empty_cache)

        if rank == 0 and 'eval_result' in outputs[0]:
            print('')
            # print metrics
            stats = dataset.evaluate(outputs)
            for stat in stats:
                print('Eval-{}: {}'.format(stat, stats[stat]))

            # save result pickle
            if args.out:
                print('writing results to {}'.format(args.out))
                mmcv.dump(outputs, args.out)
        
        test_batch.full_test(work_path,id_mode='mse')


if __name__ == '__main__':
    main()
