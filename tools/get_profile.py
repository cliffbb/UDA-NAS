import _init_paths
import argparse
import torch
import numpy as np
from tqdm import tqdm

from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmseg.models import build_segmentor

from torchsummary import summary


def parse_args():
    parser = argparse.ArgumentParser(description='Get the FLOPs of a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--shape', type=int, nargs='+', default=[1024, 1024], help='input image size')
    parser.add_argument('--num-runs', type=int, default=300, help='number of runs to compute average forward timing. default is 300')
    parser.add_argument('--warmup-runs', type=int, default=10, help='number of warmup runs to avoid initial slow speed. default is 10')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    
    dummy_input = torch.randn(1, *input_shape, dtype=torch.float).cuda()
    
    model.eval()
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
    
    # calulating FLOP and params
    print('Calculating FLOP and number of params...')
    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}'.format(split_line, input_shape, flops, params))
    
    # calculating latency and FPS
    print('Calculating latency and FPS...')
    # initialize timers
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = np.zeros((args.num_runs, 1))
    # GPU warmup
    for _ in range(args.warmup_runs):
        _ = model(dummy_input)
    # measure latency
    model.eval()
    with torch.no_grad():
        starter.record()
        # GPU synchronization
        torch.cuda.synchronize()
        for rep in tqdm(range(args.num_runs)):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # GPU synchronization
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    latency = np.sum(timings) / args.num_runs # in milliseconds
    fps = 1000 / latency # in seconds (s)
    print('Input shape: {:}\nFPS: {:.2f} ms\nLatency: {:.2f} ms\n{:}'.format(input_shape, fps, latency, split_line))
    torch.cuda.empty_cache()
    
    
if __name__ == '__main__':
    main()
    

    