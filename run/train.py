import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import random_split
import torch.utils.data.distributed
import torch.utils.data.dataset
import torchvision
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from tensorboardX import SummaryWriter
import argparse
import os
import pprint
import logging
import json
import _init_paths
import dataset
from tqdm import tqdm
import numpy as np
import pickle
import time
import math
from pathlib import Path

from utils.utils import save_checkpoint, load_checkpoint, create_logger, load_model_state
from core.config import config as cfg
from core.function import train, validate
from utils.vis import save_torch_image
from utils.vis import save_pred_batch_images
from core.metrics import eval_metrics, AverageMeter
import segmentation_models_pytorch as smp


def get_optimizer(model):
    lr = cfg.TRAIN.LR
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def main():
    # 출력 경로 설정
    this_dir = Path(os.path.dirname(__file__))
    data_dir = (this_dir / '..' / cfg.DATA_DIR).resolve()
    output_dir = (this_dir / '..' / cfg.OUTPUT_DIR).resolve()

    # Cudnn 설정
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.autograd.set_detect_anomaly(True)
    gpus = [int(i) for i in cfg.GPUS.split(',')]

    # 데이터셋 생성
    print('=> Loading dataset ..')

    # 데이터 변형 함수 정의 (Augmentation)
    dataset = torchvision.datasets.VOCSegmentation(root=data_dir, download=True)

    num_data = dataset.__len__()
    num_valid = int(num_data * cfg.VALIDATION_RATIO)
    num_train = num_data - num_valid
    num_classes = 21

    train_dataset, valid_dataset = random_split(dataset, [num_train, num_valid])

    # 데이터로더 적재
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True)

    # 모델 생성
    print('=> Constructing models ..')
    model = torchvision.models.segmentation.fcn_resnet101(pretrained=True, num_classes=num_classes)

    # 모델 병렬화
    print('=> Paralleling models ..')
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # 옵티마이저 설정
    model, optimizer = get_optimizer(model)

    start_epoch = cfg.TRAIN.BEGIN_EPOCH
    end_epoch = cfg.TRAIN.END_EPOCH
    best_precision = 0
    step = 0
    if cfg.TRAIN.RESUME:
        start_epoch, model, optimizer, precision = load_checkpoint(model, optimizer, output_dir)

    # 학습
    print('=> Training patch model ..')
    for epoch in range(start_epoch, end_epoch):
        # Training Loop
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        criterion = nn.CrossEntropyLoss()
        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            with torch.autograd.set_detect_anomaly(True):
                # 예측
                pred = model(input)

                # 손실 계산
                loss = criterion(input, target)
                losses.update(loss.item())

                # 손실 역전파
                optimizer.zero_grad()
                if loss > 0:
                    loss.backward()
                optimizer.step()

                # 연산 시간 계산
                batch_time.update(time.time() - end)
                end = time.time()

            # 학습 정보 출력
            if step % cfg.PRINT_FREQ == 0:
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                      'Memory {memory:.1f}'.format(
                    epoch, step, len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    memory=gpu_memory_usage)
                print(msg)

                # 이미지 출력
                prefix = '{}_{:05}'.format(os.path.join(output_dir, 'train'), i)
                save_pred_batch_images(input, pred, target, prefix)

        # Validation Loop
        batch_time = AverageMeter()
        data_time = AverageMeter()
        overall_acc = AverageMeter()
        avg_per_class_acc = AverageMeter()
        avg_jacc = AverageMeter()
        avg_dice = AverageMeter()

        model.eval()

        with torch.no_grad():
            end = time.time()
            for j, (input, target) in enumerate(valid_loader):
                data_time.update(time.time() - end)

                # 예측
                pred = model(input)

                # 연산 시간 계산
                batch_time.update(time.time() - end)
                end = time.time()

                # 평가
                metric = eval_metrics(pred, target, num_classes)
                overall_acc.update(metric[0])
                avg_per_class_acc.update(metric[1])
                avg_jacc.update(metric[2])
                avg_dice.update(metric[3])

                # 학습 정보 출력
                if step % cfg.PRINT_FREQ == 0:
                    gpu_memory_usage = torch.cuda.memory_allocated(0)
                    msg = 'Test: [{0}/{1}]\t' \
                          'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed: {speed:.1f} samples/s\t' \
                          'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Memory {memory:.1f}'.format(
                        step, len(valid_loader), batch_time=batch_time,
                        speed=len(input) * input[0].size(0) / batch_time.val,
                        data_time=data_time, memory=gpu_memory_usage)
                    print(msg)

                    # 이미지로 출력
                    prefix = '{}_{:08}'.format(os.path.join(output_dir, 'valid'), step)
                    save_pred_batch_images(input, pred, target, prefix)

            # 패치 단위 PSNR 평가
                overall_acc.update(metric[0])
                avg_per_class_acc.update(metric[1])
                avg_jacc.update(metric[2])
                avg_dice.update(metric[3])
            msg = '(Evaluation)\tTOTAL_ACC: {0:.4f}\t' \
                  'AVG_CLASS_ACC: {1:.4f}\t' \
                  'AVG_JACC: {2:.4f}\t' \
                  'AVG_DICE: {3:.4f}'.format(
                      overall_acc.val,
                      avg_per_class_acc.val,
                      avg_jacc.val,
                      avg_dice.val
                  )
            print(msg)
            precision = overall_acc.val

        if precision > best_precision:
            best_precision = precision
            best_model = True
        else:
            best_model = False

        print('=> saving checkpoint to {} (Best: {})'.format(output_dir, best_model))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'precision': best_precision,
            'optimizer': optimizer.state_dict(),
        }, best_model, output_dir)

    final_model_state_file = os.path.join(output_dir, 'patch_final_state.pth.tar')
    print('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)


if __name__ == '__main__':
    main()
