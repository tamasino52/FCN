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
import _init_paths # 이거 사용이 왜 안될까?
import pprint
import logging
import json
import lib.dataset
from tqdm import tqdm
import numpy as np
import pickle
import time
import math
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from lib.utils.utils import save_checkpoint, load_checkpoint, create_logger, load_model_state
from lib.core.config import config as cfg
from lib.core import function
from lib.utils.vis import save_pred_batch_images
from lib.core.metrics import eval_metrics, AverageMeter
import segmentation_models_pytorch as smp
from torchvision.datasets import VOCSegmentation
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib as mpl

from albumentations import HorizontalFlip, Compose, Resize, Normalize
import segmentation_models_pytorch as seg
from lib.dataset.voc import myVOCSegmentation
from lib.core.metrics import eval_metrics
from torchmetrics import IoU


def get_optimizer(model):
    lr = cfg.TRAIN.LR
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer

def custom_imshow(img):
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

def main():

    # 출력 경로 설정
    this_dir = Path(os.path.dirname(__file__))
    data_dir = (this_dir / '..' / cfg.DATA_DIR).resolve()
    log_dir = (this_dir / '..' / cfg.LOG_DIR).resolve()
    output_dir = (this_dir / '..' / cfg.OUTPUT_DIR).resolve()

    # 폴더가 없다면 폴더 생성
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)


    # Cudnn 설정
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')

        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
        torch.autograd.set_detect_anomaly(True)
        gpus = [int(i) for i in cfg.GPUS.split(',')]
    else:
        DEVICE = torch.device('cpu')

    print(DEVICE)
    print(torch.cuda.is_available())


    # 데이터셋 생성
    print('=> Loading dataset ..')

    # 데이터 변형 함수 정의 (Augmentation)

    # transformation을 정의합니다.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    h, w = 224, 224

    transform_tran = Compose([Resize(h, w),
                              HorizontalFlip(p=0.5),
                              Normalize(mean=mean, std=std)])

    transform_val = Compose([Resize(h, w),
                             Normalize(mean=mean, std=std)
                             ])

    train_dataset = myVOCSegmentation(cfg.DATA_DIR, year='2012', image_set='train', download=True,
                                      transforms=transform_tran)
    val_dataset = myVOCSegmentation(cfg.DATA_DIR, year='2012', image_set='val', download=True,
                                    transforms=transform_val)

    print('Dataset Length : train({}), validation({})'.format(len(train_dataset), len(val_dataset)))


    # 데이터로더 생성
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True)

    # 모델 생성
    print('=> Constructing models ..')
    model = seg.Unet(classes=21, activation='softmax2d')
    model = model.cuda()

    # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # 옵티마이저 설정
    model, optimizer = get_optimizer(model)

    start_epoch = cfg.TRAIN.BEGIN_EPOCH
    end_epoch = cfg.TRAIN.END_EPOCH
    best_precision = 0

    if cfg.TRAIN.RESUME:
        start_epoch, model, optimizer, precision = load_checkpoint(model, optimizer, output_dir)

    criterion = nn.BCELoss()

    writer_dict = {
        'writer': SummaryWriter(log_dir=log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # 학습
    print('=> Training model ..')
    for epoch in range(start_epoch, end_epoch):
        # Training Loop
        batch_time = AverageMeter()
        losses = AverageMeter()

        model.train()
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            input = input.cuda()
            target = target.cuda()

            # 예측
            pred = model(input)

            # 손실 계산
            loss = criterion(pred, target)
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
            if i % cfg.PRINT_FREQ == 0:
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                      'Memory {memory:.1f}'.format(
                    epoch, i, len(train_loader),
                    batch_time=batch_time,
                    loss=losses,
                    memory=gpu_memory_usage)
                print(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

                # 이미지 출력
                prefix = '{}_{:05}'.format(os.path.join(output_dir, 'train'), i)
                grid_iamge = save_pred_batch_images(input, pred, target, prefix)
                cv2.imshow("Images", grid_iamge)


        # Validation Loop
        batch_time = AverageMeter()
        avg_iou = AverageMeter()
        eval_metrics = IoU(21)
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(valid_loader):

                input = input.cuda()
                target = target.cuda()

                # 예측
                pred = model(input)

                # 연산 시간 계산
                batch_time.update(time.time() - end)
                end = time.time()

                pred = pred.cpu()
                target = target.cpu()

                # 평가
                metric = eval_metrics(pred, target.type(torch.int))
                avg_iou.update(metric)

                # 학습 정보 출력
                if i % cfg.PRINT_FREQ == 0:
                    gpu_memory_usage = torch.cuda.memory_allocated(0)
                    msg = 'Test: [{0}/{1}]\t' \
                          'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed: {speed:.1f} samples/s\t' \
                          'Memory {memory:.1f}'.format(
                        i, len(valid_loader), batch_time=batch_time,
                        speed=len(input) / batch_time.val,
                        memory=gpu_memory_usage)
                    print(msg)

                    writer = writer_dict['writer']
                    global_steps = writer_dict['valid_global_steps']
                    writer.add_scalar('batch_time', batch_time.val, global_steps)
                    writer.add_scalar('avg_iou', avg_iou.val, global_steps)
                    writer_dict['train_global_steps'] = global_steps + 1

                    # 이미지로 출력
                    prefix = '{}_{:08}'.format(os.path.join(output_dir, 'valid'), i)
                    grid_iamge = save_pred_batch_images(input, pred, target, prefix)
                    cv2.imshow("Images", grid_iamge)

            avg_iou.update(metric)

            msg = '(Evaluation)\tMEAN IOU: {0:.4f}'.format(avg_iou.val)
            print(msg)
            precision = avg_iou.val


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

    final_model_state_file = os.path.join(output_dir)
    print('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

    writer_dict['writer'].close()

if __name__ == '__main__':
    main()
