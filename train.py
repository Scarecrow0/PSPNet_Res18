import argparse
import logging
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model.pspnet import PSPNet
from util import dataset, transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU

manual_seed = 666

random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
cudnn.deterministic = True


def main():
    global args
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    model = PSPNet(layers=args.layers, classes=args.classes,
                   zoom_factor=args.zoom_factor,
                   criterion=criterion,
                   pretrained=args.pretrained,
                   naive_ppm=args.naive_ppm)

    # set diffrent learning rate on different part of models
    modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
    modules_new = [model.ppm, model.cls_head, model.aux_head]
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
    args.index_split = 5
    optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # image pre-processing and augmentation
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    train_transform = transform.Compose([
        transform.Resize((args.train_h, args.train_w)),
        # augmentation
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    # initialize dataloader
    train_data = dataset.SemData(split='trainval', transform=train_transform)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    if args.evaluate:
        val_transform = transform.Compose([

            transform.Crop([args.train_h, args.train_w], crop_type='center', padding=mean,
                           ignore_label=args.ignore_label),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
        val_data = dataset.SemData(split='test', transform=val_transform)
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # start training
    logger.info('Starting training.')
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, epoch)
        writer.add_scalar('loss/train', loss_train, epoch_log)
        writer.add_scalar('mIoU/train', mIoU_train, epoch_log)
        writer.add_scalar('mAcc/train', mAcc_train, epoch_log)
        writer.add_scalar('allAcc/train', allAcc_train, epoch_log)

        if args.evaluate:
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
            writer.add_scalar('loss/val', loss_val, epoch_log)
            writer.add_scalar('mIoU/val', mIoU_val, epoch_log)
            writer.add_scalar('mAcc/val', mAcc_val, epoch_log)
            writer.add_scalar('allAcc/val', allAcc_val, epoch_log)

        if (epoch_log % args.save_freq == 0):
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       filename)


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data_time.update(time.time() - end)
        if args.zoom_factor != 8:
            h = int((target.size()[1] - 1) / 8 * args.zoom_factor + 1)
            w = int((target.size()[2] - 1) / 8 * args.zoom_factor + 1)
            # 'nearest' mode doesn't support align_corners mode and 'bilinear' mode is fine for downsampling
            target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='bilinear',
                                   align_corners=True).squeeze(1).long()
        input = input.cuda()
        target = target.cuda()

        output, main_loss, aux_loss = model(input, target)
        main_loss = main_loss.mean()
        aux_loss = aux_loss.mean()
        loss = main_loss + args.aux_weight * aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)

        # learning rate scheduling
        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        for index in range(0, args.index_split):
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)

        if (i + 1) % args.print_freq == 0:
            writer.add_scalar('loss/train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU/train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc/train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc/train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info(
        'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs, mIoU,
                                                                                       mAcc, allAcc))
    logger.info(f'remaining time: {int(t_h)}h {int(t_m)}min {int(t_s)}sec')

    return main_loss_meter.avg, mIoU, mAcc, allAcc


@torch.no_grad()
def validate(val_loader, model, criterion):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        if args.zoom_factor != 8:
            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        loss = criterion(output, target)

        loss = torch.mean(loss)

        output = F.softmax(output).max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        loss_meter.update(loss.item(), input.size(0))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    check(cfg)

    # backup the config file
    if os.path.exists(cfg.save_path):
        shutil.rmtree(cfg.save_path)

    os.makedirs(cfg.save_path)
    shutil.copyfile(args.config, os.path.join(cfg.save_path, "config.yaml"))
    return cfg


if __name__ == '__main__':
    args = get_parser()

    main()
