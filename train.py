import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import nvidia_smi

from model import NetworkCIFAR as Network

TOP5 = 'top5'
TEST_TOP5 = 'test_top5'
TEST_LOSS = 'test_loss'
TEST_ACCURACY = 'test_acc'
VALID_TOP5 = 'valid_top5'
VALID_LOSS = 'valid_loss'
VALID_ACCURACY = 'valid_acc'
LOSS = 'loss'
ACCURACY = 'accuracy'


def create_parser():
    parser = argparse.ArgumentParser("train")
    parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
    parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_lines', type=int, default=5, help='number of report lines per stage')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device index')
    parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--model_path', type=str, help='path to save the model')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--arch', type=str, default='PC_DARTS_cifar', help='which architecture to use')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.9, help='portion of training data')
    return parser


def main(args_temp, force_log=False):
    global args
    args = args_temp

    log_path = 'logs/eval-{}-{}'.format(args.arch if args.save is None else args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(log_path, scripts_to_save=None)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %H:%M:%S', force=force_log)
    fh = logging.FileHandler(os.path.join(log_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    global log
    log = logging.getLogger("train")
    CIFAR_CLASSES = 10
    if args.set == 'cifar100':
        CIFAR_CLASSES = 100

    if not torch.cuda.is_available():
        log.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    log.info('gpu device = %d' % args.gpu)
    log.info("args = %s", args)

    genotype = genotypes.__dict__[args.arch]
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()
    if args.model_path is not None:
        utils.load(model, args.model_path)

    log.info("param size = %.2fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, test_transform = utils._data_transforms_cifar10(args)
    if args.set == 'cifar100':
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=test_transform)
        test_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=test_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=test_transform)
        test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=4)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=4)

    test_queue = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    history = defaultdict(list)
    for epoch in range(args.epochs):
        log.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, top5, train_obj = train(train_queue, model, criterion, optimizer)
        log.info('train_acc %f', train_acc)
        log.info('train_loss %f', train_obj)
        history[ACCURACY].append(train_acc)
        history[TOP5].append(top5)
        history[LOSS].append(train_obj)
        scheduler.step()

        log.info("Validation")
        valid_acc, valid_top5, valid_obj = infer(valid_queue, model, criterion)
        log.info('%s %f', VALID_ACCURACY, valid_acc)
        log.info('%s %f', VALID_LOSS, valid_obj)
        history[VALID_ACCURACY].append(valid_acc)
        history[VALID_TOP5].append(valid_top5)
        history[VALID_LOSS].append(valid_obj)

        utils.save(model, os.path.join(log_path, 'weights.pt'))

    log.info("Test")
    test_acc, test_top5, test_obj = infer(test_queue, model, criterion)
    log.info('%s %f', TEST_ACCURACY, test_acc)
    log.info('%s %f', TEST_TOP5, test_top5)
    log.info('%s %f', TEST_LOSS, test_obj)

    plt.gcf().set_size_inches(10, 7, forward=True)
    # Show the loss plot
    plt.plot(history[LOSS], label="Train loss")
    plt.plot(history[VALID_LOSS], label="Valid loss")
    plt.title("Train and Valid loss per epoch")
    plt.legend()
    plt.xlabel('epoch', fontsize=12)
    plt.yscale('log')
    plt.ylabel('loss', fontsize=12)
    plt.savefig(os.path.join(log_path, "loss.png"), bbox_inches='tight', dpi=100)
    plt.show()
    plt.clf()

    # Show the acc plot
    plt.plot(history[ACCURACY], label="Train accuracy")
    plt.plot(history[VALID_ACCURACY], label="Valid accuracy")
    plt.title("Train and Valid accuracy per epoch")
    plt.legend()
    plt.xlabel('epoch', fontsize=12)
    plt.xscale('log')
    plt.ylabel('accuracy %', fontsize=12)
    plt.savefig(os.path.join(log_path, "acc.png"), bbox_inches='tight', dpi=100)
    plt.show()
    plt.clf()

    # Show the top5 acc plot
    plt.plot(history[TOP5], label="Train top5 acc")
    plt.plot(history[VALID_TOP5], label="Valid top5 acc")
    plt.title("Train and Valid top5 accuracy per epoch")
    plt.legend()
    plt.xlabel('epoch', fontsize=12)
    plt.xscale('log')
    plt.ylabel('accuracy %', fontsize=12)
    plt.savefig(os.path.join(log_path, "top5.png"), bbox_inches='tight', dpi=100)
    plt.show()
    plt.clf()

    # remove handler to avoid getting other models output
    logging.getLogger().removeHandler(fh)


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % (len(train_queue) // args.report_lines) == 0:
            log.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(args.gpu)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            const = math.pow(2, -20)
            log.info("Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(100 * info.free / info.total,
                                                                                       info.total * const,
                                                                                       info.free * const,
                                                                                       info.used * const,))

    return top1.avg, top5.avg, objs.avg


def infer(queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            # 3 fixed lines of report for inference
            if step % (len(queue) // 3) == 0:
                log.info('infer %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    nvidia_smi.nvmlInit()
    main(args, True)
    nvidia_smi.nvmlShutdown()
