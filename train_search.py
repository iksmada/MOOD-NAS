import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import utils
import logging
import argparse
import torch.nn as nn
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from model_search import Network
from model import NetworkCIFAR as TrainNetwork
from architect import Architect

from sklearn.model_selection import train_test_split

TRAIN_ACC = "train_acc"
VALID_ACC = "valid_acc"
REG_LOSS = "reg_loss"
L1_LOSS = "L1_loss"
L2_LOSS = "L2_loss"
CRITERION_LOSS = "criterion_loss"
SIZE = "model_size"
GENOTYPE = "genotype"


def main(args):
    global log
    log = logging.getLogger("train_search")
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

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()
    log.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, _ = utils._data_transforms_cifar10(args)
    if args.set == 'cifar100':
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    targets = train_data.targets
    train_idx = np.arange(len(targets))
    if args.subsample > 0:
        train_idx, _ = train_test_split(
            train_idx,
            test_size=1 - args.subsample,
            shuffle=True,
            stratify=targets)
    num_train = len(train_idx)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idx[indices[:split]]),
        pin_memory=True, num_workers=4)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idx[indices[split:num_train]]),
        pin_memory=True, num_workers=4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    train_acc = None
    valid_acc = None
    l1_loss = torch.zeros(1)
    l2_loss = torch.zeros(1)
    criterion_loss = torch.zeros(1)
    genotype = model.genotype()
    log.info('initial genotype = %s', genotype)
    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]
        log.info('epoch %d lr %e', epoch, lr)

        # model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        # training
        train_acc, train_obj, l1_loss, l2_loss, criterion_loss = train(
            train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, args.grad_clip,
            args.report_lines, args.unrolled,
            args.criterion_weight, args.l1_weight, args.l2_weight
        )
        scheduler.step()
        log.info('train_acc %f', train_acc)
        log.info('%s %f', L1_LOSS, l1_loss)
        log.info('%s %f', L2_LOSS, l2_loss)
        log.info('criterion_loss %f', criterion_loss)

        # validation
        if args.epochs - epoch <= 1:
            valid_acc, valid_obj = infer(valid_queue, model, criterion, args.report_lines)
            log.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))
        genotype = model.genotype()
        log.info('genotype = %s', genotype)

    log.info('last genotype = %s', genotype)
    model = TrainNetwork(36, CIFAR_CLASSES, 20, False, genotype)
    model_size_mb = utils.count_parameters_in_MB(model)
    log.info("Train model param size = %.2fMB", model_size_mb)

    return {
        L1_LOSS: {
            tuple([args.l1_weight, args.criterion_weight]): {
                TRAIN_ACC: train_acc,
                VALID_ACC: valid_acc,
                REG_LOSS: l1_loss.cpu().data.item(),
                CRITERION_LOSS: criterion_loss.cpu().data.item(),
                SIZE: model_size_mb,
                GENOTYPE: genotype

            }
        },
        L2_LOSS: {
            tuple([args.l2_weight, args.criterion_weight]): {
                TRAIN_ACC: train_acc,
                VALID_ACC: valid_acc,
                REG_LOSS: l2_loss.cpu().data.item(),
                CRITERION_LOSS: criterion_loss.cpu().data.item(),
                SIZE: model_size_mb,
                GENOTYPE: genotype
            }
        }
    }


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, grad_clip, report_lines,
          unrolled, criterion_weight=1.0, l1_weight=-1, l2_weight=-1):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    criterion_loss = torch.zeros(1)
    l1_loss = torch.zeros(1)
    l2_loss = torch.zeros(1)
    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        # input_search, target_search = next(iter(valid_queue))
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)
        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)

        if epoch >= 15:
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=unrolled)

        optimizer.zero_grad()
        logits = model(input)
        criterion_loss = criterion(logits, target)
        loss = criterion_weight * criterion_loss
        if l1_weight >= 0:
            l1_loss = param_loss(model, nn.L1Loss(reduction='sum'))
            loss += l1_weight * l1_loss
        if l2_weight >= 0:
            l2_loss = param_loss(model, nn.MSELoss(reduction='sum'))
            loss += l2_weight * l2_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % (len(train_queue) // report_lines) == 0:
            log.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg, l1_loss, l2_loss, criterion_loss


def param_loss(model, penalty):
    reg_loss = 0
    for param in model.parameters():
        reg_loss += penalty(param, torch.zeros(param.size(), device=torch.device('cuda')))
    return reg_loss


def infer(valid_queue, model, criterion, report_lines):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % (len(valid_queue) // report_lines) == 0:
                log.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def create_parser():
    parser = argparse.ArgumentParser("train_search")
    parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
    parser.add_argument('--set', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_lines', type=int, default=5, help='number of report lines per stage')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device index')
    parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--subsample', type=float, default=0, help='Sub sample proportion from 0 to 1. Use it to reduce'
                        ' number of samples')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    parser.add_argument('-l1', '--l1_weight', type=float, default=-1,
                        help='Regularization weight (positive value) to add to the model')
    parser.add_argument('-l2', '--l2_weight', type=float, default=-1,
                        help='Regularization weight (positive value) to add to the model')
    parser.add_argument('-cw', '--criterion_weight', type=float, default=1.0,
                        help='Criterion loss weight (positive value) to add to the model')

    args = parser.parse_args()
    args.save = 'logs/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=None)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %H:%M:%S', force=True)
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(main(args))
