import os
import sys
import time
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import copy

from torch.autograd import Variable
from model_search_imagenet import Network
from model import NetworkImageNet as TrainNetwork
from architect import Architect
from train_search import L1_LOSS, TRAIN_ACC, VALID_ACC, REG_LOSS, CRITERION_LOSS, SIZE, L2_LOSS, GENOTYPE, param_loss

#data preparation, we random sample 10% and 2.5% from training set(each class) as train and val, respectively.
#Note that the data sampling can not use torch.utils.data.sampler.SubsetRandomSampler as imagenet is too large   
CLASSES = 1000



def main(args):
    global log
    log = logging.getLogger("train_search_imagenet")
    if not torch.cuda.is_available():
        log.info('no gpu device available')
        sys.exit(1)

    data_dir = os.path.join(args.data, 'imagenet_search')

    np.random.seed(args.seed)
    #torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    #log.info('gpu device = %d' % args.gpu)
    log.info("args = %s", args)
    #dataset_dir = '/cache/'
    #pre.split_dataset(dataset_dir)
    #sys.exit(1)
   # dataset prepare
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir,  'val')
        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    #dataset split     
    train_data1 = dset.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_data2 = dset.ImageFolder(valdir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_data = dset.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    num_train = len(train_data1)
    num_val = len(train_data2)
    print('# images to train network: %d' % num_train)
    print('# images to validate network: %d' % num_val)
    
    model = Network(args.init_channels, CLASSES, args.layers, criterion)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    log.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
               lr=args.arch_learning_rate, betas=(0.5, 0.999), 
               weight_decay=args.arch_weight_decay)
    
    test_queue = torch.utils.data.DataLoader(
                        valid_data, 
                        batch_size=args.batch_size, 
                        shuffle=False, 
                        pin_memory=True, 
                        num_workers=args.workers)

    train_queue = torch.utils.data.DataLoader(
        train_data1, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data2, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.workers)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    train_acc = None
    valid_acc = None
    l1_loss = torch.zeros(1)
    l2_loss = torch.zeros(1)
    criterion_loss = torch.zeros(1)
    genotype = model.module.genotype()
    log.info('initial genotype = %s', genotype)
    #architect = Architect(model, args)
    lr = args.learning_rate
    for epoch in range(args.epochs):
        current_lr = scheduler.get_lr()[0]
        log.info('Epoch: %d lr: %e', epoch, current_lr)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            log.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
            print(optimizer) 
        genotype = model.module.genotype()
        log.info('genotype = %s', genotype)
        arch_param = model.module.arch_parameters()
        #log.info(F.softmax(arch_param[0], dim=-1))
        #log.info(F.softmax(arch_param[1], dim=-1))
        # training
        train_acc, train_obj, l1_loss, l2_loss, criterion_loss = train(
            train_queue, valid_queue, model, optimizer, optimizer_a, criterion, lr, epoch, args.grad_clip,
            args.report_lines, args.begin, args.criterion_weight, args.l1_weight, args.l2_weight
        )
        scheduler.step()
        log.info('train_acc %f', train_acc)
        log.info('%s %f', L1_LOSS, l1_loss)
        log.info('%s %f', L2_LOSS, l2_loss)
        log.info('criterion_loss %f', criterion_loss)

        # validation
        if epoch >= args.epochs - 3:
            valid_acc, valid_obj = infer(valid_queue, model, criterion, args.report_lines)
            #test_acc, test_obj = infer(test_queue, model, criterion)
            log.info('valid_acc %f', valid_acc)
            #log.info('Test_acc %f', test_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))
        genotype = model.genotype()
        log.info('genotype = %s', genotype)

    log.info('last genotype = %s', genotype)
    model = TrainNetwork(48, CLASSES, 14, False, genotype)
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


def train(train_queue, valid_queue, model, optimizer, optimizer_a, criterion, lr, epoch, grad_clip, report_lines, begin,
          criterion_weight=1.0, l1_weight=-1, l2_weight=-1):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    criterion_loss = torch.zeros(1)
    l1_loss = torch.zeros(1)
    l2_loss = torch.zeros(1)
    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)
        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)
        
        if epoch >= begin:
            optimizer_a.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)
            loss_a.sum().backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            optimizer_a.step()
        #architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

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
        nn.utils.clip_grad_norm_(model.module.parameters(), grad_clip)
        optimizer.step()


        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % (len(train_queue) // report_lines) == 0:
            log.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg, l1_loss, l2_loss, criterion_loss



def infer(valid_queue, model, criterion, report_lines):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
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


def create_parser(parser=argparse.ArgumentParser("imagenet")):
    parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
    parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
    parser.add_argument('--set', type=str, default='imagenet', choices=['imagenet'])
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_lines', type=int, default=5, help='number of report lines per stage')
    parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float, default=6e-3, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--begin', type=int, default=35, help='batch size')

    parser.add_argument('--note', type=str, default='try', help='note for this run')
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
    args.save = 'logs/search-imagenet-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=None)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %H:%M:%S', force=True)
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(main(args))
