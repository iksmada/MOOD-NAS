import logging
import os
import sys
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

import utils
from train_search import main, create_parser, L1_LOSS, L2_LOSS, CRITERION_LOSS, REG_LOSS


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


if __name__ == '__main__':
    parser = create_parser()
    reg_group = parser.add_mutually_exclusive_group()
    reg_group.add_argument('-l1', '--l1_weight', type=float, nargs='+',
                           help='L1 loss weight list negative power of ten to try with the model')
    reg_group.add_argument('-l2', '--l2_weight', type=float, nargs='+',
                           help='Regularization weight list negative power of ten to try with the model')
    args = parser.parse_args()

    args.save = 'logs/multi-search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=None)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %H:%M:%S')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    l2_mode = args.l1_weight is None

    weight_list = 10.0 ** -np.array(args.l2_weight if l2_mode else args.l1_weight)
    hist = {}
    for weight in weight_list:
        if l2_mode:
            args.l2_weight = weight
            args.l1_weight = 0
        else:
            args.l1_weight = weight
            args.l2_weight = 0
        stats = main(args)
        logging.info("stats = %s" % stats)
        for key in stats.keys():
            temp = hist.get(key, {})
            temp.update(stats[key])
            hist[key] = temp
        logging.info("hist = %s" % hist)
        torch.cuda.empty_cache()

    LOSS = (L2_LOSS if l2_mode else L1_LOSS)
    reg_losses = [x[REG_LOSS] for x in hist[LOSS].values()]
    criterion_losses = [x[CRITERION_LOSS] for x in hist[LOSS].values()]

    # Plot the data
    cmap = get_cmap(len(reg_losses) + 1)
    for i, (x, y, w) in enumerate(zip(reg_losses, criterion_losses, hist[LOSS].keys())):
        plt.scatter(x, y, color=cmap(i), label="w = %.0E" % w)

    # Add a legend
    plt.legend()
    if l2_mode:
        plt.xlabel("L2 loss")
    else:
        plt.xlabel("L1 loss")
    plt.ylabel("Cross Entropy loss")
    plt.title("Regularization vs Criterion Loss per weight of regularization ")

    # Show the plot
    plt.savefig(os.path.join(args.save, "losses.png"))
    plt.show()
