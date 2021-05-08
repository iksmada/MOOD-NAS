import logging
import os
import sys
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

import utils
from train_search import main, create_parser, TRAIN_ACC, REG_LOSS, CRITERION_LOSS

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

if __name__ == '__main__':
    parser = create_parser()
    parser.add_argument('-rw', '--reg_weight', type=float, default=list(range(1, 10)), nargs='+',
                        help='Regularization weight list negative power of ten to try with the model')
    args = parser.parse_args()

    args.save = 'multi-search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=None)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %H:%M:%S')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    weight_list = 10.0 ** -np.array(args.reg_weight)
    hist = {}
    for weight in weight_list:
        args.reg_weight = weight
        logging.info("Running train_search main with %s" % args)
        stats = main(args)
        if stats[weight].get(TRAIN_ACC) is not None:
            hist.update(stats)
            logging.info("hist = %s" % hist)
        torch.cuda.empty_cache()

    reg_losses = [x[REG_LOSS] for x in hist.values()]
    criterion_losses = [x[CRITERION_LOSS] for x in hist.values()]

    # Plot the data
    cmap = get_cmap(len(reg_losses)+1)
    for i, (x, y, w) in enumerate(zip(reg_losses, criterion_losses, hist.keys())):
        plt.scatter(x, y, color=cmap(i), label="w = %.0E" % w)

    # Add a legend
    plt.legend()
    plt.xlabel("L1 loss")
    plt.ylabel("Cross Entropy loss")
    plt.title("Regularization vs Criterion Loss per weight of regularization ")

    # Show the plot
    plt.savefig(os.path.join(args.save, "losses.png"))
    plt.show()
