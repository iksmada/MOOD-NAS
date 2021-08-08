import logging
import os
import sys
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

import utils
import train_search
from WeightEstimator import WeightEstimator
from train_search import create_parser, L1_LOSS, L2_LOSS, CRITERION_LOSS, REG_LOSS


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def plot_frontier(title: str, weight_est: WeightEstimator, save_path="", xlabel=None, ylabel=None, ):
    plt.clf()
    plt.title(title)
    plt.scatter(list(pair[0] for pair in weight_est.results.values()),
                list(pair[1] for pair in weight_est.results.values()),
                label="Dominated Results")
    plt.scatter(list(pair[0] for pair in weight_est.optimal_results),
                list(pair[1] for pair in weight_est.optimal_results),
                label="Optimal Results")
    if xlabel is not None:
        plt.xlabel(xlabel)
        plt.xscale('log')
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join(save_path, "frontier_%s.png" % str(len(weight_est.visited_pairs)).zfill(2)))
    plt.show()


if __name__ == '__main__':
    log = logging.getLogger("multiobjective")
    parser = create_parser()
    parser.add_argument('-o', '--objective', type=str, required=True, choices={"l1", "l2"},
                        help='Set the second objective to optimize')
    args = parser.parse_args()

    args.save = 'logs/multi-search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=None)
    log_format = '%(asctime)s %(name)s - %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %H:%M:%S', force=True)
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # save the REG type
    if args.objective == "l2":
        REG = L2_LOSS
    elif args.objective == "l1":
        REG = L1_LOSS
    else:
        raise ValueError("Argument '--objective %s' is not implemented" % args.objective)

    weightEst = WeightEstimator(initial_weights=(np.array([0.2, 0.8]), np.array([0.0, 1.0])))
    hist = {}
    while weightEst.has_next():
        weight = weightEst.get_next()
        log.info("weight = %s", weight)
        args.criterion_weight = weight[1]
        if REG is L2_LOSS:
            args.l2_weight = weight[0]
            args.l1_weight = -1
        elif REG is L1_LOSS:
            args.l1_weight = weight[0]
            args.l2_weight = -1

        # do the training, about 2,5 hours
        stats = train_search.main(args)

        log.info("stats = %s" % stats)
        for key in stats.keys():
            temp = hist.get(key, {})
            temp.update(stats[key])
            hist[key] = temp
        log.info("hist = %s" % hist)

        # save the values to the weight estimator
        reg_stats = stats.get(REG).get(weight[0])
        reg_loss = reg_stats.get(REG_LOSS)
        log.info("reg_loss = %f", reg_loss)
        criterion_loss = reg_stats.get(CRITERION_LOSS)
        log.info("criterion_loss = %f", criterion_loss)
        weightEst.set_result(np.array([reg_loss, criterion_loss]), weight)
        # plot
        plot_frontier("Regularization vs Criterion loss Pareto frontier", weightEst, args.save, REG, "Cross Entropy loss")
        torch.cuda.empty_cache()

    reg_losses = [x[REG_LOSS] for x in hist[REG].values()]
    criterion_losses = [x[CRITERION_LOSS] for x in hist[REG].values()]

    # Plot the data
    cmap = get_cmap(len(reg_losses) + 1)
    for i, (x, y, w) in enumerate(zip(reg_losses, criterion_losses, hist[REG].keys())):
        plt.scatter(x, y, color=cmap(i), label="w = %.0E" % w)

    # Add a legend
    plt.legend()
    plt.xlabel(REG.replace("_", " "))
    plt.ylabel("Cross entropy loss")
    plt.title("Regularization vs Criterion loss per weight of regularization ")

    # Show the plot
    plt.savefig(os.path.join(args.save, "losses.png"))
    plt.show()
