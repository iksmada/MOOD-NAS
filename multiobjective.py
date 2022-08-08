import logging
import os
import sys
import time
from math import floor, log10

import numpy as np
import torch
import matplotlib.pyplot as plt

import utils
import train_search
from WeightEstimator import WeightEstimator
from train_search import create_parser, L1_LOSS, L2_LOSS, CRITERION_LOSS, REG_LOSS, GENOTYPE


def get_cmap(n, name='gist_rainbow'):
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
    plt.savefig(os.path.join(save_path, "frontier_%s.png" % str(len(weight_est.visited_pairs)).zfill(2)),
                bbox_inches='tight', dpi=100)
    plt.show()


def create_genotype_name(weights: np.ndarray, loss: str, weights_list: list) -> str:
    wl = weights[0]  # local weight
    assert wl in weights_list, f"The weight {wl} must be in the weight list"
    number_of_digits = 1
    round_n = round_to_n(wl, number_of_digits)
    while number_of_digits <= 5:
        round_n = round_to_n(wl, number_of_digits)
        round_n_list = [round_to_n(item, number_of_digits) for item in weights_list]
        if round_n_list.count(round_n) > 1:
            number_of_digits += 1
        else:
            break
    if wl == 0:
        exponent = 0
    else:
        # remove the negative value of the exponent
        exponent = -int(floor(log10(abs(wl)))) + (number_of_digits - 1)
    mantissa = round(round_n * pow(10, exponent))
    # it should be negative exponent, but we are removing the negative symbol (-)
    # of the exponent because var names cant contain this char
    return "%s_%de%02d" % (loss, mantissa, exponent)


def round_to_n(x: float, n: int) -> float:
    """
    Method to round to n significant figures.
    :param x: number to be rounded to n significant digits
    :param n: number of significant digits
    :return: number with n significant digits
    """
    if x == 0:
        return 0
    else:
        return round(x, -int(floor(log10(abs(x)))) + (n - 1))


if __name__ == '__main__':
    log = logging.getLogger("multiobjective")
    parser = create_parser()
    parser.add_argument('-o', '--objective', type=str, required=True, choices={"l1", "l2"},
                        help='Set the second objective to optimize')
    parser.add_argument('-d', '--delta', type=float, required=False, default=0.15,
                        help='Set the minimum normalized distance between points on the Pareto frontier to stop'
                             ' populating it.')
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
    log.info("Selected regularization %s", REG)

    log.info("Start of multi weight search algorithm")
    weightEst = WeightEstimator(delta=args.delta, initial_weights=(np.array([0.2, 0.8]), np.array([0.0, 1.0])))
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

        log.info("stats = %s", stats)
        for key in stats.keys():
            temp = hist.get(key, {})
            temp.update(stats[key])
            hist[key] = temp

        # save the values to the weight estimator
        reg_stats = stats.get(REG).get(tuple(weight))
        reg_loss = reg_stats.get(REG_LOSS)
        log.info("reg_loss = %f", reg_loss)
        criterion_loss = reg_stats.get(CRITERION_LOSS)
        log.info("criterion_loss = %f", criterion_loss)
        weightEst.set_result(np.array([reg_loss, criterion_loss]), weight)
        # plot
        plot_frontier("Regularization vs Criterion loss Pareto frontier", weightEst, args.save, REG,
                      "Cross Entropy loss")
        torch.cuda.empty_cache()

    log.info("hist = %s", hist)
    log.info("End of multi weight search algorithm")
    log.info("Plotting and printing optimal result")
    reg_losses = [x[REG_LOSS] for x in hist[REG].values()]
    criterion_losses = [x[CRITERION_LOSS] for x in hist[REG].values()]
    genotypes = [x[GENOTYPE] for x in hist[REG].values()]

    # remove not optimal
    opt_reg_losses = []
    opt_criterion_losses = []
    opt_weights = []
    all_weights = [weight[0] for weight in hist[REG].keys()]
    for reg, criterion, gen, weight in zip(reg_losses, criterion_losses, genotypes, hist[REG].keys()):
        candidate = np.array([reg, criterion])
        # check if candidate is optimal
        if np.any(np.all(candidate == weightEst.optimal_results, axis=1)):
            opt_reg_losses.append(reg)
            opt_criterion_losses.append(criterion)
            opt_weights.append(weight[0])
            # eg: l2_loss_1e3
            gen_name = create_genotype_name(weight, REG, all_weights)
            log.info("Optimal weight = %s, reg loss = %f, criterion loss = %f\n%s = %s",
                     weight[0], reg, criterion, gen_name, gen)

    # order by lambda to make legend linear
    data = zip(opt_reg_losses, opt_criterion_losses, opt_weights)
    data = sorted(data, key=lambda tup: tup[2], reverse=True)

    plt.clf()
    # Plot the data
    cmap = get_cmap(len(opt_reg_losses))
    plt.gcf().set_size_inches(10, 7, forward=True)
    for i, (x, y, w) in enumerate(data):
        plt.scatter(x, y, color=cmap(i), label="$\\nu$ = %.0E" % w)

    # Add a legend
    plt.legend()
    plt.xlabel(REG.replace("_", " "))
    plt.ylabel("Cross Entropy loss")
    plt.title("Regularization vs Criterion loss per weight of regularization")

    # Show the plot
    plt.savefig(os.path.join(args.save, "losses.png"), bbox_inches='tight', dpi=100)
    plt.show()
