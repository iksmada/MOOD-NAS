import argparse
import re
from pprint import pprint
from genotypes import Genotype

import numpy as np

from multiobjective import get_cmap
import matplotlib.pyplot as plt

from train_search import REG_LOSS, CRITERION_LOSS, L1_LOSS, L2_LOSS


def filter_hist(hist: dict) -> None:
    """
    Change historical dict inplace with optimal results only
    :param hist: Dict with structure:
    {(0.0, 1.0): {'criterion_loss': 1.5269827842712402,
                  'model_size': 1.370134,
                  'reg_loss': 0.8349355459213257,
                  'train_acc': 46.3520000012207,
                  'valid_acc': 35.896000002441404},
        ...
    }
    """
    optimals = {}
    all_candidates = {key: (np.array([value[REG_LOSS], value[CRITERION_LOSS]])) for key, value in hist.items()}
    for key, value in all_candidates.items():
        if len(optimals) == 0:
            optimals[key] = value
            continue
        is_dominated = False
        for opt_objective in optimals.values():
            # if the result is greater in all objectives, then it is dominated
            if np.all(np.greater_equal(value, opt_objective)):
                is_dominated = True
                break
        if is_dominated:
            hist.pop(key)
            continue
        optimals[key] = value
        # clean
        temp_optimals={}
        for weight, optimal in optimals.items():
            if np.any(np.less(optimal, value)) or np.array_equal(optimal, value):
                # stays
                temp_optimals[weight] = optimal
            else:
                # removes
                hist.pop(weight)
        optimals = temp_optimals


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Plot frontier from logs")
    parser.add_argument("-l", "--log", type=argparse.FileType('r'), help="Multi search stage log", required=True)
    parser.add_argument("-o", "--output", type=str, required=False, help="Output image name", default="plot.png")
    parser.add_argument("-i", "--inches", type=float, help="Image size in inches", nargs='+', default=[6.4, 4.8])
    parser.add_argument('-d', '--dominated', action='store_true', default=False, help='Show dominated points')
    args = parser.parse_args()

    # filter logs
    lines = str(args.log.readlines())
    match = re.search(r"Selected regularization(?P<reg>.*?)\\n", lines)
    reg = match.group("reg")
    if L1_LOSS in reg:
        x_label = "L1 loss"
        loss = L1_LOSS
    elif L2_LOSS in reg:
        x_label = "L2 loss"
        loss = L2_LOSS
    else:
        raise RuntimeError("Cant decode line Selected regularization")

    match = re.finditer(r"hist = (?P<hist>.*?)\\n", lines)
    hist_str = list(match)[-1].group("hist")

    filename = args.output
    y_label = "Cross Entropy loss"

    orig_hist = eval(hist_str)[loss]
    pprint(orig_hist)
    hist = eval(hist_str)[loss]
    filter_hist(hist)  # inplace filter

    x_axis = [entry[REG_LOSS] for entry in hist.values()]
    y_axis = [entry[CRITERION_LOSS] for entry in hist.values()]

    if args.dominated:
        title = "Regularization vs Criterion loss Pareto frontier"
        x_axis_all = [entry[REG_LOSS] for entry in orig_hist.values()]
        y_axis_all = [entry[CRITERION_LOSS] for entry in orig_hist.values()]
        plt.scatter(x_axis_all, y_axis_all, label="Dominated Results")
        plt.scatter(x_axis, y_axis, label="Optimal Results")
        plt.xscale('log')
    else:
        title = "Regularization vs Criterion loss per weight of regularization"
        weights = [weight[0] for weight in hist.keys()]
        data = zip(x_axis, y_axis, weights)
        data = sorted(data, key=lambda tup: tup[2], reverse=True)

        # Plot the data
        cmap = get_cmap(len(y_axis))
        plt.gcf().set_size_inches(args.inches, forward=True)
        for i, (x, y, w) in enumerate(data):
            plt.scatter(x, y, color=cmap(i), label="$\\nu$ = %.0E" % w)

    # Add a legend
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Show the plot
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.show()
