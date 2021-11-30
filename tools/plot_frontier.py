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

    match = re.search(r"hist = (?P<hist>.*?)\\n", lines)
    hist = match.group("hist")

    hist = eval(hist)[loss]
    pprint(hist)
    filter_hist(hist)
    filename = args.output
    y_label = "Cross Entropy loss"
    title = "Regularization vs Criterion loss per weight of regularization"
    x_axis = [entry[REG_LOSS] for entry in hist.values()]
    y_axis = [entry[CRITERION_LOSS] for entry in hist.values()]
    weights = [weight[0] for weight in hist.keys()]

    data = zip(x_axis, y_axis, weights)
    data = sorted(data, key=lambda tup: tup[2], reverse=True)

    # Plot the data
    cmap = get_cmap(len(y_axis) + 1)
    plt.gcf().set_size_inches(args.inches, forward=True)
    for i, (x, y, w) in enumerate(data):
        plt.scatter(x, y, color=cmap(i), label="w = %.0E" % w)

    # Add a legend
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Show the plot
    plt.savefig(filename)
    plt.show()