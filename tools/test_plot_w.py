from pprint import pprint
from genotypes import Genotype

import numpy as np

from multiobjective import get_cmap
import matplotlib.pyplot as plt

from train_search import REG_LOSS, CRITERION_LOSS


def filterhist(hist: dict) -> None:
    """
    Change historical dict inplace with optimal results only
    :param hist: Dict with structure:
    {
        0.2: {'criterion_loss': 1.5269827842712402,
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
        #clean
        temp_optimals={}
        for weight, optimal in optimals.items():
            if np.any(np.less(optimal, value)) or np.array_equal(optimal, value):
                #fica
                temp_optimals[weight] = optimal
            else:
                #sai
                hist.pop(weight)
        optimals = temp_optimals


if __name__ == '__main__':
    hist = {}
    pprint(hist)
    filterhist(hist)
    filename = "../test_plot.png"
    y_label = "Valid Accuracy"
    x_label = "L1 loss"
    title = "Regularization vs Criterion loss per weight of regularization"
    x_axis = [entry[REG_LOSS] for entry in hist.values()]
    y_axis = [entry[CRITERION_LOSS] for entry in hist.values()]
    weights = [weight[0] / weight[1] for weight in hist.keys()]

    data = zip(x_axis, y_axis, weights)
    data = sorted(data, key=lambda tup: tup[2], reverse=True)

    # Plot the data
    cmap = get_cmap(len(y_axis) + 1)
    plt.gcf().set_size_inches(10, 7, forward=True)
    for i, (x, y, w) in enumerate(data):
        plt.scatter(x, y, color=cmap(i), label="λ = %.0E" % w)


    # Add a legend
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Show the plot
    plt.savefig(filename)
    plt.show()