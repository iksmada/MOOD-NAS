import argparse
import os
import re

import numpy as np
import pandas as pd

from ofa.utils import count_parameters, count_net_flops, measure_net_latency
import matplotlib.pyplot as plt
from pandas import DataFrame

import genotypes
from genotypes import Genotype
from model import NetworkCIFAR
from multiobjective import get_cmap
from train_search import L1_LOSS, L2_LOSS, TRAIN_ACC, VALID_ACC, CRITERION_LOSS

WEIGHT = "weight"

MODEL_NAME = "Model name"

TRAIN_LOSS = "train_loss"
VALID_LOSS = "valid_loss"
TEST_LOSS = "test_loss"
TEST_ACC = "test_acc"
SEARCH_LOSS = "search_loss"
SEARCH_ACC = "search_acc"


def plot_columns(df: DataFrame, x_column, y_column, filename="plot_table.png", negate=None):
    x_axis = df.loc[:, x_column]
    y_axis = df.loc[:, y_column]
    weights = df.loc[:, WEIGHT]

    data = zip(x_axis, y_axis, weights)
    data = sorted(data, key=lambda tup: tup[2], reverse=True)
    # Plot the data
    cmap = get_cmap(len(y_axis) + 1)
    plt.gcf().set_size_inches(10, 7, forward=True)
    for i, (x, y, w) in enumerate(data):
        if isinstance(negate, (int, float)):
            plt.scatter(negate - x, negate - y, color=cmap(i), label="w = %.0E" % w)
        else:
            plt.scatter(x, y, color=cmap(i), label="w = %.0E" % w)

    # Add a legend
    plt.legend()
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"{x_column} vs {y_column} per weight")

    # Show the plot
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("train_search")
    parser.add_argument("-t", "--train", type=argparse.FileType('r'), nargs='+', required=True,
                        help="Train/Evaluation stage logs")
    parser.add_argument("-s", "--search", type=argparse.FileType('r'), nargs='+', required=True,
                        help="Search stage logs")
    parser.add_argument("-o", "--output", type=str, required=False, help="Output file name", default="loss_table.csv")
    args = parser.parse_args()

    data = []
    for log in args.train:
        row = []
        try:
            # evaluation stage metrics
            lines = str(log.readlines())
            match = re.search(r"arch='(?P<name>.*?)'", lines)
            name = match.group("name")
            row.append(name)
            # l2_loss_2e01 -> 2e-01
            weight_value = float(name.split("_")[-1].replace("e", "e-"))
            row.append(weight_value)
            match = re.search(r"param size.*?(?P<value>\d*\.\d+)MB", lines)
            param_size = float(match.group("value"))
            row.append(param_size)
            for metric in [TRAIN_LOSS, TRAIN_ACC, VALID_LOSS, VALID_ACC, TEST_LOSS, TEST_ACC]:
                value = float(re.findall(rf'{metric}(?:uracy)? (?P<value>\d*\.\d+)', lines)[-1])
                row.append(value)
        except Exception as e:
            print(f"Error '{e}' while processing file {log.name}")
            while len(row) < 9:
                row.append(None)

        try:
            # search stage metrics
            genotype = genotypes.__dict__[name]
            genotype_str = str(genotype)
            match = False
            for s_log in args.search:
                s_lines = str(s_log.readlines())
                s_log.seek(0, 0)
                # ((?!\\n).)* = anything except new line escaped
                match = re.search(r"stats = (?P<stats>{((?!\\n).)*" + re.escape(genotype_str) + r".*?})\\n\",", s_lines)
                if match:
                    stats = eval(match.group("stats"))
                    # L2 loss case
                    if list(stats.get(L1_LOSS).keys())[0][0] == -1:
                        LOSS = L2_LOSS
                    # L1 loss case
                    elif list(stats.get(L2_LOSS).keys())[0][0] == -1:
                        LOSS = L1_LOSS
                    else:
                        raise Exception("L1 and L2 loss have w = -1")
                    values = list(stats.get(LOSS).values())[0]
                    search_loss = values[CRITERION_LOSS]
                    row.append(search_loss)
                    search_acc = values[VALID_ACC]
                    row.append(search_acc)
                    break
            if not match:
                raise Exception(f"Didn't find {name} on eval logs")
        except Exception as e:
            print(f"Error '{e}' while processing file {log.name}")
            while len(row) < 11:
                row.append(None)

        try:
            # model profiling
            genotype = genotypes.__dict__[name]
            match = re.search(r"init_channels=(?P<value>\d+)", lines)
            init_channels = int(match.group("value"))
            match = re.search(r"layers=(?P<value>\d+)", lines)
            layers = int(match.group("value"))
            match = re.search(r"drop_path_prob=(?P<value>\d+\.\d+)", lines)
            drop_path_prob = float(match.group("value"))
            match = re.search(r"auxiliary=(?P<value>\w+)", lines)
            auxiliary = bool(match.group("value"))
            model = NetworkCIFAR(init_channels, 10, layers, auxiliary, genotype)
            model.cuda()
            model.drop_path_prob = drop_path_prob
            parameters = int(count_parameters(model))
            row.append(parameters)
            data_shape = (1, 3, 32, 32)
            net_flops = int(count_net_flops(model, data_shape=data_shape))
            row.append(net_flops)
            total_time, measured_latency = measure_net_latency(model, l_type='gpu8', fast=True,
                                                               input_shape=data_shape[1:], clean=True)
            row.append(total_time)
            print("latency GPU %s: %s" % (name, measured_latency))

            total_time, measured_latency = measure_net_latency(model, l_type='cpu', fast=True,
                                                               input_shape=data_shape[1:], clean=True)
            row.append(total_time)
            print("latency CPU %s: %s" % (name, measured_latency))
        except Exception as e:
            print(f"Error '{e}' while processing file {log.name}")

        if len(row) > 0:
            data.append(row)

    df = pd.DataFrame(data, columns=[MODEL_NAME, WEIGHT, "Size",
                                     TRAIN_LOSS, TRAIN_ACC,
                                     VALID_LOSS, VALID_ACC,
                                     TEST_LOSS, TEST_ACC,
                                     SEARCH_LOSS, SEARCH_ACC,
                                     "Parameters", "FLOPs",
                                     "Latency GPU", "Latency CPU"])
    df.set_index(keys=MODEL_NAME, inplace=True)
    df.sort_values(by=WEIGHT, inplace=True, ascending=False)
    pd.set_option("display.max_rows", None, "display.max_columns", None, "display.width", None)
    print(df)
    print(df.loc[:, np.invert(df.columns.isin([TRAIN_LOSS, TRAIN_ACC, VALID_LOSS, VALID_ACC]))])
    output = args.output
    df.to_csv(output)
    filename, file_extension = os.path.splitext(output)
    plot_columns(df, SEARCH_LOSS, VALID_LOSS, f"{filename}_search_vs_valid_loss.png")
    plot_columns(df, SEARCH_ACC, VALID_ACC, f"{filename}_search_vs_valid_acc.png", 100)




