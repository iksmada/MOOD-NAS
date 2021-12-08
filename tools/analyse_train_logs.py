import argparse
import os
import re

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm

from ofa.utils import count_parameters, count_net_flops, measure_net_latency
import matplotlib.pyplot as plt
from pandas import DataFrame

import genotypes
from genotypes import Genotype
from model import NetworkCIFAR
from multiobjective import get_cmap
from train_search import L1_LOSS, L2_LOSS, TRAIN_ACC, VALID_ACC, CRITERION_LOSS, REG_LOSS

WEIGHT = "Weight"

MODEL_NAME = "Model name"
TRAIN_LOSS = "train_loss"
VALID_LOSS = "valid_loss"
TEST_LOSS = "test_loss"
TEST_ACC = "test_acc"
SEARCH_CRIT_LOSS = "Search crit loss"
SEARCH_REG_LOSS = "Search reg loss"
SEARCH_ACC = "Search acc"
FLOPS = "FLOPs"
PARAMETERS = "Parameters"
LATENCY_CPU = "Latency CPU"

name = {
    TRAIN_LOSS: "Train loss",
    TRAIN_ACC: "Train acc",
    VALID_LOSS: "Valid loss",
    VALID_ACC: "Valid acc",
    TEST_LOSS: "Test loss",
    TEST_ACC: "Test acc",
}


def plot_columns(df: DataFrame, x_column: str, y_column: str, filename="plot_table.png", negate: int = None,
                 x_scale='log', y_scale='log', show_weights: bool = True, show_colorbar=False, inches: tuple = (6.4, 4.8)):
    """
    Plot DataFrame columns

    :param df:
    :param x_column: Column name to set as x axis
    :param y_column: Column name to set as y axis
    :param filename: Save path
    :param negate: Negate the values and add it to some offset passed as int here
    :param x_scale: {"linear", "log", "symlog", "logit", ...} or `.ScaleBase`
    :param y_scale: {"linear", "log", "symlog", "logit", ...} or `.ScaleBase`
    :param show_weights: Show detailed weight values instead of colorbar
    :param inches: Figure dimension width x height tuple in inches
    """
    plt.clf()
    x_axis = df.loc[:, x_column]
    y_axis = df.loc[:, y_column]
    if isinstance(negate, (int, float)):
        x_axis = negate - x_axis.to_numpy()
        y_axis = negate - y_axis.to_numpy()
    weights = df.loc[:, WEIGHT]

    # Plot the data
    cmap = get_cmap(len(y_axis))
    plt.gcf().set_size_inches(inches, forward=True)
    if show_weights:
        data = zip(x_axis, y_axis, weights)
        data = sorted(data, key=lambda tup: tup[2], reverse=True)
        for i, (x, y, w) in enumerate(data):
            plt.scatter(x, y, color=cmap(i), label="$\\nu$ = %.0E" % w)
        # Add a legend
        plt.legend()

    else:
        plt.scatter(x_axis, y_axis, c=range(len(y_axis)), cmap=cmap)
        if show_colorbar:
            cbar = plt.colorbar(ScalarMappable(norm=LogNorm(vmin=min(weights), vmax=max(weights)), cmap=cmap))
            cbar.set_label("$\\nu$", rotation=0)

    x_label = name.get(x_column, x_column)
    plt.xlabel(x_label)
    y_label = name.get(y_column, y_column)
    plt.ylabel(y_label)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.title(f"{x_label} vs {y_label} per Weight ($\\nu$)")

    # Show the plot
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.show()


def plot_correlation(df: DataFrame, filename="correlation_matrix.png"):
    plt.clf()
    f = plt.figure(figsize=(11, 10))
    data = np.abs(df.corr(method="spearman").to_numpy())
    plt.matshow(data, fignum=f.number)

    # legend
    columns = [name.get(column, column) for column in df.select_dtypes(['number']).columns]
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), columns, fontsize=14, rotation=45, ha="left",
               rotation_mode="anchor")
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), columns, fontsize=14)

    # colorbar
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.clim(min(0.5, np.min(data)))

    # cell values
    for (x, y), value in np.ndenumerate(data):
        plt.text(x, y, f"{value:.2f}", va="center", ha="center", fontsize=18)

    plt.title('Absolute Spearman Rank Correlation Matrix', fontsize=16)
    # Show the plot
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.show()


def process_logs(args) -> DataFrame:
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
            weight_value = float(name.split("_")[2].replace("e", "e-"))
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
                    search_criterion_loss = values[CRITERION_LOSS]
                    search_reg_loss = values[REG_LOSS]
                    row.append(search_criterion_loss)
                    row.append(search_reg_loss)
                    search_acc = values[VALID_ACC]
                    row.append(search_acc)
                    break
            if not match:
                raise Exception(f"Didn't find {name} on eval logs")
        except Exception as e:
            print(f"Error '{e}' while processing file {log.name}")
            while len(row) < 12:
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
            parameters, net_flops, total_time_gpu, total_time_cpu = model_profiling(model, name)
            row.append(parameters)
            row.append(net_flops)
            row.append(total_time_gpu)
            row.append(total_time_cpu)
        except Exception as e:
            print(f"Error '{e}' while processing file {log.name}")

        if len(row) > 0:
            data.append(row)
    df = pd.DataFrame(data, columns=[MODEL_NAME, WEIGHT, "Size",
                                     TRAIN_LOSS, TRAIN_ACC,
                                     VALID_LOSS, VALID_ACC,
                                     TEST_LOSS, TEST_ACC,
                                     SEARCH_CRIT_LOSS, SEARCH_REG_LOSS, SEARCH_ACC,
                                     PARAMETERS, FLOPS,
                                     "Latency GPU", LATENCY_CPU])
    df.set_index(keys=MODEL_NAME, inplace=True)
    df.sort_values(by=WEIGHT, inplace=True, ascending=False)
    pd.set_option("display.max_rows", None, "display.max_columns", None, "display.width", None)
    print(df)
    df.to_csv(args.output)
    return df


def model_profiling(model: NetworkCIFAR, model_name: str) -> tuple:
    parameters = int(count_parameters(model))
    data_shape = (1, 3, 32, 32)
    net_flops = int(count_net_flops(model, data_shape=data_shape))
    total_time_gpu, measured_latency = measure_net_latency(model, l_type='gpu8', fast=False,
                                                           input_shape=data_shape[1:], clean=True)
    print("latency GPU %s: %s" % (model_name, measured_latency))
    total_time_cpu, measured_latency = measure_net_latency(model, l_type='cpu', fast=False,
                                                           input_shape=data_shape[1:], clean=True)
    print("latency CPU %s: %s" % (model_name, measured_latency))
    return parameters, net_flops, total_time_gpu, total_time_cpu


if __name__ == '__main__':
    parser = argparse.ArgumentParser("analyse logs")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-t", "--train", type=argparse.FileType('r'), nargs='+',
                             help="Train/Evaluation stage logs")
    parser.add_argument("-s", "--search", type=argparse.FileType('r'), nargs='+', required=False,
                        help="Search stage logs")
    input_group.add_argument("-d", "--data", type=argparse.FileType('r'), help="Csv table generated from this code")
    parser.add_argument('-c', '--colorbar', action='store_true', default=False,
                        help='Show colorbar instead of precise values')
    parser.add_argument("-i", "--inches", type=float, help="Image size in inches", nargs='+', default=(6.4, 4.8))
    parser.add_argument("-o", "--output", type=str, required=False, help="Output file name", default="loss_table.csv")
    args = parser.parse_args()

    if args.data is None:
        df = process_logs(args)
    else:
        df = pd.read_csv(args.data)

    pd.set_option("display.max_rows", None, "display.max_columns", None, "display.width", None)
    clean_df = df.loc[:, np.invert(df.columns.isin([TRAIN_LOSS, TRAIN_ACC, VALID_LOSS, VALID_ACC]))]
    print(clean_df)
    filename, file_extension = os.path.splitext(args.output)
    plt.gcf().set_size_inches(args.inches, forward=True)
    plot_columns(df, SEARCH_CRIT_LOSS, VALID_LOSS, f"{filename}_search_vs_valid_loss.png",
                 show_weights=not args.colorbar, show_colorbar=args.colorbar, inches=args.inches)
    plot_columns(df, SEARCH_ACC, VALID_ACC, f"{filename}_search_vs_valid_acc.png", 100, show_weights=not args.colorbar,
                 show_colorbar=args.colorbar, inches=args.inches)
    plot_columns(df, WEIGHT, VALID_ACC, f"{filename}_weight_vs_valid_acc.png", y_scale='linear',
                 show_weights=not args.colorbar, show_colorbar=args.colorbar, inches=args.inches)

    clean_df = df.loc[:, np.invert(df.columns.isin([
        TRAIN_LOSS, TRAIN_ACC, VALID_LOSS, TEST_LOSS, SEARCH_REG_LOSS, SEARCH_CRIT_LOSS, PARAMETERS, LATENCY_CPU]))]
    plot_correlation(clean_df, f"{filename}_correlation_matrix.png")
