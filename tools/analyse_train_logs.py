import argparse
import os
import re

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

from ofa.utils import count_parameters, count_net_flops, measure_net_latency
import matplotlib.pyplot as plt
from pandas import DataFrame

import genotypes
from genotypes import Genotype
from model import NetworkCIFAR
from multiobjective import get_cmap, round_number
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
PARAMETERS_DARTS = "Params"
PARAMETERS_OFA = "Parameters"
LATENCY_CPU = "Latency CPU"
LATENCY_GPU = "Latency GPU"

name = {
    TRAIN_LOSS: "Train loss",
    TRAIN_ACC: "Train acc",
    VALID_LOSS: "Valid loss",
    VALID_ACC: "Valid acc",
    TEST_LOSS: "Test loss",
    TEST_ACC: "Test acc",
}


def plot_columns(df: DataFrame, x_column: str, y_column: str, filename="plot_table.png", negate: int = None,
                 x_scale='log', y_scale='log', show_weights: bool = True, show_colorbar=False,
                 inches: tuple = (6.4, 4.8), x_label: str = None, y_label: str = None, topk: int = 0):
    """
    Plot DataFrame columns

    :param df:
    :param x_column: Column name to use as x-axis
    :param x_label: Display name for x-axis
    :param y_column: Column name to use as y-axis
    :param y_label: Display name for y-axis
    :param filename: Save path
    :param negate: Negate the values and add it to some offset passed as int here
    :param x_scale: {"linear", "log", "symlog", "logit", ...} or `.ScaleBase`
    :param y_scale: {"linear", "log", "symlog", "logit", ...} or `.ScaleBase`
    :param show_weights: Show detailed weight values instead of colorbar
    :param inches: Figure dimension width x height tuple in inches
    :param topk: tok k items to use X marker instead of circle
    """
    if x_column not in df or y_column not in df:
        print(f"Can't plot {x_column} vs {y_column} because one or both is missing on the data table")
        return

    plt.clf()
    x_axis = df.loc[:, x_column]
    y_axis = df.loc[:, y_column]
    if isinstance(negate, (int, float)):
        x_axis = negate - x_axis.to_numpy()
        y_axis = negate - y_axis.to_numpy()
    weights = df.loc[:, WEIGHT].to_list()

    # Plot the data
    cmap = get_cmap(len(y_axis))
    plt.gcf().set_size_inches(inches, forward=True)
    if show_weights:
        data = zip(x_axis, y_axis, weights)
        data = sorted(data, key=lambda tup: tup[2], reverse=True)
        for i, (x, y, w) in enumerate(data):
            exponent, round_n = round_number(weights, w)
            mantissa = round(round_n * pow(10, exponent))
            if len(data) - 1 - i < topk:
                marker = "x"
            else:
                marker = "o"
            plt.scatter(x, y, color=cmap(i), marker=marker, label="%de-%02d" % (mantissa, exponent))
        # Add a legend
        plt.legend(title='$\\nu$ value')

    else:
        plt.scatter(x_axis, y_axis, c=range(len(y_axis)), cmap=cmap)
        if show_colorbar:
            cbar = plt.colorbar(ScalarMappable(norm=LogNorm(vmin=min(weights), vmax=max(weights)), cmap=cmap.reversed()))
            cbar.set_label("$\\nu$", rotation=0)

    if not x_label:
        x_label = name.get(x_column, x_column)
    plt.xlabel(x_label)
    if not y_label:
        y_label = name.get(y_column, y_column)
    plt.ylabel(y_label)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.title(f"{x_label} vs {y_label} per Weight ($\\nu$)".replace(" (%)", ""))

    # Show the plot
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.show()


def plot_correlation(df: DataFrame, filename="correlation_matrix.png"):
    plt.clf()
    f = plt.figure(figsize=(11, 10))
    data = np.abs(df.corr(method="spearman").to_numpy())
    plt.matshow(data, fignum=f.number)
    ax = plt.gca()

    # legend
    columns = [name.get(column, column) for column in df.select_dtypes(['number']).columns]
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), columns, fontsize=14, rotation=45, ha="left",
               rotation_mode="anchor")
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), columns, fontsize=14)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), columns, fontsize=14, rotation=45, ha="left",
               rotation_mode="anchor")
    if any(c in df for c in [TEST_ACC, VALID_ACC, TEST_LOSS, VALID_LOSS]):
        ax.axvline(4.5, 0.0, 1, linewidth=4, color="red", linestyle="solid")
        # Now let's add your additional information
        ax.annotate('Search stage data',
                    xy=(190, -20), xytext=(0, 0),
                    xycoords=('axes points', 'axes points'),
                    textcoords='offset points',
                    size=14, ha='center', va='bottom')
        ax.annotate('Evaluation stage data',
                    xy=(420, -20), xytext=(0, 0),
                    xycoords=('axes points', 'axes points'),
                    textcoords='offset points',
                    size=14, ha='center', va='bottom')


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
        lines = str(log.readlines())
        # if train is the batch train logs with all logs
        matches = list(re.finditer(r"arch='(?P<name>.*?)'", lines))
        for i, match in enumerate(matches):
            arch = match.group("name")
            row = [arch]
            if i == 0:
                start_lines = 0
            else:
                start_lines = match.start()
            if i >= len(matches) - 1:
                end_lines = len(lines)
            else:
                end_lines = matches[i+1].start() - 1
            local_lines = lines[start_lines:end_lines]
            try:
                # evaluation stage metrics
                # l2_loss_2e01 -> 2e-01
                weight_value = float(arch.split("_")[2].replace("e", "e-"))
                row.append(weight_value)
                match = re.search(r"param size.*?(?P<value>\d*\.\d+)MB", local_lines)
                param_size = float(match.group("value"))
                row.append(param_size)
                for metric in [TRAIN_LOSS, TRAIN_ACC, VALID_LOSS, VALID_ACC, TEST_LOSS, TEST_ACC]:
                    metric_all = re.findall(rf'{metric}(?:uracy)? (?P<value>\d*\.\d+)', local_lines)
                    value = None
                    if metric_all:
                        value = float(metric_all[-1])
                    row.append(value)
            except Exception as e:
                print(f"Error '{e}' while processing file {log.name}")
                while len(row) < 9:
                    row.append(None)

            try:
                # search stage metrics
                genotype = genotypes.__dict__[arch]
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
                    raise Exception(f"Didn't find {arch} on eval logs")
            except Exception as e:
                print(f"Error '{e}' while processing file {log.name}")
                while len(row) < 12:
                    row.append(None)

            try:
                # model profiling
                genotype = genotypes.__dict__[arch]
                match = re.search(r"init_channels=(?P<value>\d+)", local_lines)
                init_channels = int(match.group("value"))
                match = re.search(r"layers=(?P<value>\d+)", local_lines)
                layers = int(match.group("value"))
                match = re.search(r"drop_path_prob=(?P<value>\d+\.\d+)", local_lines)
                drop_path_prob = float(match.group("value"))
                match = re.search(r"auxiliary=(?P<value>\w+)", local_lines)
                auxiliary = bool(match.group("value"))
                model = NetworkCIFAR(init_channels, 10, layers, auxiliary, genotype)
                model.cuda()
                model.drop_path_prob = drop_path_prob
                parameters, net_flops, total_time_gpu, total_time_cpu = model_profiling(model, arch)
                row.append(parameters)
                row.append(net_flops)
                row.append(total_time_gpu)
                row.append(total_time_cpu)
            except Exception as e:
                print(f"Error '{e}' while processing file {log.name}")

            if len(row) > 0:
                data.append(row)
    df = pd.DataFrame(data, columns=[MODEL_NAME, WEIGHT, PARAMETERS_DARTS,
                                     TRAIN_LOSS, TRAIN_ACC,
                                     VALID_LOSS, VALID_ACC,
                                     TEST_LOSS, TEST_ACC,
                                     SEARCH_CRIT_LOSS, SEARCH_REG_LOSS, SEARCH_ACC,
                                     PARAMETERS_OFA, FLOPS,
                                     LATENCY_GPU, LATENCY_CPU])
    df.set_index(keys=MODEL_NAME, inplace=True)
    df.sort_values(by=WEIGHT, inplace=True, ascending=False)
    # remove None/null columns because missing on logs or due to errors
    df.dropna(axis=1, how='all', inplace=True)
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
                 show_colorbar=args.colorbar, inches=args.inches, x_label="Search error (%)", y_label="Valid error (%)")
    plot_columns(df, WEIGHT, VALID_ACC, f"{filename}_weight_vs_valid_acc.png", y_scale='linear',
                 show_weights=not args.colorbar, show_colorbar=args.colorbar, inches=args.inches,
                 y_label="Valid acc (%)")
    plot_columns(df, WEIGHT, TEST_ACC, f"{filename}_weight_vs_test_acc.png", y_scale='linear',
                 show_weights=not args.colorbar, show_colorbar=args.colorbar, inches=args.inches,
                 y_label="Test Acc (%)")

    clean_df = df.loc[:, np.invert(df.columns.isin([
        TRAIN_LOSS, TRAIN_ACC, VALID_LOSS, TEST_LOSS, SEARCH_REG_LOSS, SEARCH_CRIT_LOSS, PARAMETERS_OFA, LATENCY_CPU]))]
    clean_df = clean_df.filter([WEIGHT, PARAMETERS_DARTS, FLOPS, LATENCY_GPU, SEARCH_ACC, VALID_ACC, TEST_ACC])
    plot_correlation(clean_df, f"{filename}_correlation_matrix.png")
