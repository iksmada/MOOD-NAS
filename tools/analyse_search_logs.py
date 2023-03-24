import argparse
import os
import re

import pandas as pd

from pandas import DataFrame

from genotypes import Genotype
from model import NetworkCIFAR
from multiobjective import create_genotype_name
from tools.analyse_train_logs import plot_columns, plot_correlation, model_profiling, LATENCY_CPU, WEIGHT, \
    PARAMETERS_DARTS
from tools.plot_frontier import filter_hist
from train_search import L1_LOSS, L2_LOSS, TRAIN_ACC, VALID_ACC, CRITERION_LOSS, REG_LOSS, SIZE, GENOTYPE

MODEL_NAME = "Model name"

SEARCH_CRIT_LOSS = "Search crit loss"
SEARCH_REG_LOSS = "Search reg loss"
SEARCH_VAL_ACC = "Search valid acc"
SEARCH_TRAIN_ACC = "Search train acc"
FLOPS = "FLOPs"


def process_logs(args) -> DataFrame:
    # filter logs
    lines = str(args.log.readlines())
    match = re.search(r"Selected regularization(?P<reg>.*?)\\n", lines)
    reg = match.group("reg")
    if L1_LOSS in reg:
        loss = L1_LOSS
    elif L2_LOSS in reg:
        loss = L2_LOSS
    else:
        raise RuntimeError("Cant decode line Selected regularization")
    match = re.finditer(r"hist = (?P<hist>.*?)\\n", lines)
    hist_str = list(match)[-1].group("hist")
    hist = eval(hist_str)[loss]
    print("Removing non-optimal samples")
    # filter out dominated points
    filter_hist(hist)
    data = []
    all_weights = [weight[0] for weight in hist.keys()]
    for weight, result in hist.items():
        row = []
        name = create_genotype_name(weight, loss, all_weights)
        try:
            row.append(name)
            weight_value = weight[0]
            row.append(weight_value)
            # {'train_acc': 25.035999994506835,
            # 'valid_acc': 20.171999999084473,
            # 'reg_loss': 16.01249122619629,
            # 'criterion_loss': 1.9922981262207031,
            # 'model_size': 1.81423,
            # 'genotype': Genotype(..)}
            for metric in [SIZE, TRAIN_ACC, VALID_ACC, CRITERION_LOSS, REG_LOSS]:
                row.append(result[metric])
        except Exception as e:
            print(f"Error '{e}' while processing file {args.log} w={weight}")
            while len(row) < 7:
                row.append(None)

        try:
            # model profiling
            genotype = result[GENOTYPE]
            # using default from train.py for CIFAR10
            model = NetworkCIFAR(36, 10, 20, False, genotype)
            model.cuda()
            model.drop_path_prob = 0.3
            parameters, net_flops, total_time_gpu, total_time_cpu = model_profiling(model, name)
            row.append(parameters)
            row.append(net_flops)
            row.append(total_time_gpu)
            row.append(total_time_cpu)
        except Exception as e:
            print(f"Error '{e}' while processing file {args.log} w={weight}")
            raise e

        if len(row) > 0:
            data.append(row)
    df = pd.DataFrame(data, columns=[MODEL_NAME, WEIGHT, "Params",
                                     SEARCH_TRAIN_ACC, SEARCH_VAL_ACC,
                                     SEARCH_CRIT_LOSS, SEARCH_REG_LOSS,
                                     "Parameters", FLOPS,
                                     "Latency GPU", LATENCY_CPU])
    df.set_index(keys=MODEL_NAME, inplace=True)
    df.sort_values(by=WEIGHT, inplace=True, ascending=False)
    pd.set_option("display.max_rows", None, "display.max_columns", None, "display.width", None)
    print(df)
    df.to_csv(args.output)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser("analyse logs")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-l", "--log", type=argparse.FileType('r'), help="Multi search stage log")
    input_group.add_argument("-d", "--data", type=argparse.FileType('r'), help="Csv table generated from this code")
    parser.add_argument("-o", "--output", type=str, required=False, help="Output file name", default="search_table.csv")
    parser.add_argument("-i", "--inches", type=float, help="Image size in inches", nargs='+', default=[6.4, 4.8])
    parser.add_argument("-t", "--topk", type=float, help="Top number of weights to highlight", nargs='+', default=0)
    args = parser.parse_args()

    if args.data is None:
        df = process_logs(args)
    else:
        df = pd.read_csv(args.data)
        pd.set_option("display.max_rows", None, "display.max_columns", None, "display.width", None)
        print(df)

    filename, file_extension = os.path.splitext(args.output)
    plot_columns(df, SEARCH_REG_LOSS, SEARCH_CRIT_LOSS, f"{filename}_crit_vs_reg_loss.png", x_scale='linear',
                 y_scale='linear', inches=args.inches,
                 x_label='Regularization loss',
                 y_label='Cross Entropy loss',
                 topk=5)
    plot_columns(df, WEIGHT, SEARCH_VAL_ACC, f"{filename}_weight_vs_valid_acc.png", y_scale='linear',
                 inches=args.inches, topk=args.topk)
    plot_columns(df, WEIGHT, PARAMETERS_DARTS, f"{filename}_weight_vs_params.png", y_scale='linear',
                 inches=args.inches, topk=args.topk)

    plot_correlation(df, f"{filename}_correlation_matrix.png")




