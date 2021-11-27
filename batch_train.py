import logging
import math
import os
import sys
import time
import pickle

import utils
import train
import genotypes
from genotypes import Genotype
from train import create_parser
from model import NetworkCIFAR as Network, Cell

import nvidia_smi


def calculate_batch_size(batch_model, memory, genotype: Genotype, init_channels: int, classes: int, layers: int, auxiliary: bool) -> int:
    model = Network(init_channels, classes, layers, auxiliary, genotype)
    model_size_mb = utils.count_parameters_in_MB(model)
    batch_size = 0
    consumption = 0
    while consumption < (memory - 500):
        batch_size += 1
        consumption = batch_model.predict([[model_size_mb, batch_size]])
    log.info("Predicted consumption is %d MB", consumption)
    return batch_size


if __name__ == '__main__':
    log = logging.getLogger("batch_train")
    parser = create_parser()
    parser.add_argument('--archs', type=str, required=False, nargs='+', help='list of architectures to use',
                        default=[arch for arch, gen in genotypes.__dict__.items() if isinstance(gen, Genotype)])
    parser.add_argument('--batch_model', type=str, default="batch_predict.pkl", help='list of architectures to use')
    args = parser.parse_args()

    log_path = 'logs/batch-train-%s-%s' % ("EXP" if args.save is None else args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(log_path, scripts_to_save=None)
    log_format = '%(asctime)s %(name)s - %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %H:%M:%S', force=True)
    fh = logging.FileHandler(os.path.join(log_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.getLogger(Cell.__name__).setLevel(logging.WARN)

    CIFAR_CLASSES = 10
    if args.set == 'cifar100':
        CIFAR_CLASSES = 100

    batch_model = pickle.load(open(args.batch_model, 'rb'))

    nvidia_smi.nvmlInit()
    # args.gpu is the device index
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(args.gpu)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    # 1 byte = 2^⁻20 MB
    gpu_memory = math.floor(info.free * math.pow(2, -20))

    trained_gen = {}
    archs = args.archs
    # delete this attribute because it is not necessary to pass it to the child processes
    delattr(args, 'archs')
    for arch in archs:
        genotype = genotypes.__dict__[arch]
        log.info('%s gen = %s', arch, genotype)
        if genotype in trained_gen.values():
            same_arch = list(trained_gen.keys())[list(trained_gen.values()).index(genotype)]
            log.info("Skipping arch %s because its genotype was already trained by %s", arch, same_arch)
        else:
            trained_gen[arch] = genotype
            batch_size = calculate_batch_size(batch_model, gpu_memory, genotype, args.init_channels, CIFAR_CLASSES,
                                              args.layers, args.auxiliary)
            args.arch = arch
            trained = False
            while not trained:
                try:
                    args.batch_size = batch_size
                    train.main(args)
                    trained = True
                except RuntimeError as e:
                    if "out of memory" in str(e) and batch_size > 5:
                        log.error(e)
                        batch_size -= 5
                        log.info(f"Re trying to train with smaller batch size of {batch_size}")
                    else:
                        raise e
    nvidia_smi.nvmlShutdown()
