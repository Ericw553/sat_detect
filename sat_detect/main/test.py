from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
import click
import torch
import numpy as np
import time, os
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
import torch.utils.data
import sys
sys.path.append('../')
from unet import UNet
import math
from dataloaderGEO import MyDataLoader
import logging

logging.basicConfig(level=logging.DEBUG)

data_dir = '../data'

save_dir = '/Users/ericwang/saved_model/'

BASE_TIME: float = 0


def hr() -> None:
    """Prints a horizontal line."""
    width, _ = click.get_terminal_size()
    click.echo('-' * width)


def log(msg: str, clear: bool = False, nl: bool = True) -> None:
    """Prints a message with elapsed time."""
    if clear:
        # Clear the output line to overwrite.
        width, _ = click.get_terminal_size()
        click.echo('\b\r', nl=False)
        click.echo(' ' * width, nl=False)
        click.echo('\b\r', nl=False)

    t = time.time() - BASE_TIME
    h = t // 3600
    t %= 3600
    m = t // 60
    t %= 60
    s = t

    click.echo('%02d:%02d:%02d | ' % (h, m, s), nl=False)
    click.echo(msg, nl=nl)

@click.command()
@click.pass_context
@click.option(
    '--epochs', '-e',
    type=int,
    default=1,
    help='Number of epochs (default: 10)',
)
@click.option(
    '--restart', '-r',
    type=int,
    help='0 is cold start, 1 is a restart',
)
def cli(ctx: click.Context,
        epochs: int,
        restart: int
        ) -> None:
    model: nn.Module = UNet(n_channels=4, n_classes=1)
    model.load_state_dict(torch.load("../saved_model/fmodel_20201109-094209unetlr0.0001.pt"))
    experiment = "unet"
    logging.info("Loading data")

    test_dataloader = MyDataLoader(data_dir, 1, 1).dataloader_test()

    # TRAIN ======================================================================================

    global BASE_TIME
    BASE_TIME = time.time()

    def run_epoch(epoch: int) -> Tuple[float, float]:
        tick = time.time()

        # testing procedure
        data_tested = 0
        model.eval()

        for i, (input, target) in enumerate(test_dataloader):

            data_tested += input.size(0)
            output = model(input)

            str = '{}.npy'.format(i)
            np.save(str, output.detach())

            if i % 1 == 0:
                log('%d/%d tested'
                    '' % (i, len(test_dataloader)), clear=True)

        tock = time.time()

        # 00:02:03 | 1/20 epoch | 200.000 samples/sec, 123.456 sec/epoch
        elapsed_time = tock - tick
        throughput = data_tested / elapsed_time
        log('%d/%d epoch tested | %.3f samples/sec, %.3f sec/epoch'
            '' % (epoch + 1, epochs, throughput, elapsed_time), clear=True)
        return throughput, elapsed_time

    throughputs = []
    elapsed_times = []
    hist_BCE_t = []
    hist_BCE_e = []
    hr()

    for epoch in range(epochs):
        throughput, elapsed_time = run_epoch(epoch)

        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)

    hr()

    # RESULT ======================================================================================

    result = np.array([throughputs, elapsed_times])
    result = result.T

    print(result)

if __name__ == '__main__':
    cli()
