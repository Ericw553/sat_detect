import logging
import math
import os
import sys
import time
from datetime import datetime
from typing import Tuple

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from dataloaderGEO import MyDataLoader
from torch import nn
from torch.optim import Adam
sys.path.append('../')
from unet import UNet



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


def save_checkpoint(model, save_dir, exp_id, epoch):
    """
    :param model:  nn model
    :param save_dir: save model direction
    :param exp_id: experiment id
    :param epoch:  epoch
    :return:  None
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, exp_id)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    print("save all model to {}".format(save_path))
    output = open(save_path, mode="wb")
    torch.save(model.state_dict(), output)
    output.close()


def diceloss(output, target):
    smooth = 1.
    oflat = output.view(-1)
    tflat = target.view(-1)
    intersection = (oflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / (oflat.sum() + tflat.sum() + smooth))


def iou(pred, target, n_class):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / max(union, 1))
    return ious


def precision_recall_f(pred, target):
    t_p = (np.logical_and(pred == 1, target == 1)).sum()
    f_p = (np.logical_and(pred == 1, target == 0)).sum()
    f_n = (np.logical_and(pred == 0, target == 1)).sum()
    p = t_p / (t_p + f_p)
    r = t_p / (t_p + f_n)
    f = 2 * p * r / (p + r)
    return p, r, f


@click.command()
@click.pass_context
@click.option(
    '--epochs', '-e',
    type=int,
    default=10,
    help='Number of epochs (default: 10)',
)
@click.option(
    '--skip-epochs', '-k',
    type=int,
    default=1,
    help='Number of epochs to skip in result (default: 1)',
)
@click.option(
    '--restart', '-r',
    type=int,
    help='0 is cold start, 1 is a restart',
)
@click.option(
    '--learningrate', '-lr',
    type=float,
    default=0.0001,
    help='Setup learning rate',
)
def cli(ctx: click.Context,
        epochs: int,
        skip_epochs: int,
        restart: int,
        learningrate: float,
        ) -> None:
    if skip_epochs >= epochs:
        ctx.fail('--skip-epochs=%d must be less than --epochs=%d' % (skip_epochs, epochs))

    model: nn.Module = UNet(n_channels=4, n_classes=1)
    """
    f: Experiment = EXPERIMENTS[experiment]
    try:
        model, batch_size, _devices = f(model, devices)
    except ValueError as exc:
        # Examples:
        #   ValueError: too few devices to hold given partitions (devices: 1, paritions: 2)
        ctx.fail(str(exc))
     """
    experiment = "unet"
    lr = learningrate

    print(lr)
    optimizer = Adam(model.parameters(), lr=lr)

    # Prepare dataloaders.
    logging.info("Loading data")

    train_dataloader, valid_dataloader = MyDataLoader(data_dir, 1, 1).dataloader()

    if not restart:
        exp_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f'{experiment}' + "lr" + str(lr)

    click.echo('Saving %s results to %s' % (exp_id, save_dir))

    # TRAIN ======================================================================================

    global BASE_TIME
    BASE_TIME = time.time()

    def run_epoch(epoch: int) -> Tuple[float, float]:
        tick = time.time()

        # training procedure
        data_trained = 0
        model.train()
        BCE_loss_total_t = 0

        for i, (input, target) in enumerate(train_dataloader):

            data_trained += input.size(0)
            output = model(input)

            BCE_loss_logits = F.binary_cross_entropy_with_logits(output, target)
            BCE_loss_logits.backward()

            optimizer.step()
            optimizer.zero_grad()

            if i % 1 == 0:
                log('%d/%d trained | , %.3f BCE_loss'
                    '' % (i, len(train_dataloader), BCE_loss_logits), clear=True)

            BCE_loss_total_t += BCE_loss_logits

            if i >= 40:
                break

        tock = time.time()

        # 00:02:03 | 1/20 epoch | 200.000 samples/sec, 123.456 sec/epoch
        elapsed_time = tock - tick
        throughput = data_trained / elapsed_time
        BCE_loss_average_t = BCE_loss_total_t / data_trained
        log('%d/%d epoch trained | %.3f samples/sec, %.3f sec/epoch, %.3f BCE_loss_average'
            '' % (epoch + 1, epochs, throughput, elapsed_time, BCE_loss_average_t), clear=True)

        # validation procedure
        data_validated = 0
        model.eval()

        total_iou_list = []
        precision_list = []
        recall_list = []
        fscore_list = []
        bce_list = []

        with torch.no_grad():
            for i, (input, target) in enumerate(valid_dataloader):
                data_validated += input.size(0)
                output = model(input)
                BCE_loss_e = F.binary_cross_entropy_with_logits(output, target)

                pred = output.data.cpu().numpy()
                pred = (pred > 0.5).astype('float')  # predicted labels
                gt = target.data.cpu().numpy()  # true labels

                # collect metrics
                i_o_u = iou(pred, gt, 2)
                precision_, recall_, fscore_ = precision_recall_f(pred, gt)

                if not math.isnan(precision_):
                    precision_list.append(precision_)
                if not math.isnan(recall_):
                    recall_list.append(recall_)
                if not math.isnan(fscore_):
                    fscore_list.append(fscore_)
                if not math.isnan(i_o_u[1]):
                    total_iou_list.append(i_o_u[1])
                if not math.isnan(BCE_loss_e):
                    bce_list.append(BCE_loss_e)

                if i % 1 == 0:
                    log('%d/%d validation, %.3f BCE_loss'
                        '' % (i, len(valid_dataloader), BCE_loss_e), clear=True)
                    log('%d/%d validation, %.3f IoU'
                        '' % (i, len(valid_dataloader), i_o_u[1]), clear=True)
                    log('%d/%d validation, %.3f Precision'
                        '' % (i, len(valid_dataloader), precision_), clear=True)
                    log('%d/%d validation, %.3f Recall'
                        '' % (i, len(valid_dataloader), recall_), clear=True)
                    log('%d/%d validation, %.3f F1-score'
                        '' % (i, len(valid_dataloader), fscore_), clear=True)
                if i > 40:
                    break

            avg_iou = np.array(total_iou_list).sum() / len(total_iou_list)
            avg_precision = np.array(precision_list).sum() / len(precision_list)
            avg_recall = np.array(recall_list).sum() / len(recall_list)
            avg_F1 = np.array(fscore_list).sum() / len(fscore_list)
            avg_BCE = np.array(bce_list).sum() / len(bce_list)

            log('%d/%d epoch validated | %.3f IoU'
                '' % (epoch + 1, epochs, avg_iou), clear=True)
            log('%d/%d epoch validated | %.3f Precision'
                '' % (epoch + 1, epochs, avg_precision), clear=True)
            log('%d/%d epoch validated | %.3f Recall'
                '' % (epoch + 1, epochs, avg_recall), clear=True)
            log('%d/%d epoch validated | %.3f F1-score'
                '' % (epoch + 1, epochs, avg_F1), clear=True)
            log('%d/%d epoch validated | %.3f BCE_loss_averaged'
                '' % (epoch + 1, epochs, avg_BCE), clear=True)

        return throughput, elapsed_time, BCE_loss_average_t, avg_BCE

    throughputs = []
    elapsed_times = []
    hist_BCE_t = []
    hist_BCE_e = []
    hr()

    for epoch in range(epochs):
        throughput, elapsed_time, BCE_loss_t, BCE_loss_e = run_epoch(epoch)

        if epoch < skip_epochs:
            continue

        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)
        hist_BCE_t.append(BCE_loss_t)
        hist_BCE_e.append(BCE_loss_e)

        if not epoch % 5: save_checkpoint(model, save_dir, exp_id, epoch)

    hr()

    # RESULT ======================================================================================

    result = np.array([throughputs, elapsed_times, hist_BCE_t, hist_BCE_e])
    result = result.T

    print(result)

    final_model = save_dir + 'fmodel_' + exp_id + '.pt'
    torch.save(model.state_dict(), final_model)


if __name__ == '__main__':
    cli()
