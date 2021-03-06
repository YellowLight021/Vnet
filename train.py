# !/usr/bin/env python3
import os

import paddle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from paddle.distributed import fleet
import time
import argparse
import numpy as np
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.vision.transforms as transforms
from paddle.io import DataLoader
from luna import LUNA16
import utils

import os
import sys
import math

import shutil

import setproctitle

import vnet
from functools import reduce
import operator
from visualdl import LogWriter




nodule_masks = None

lung_masks = 'labels'

ct_images = 'imgs'
ct_targets = lung_masks

target_split = [2, 2, 2]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, is_best, path, prefix, epoch=0, filename='checkpoint.pth.tar'):

    paddle.save(state, os.path.join(path, 'checkpoint_model_new.pth.tar'))
    if is_best:
        paddle.save(state, os.path.join(path, 'checkpoint_model_best.pth.tar'))



def inference(args, loader, model, transforms):
    src = args.inference
    dst = args.save

    model.eval()
    nvols = reduce(operator.mul, target_split, 1)
    # assume single GPU / batch size 1
    for data in loader:
        data, series, origin, spacing = data[0]
        shape = data.size()
        # convert names to batch tensor

        output = model(data)
        _, output = output.max(1)
        output = output.view(shape)
        output = output.cpu()
        # merge subvolumes and save
        results = output.chunk(nvols)
        results = map(lambda var: paddle.squeeze(var.data).numpy().astype(np.int16), results)
        volume = utils.merge_image([*results], target_split)
        print("save {}".format(series))
        utils.save_updated_image(volume, os.path.join(dst, series + ".mhd"), origin, spacing)


# performing post-train inference:
# train.py --resume <model checkpoint> --i <input directory (*.mhd)> --save <output directory>

def noop(x):
    return x


def main():
    # os.environ['KMP_DUPLICATE_LIB_OK'] = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=2)
    parser.add_argument('--dice', action='store_true')
    parser.add_argument('--ngpu', type=int, default=4)
    parser.add_argument('--nEpochs', type=int, default=30)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', default=True, action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-i', '--inference', default='', type=str, metavar='PATH',
                        help='run inference on data set and save results')

    # 1e-8 works well for lung masks but seems to prevent
    # rapid learning for nodule masks
    parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 1e-8)')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save', type=str, default='myoutput')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='momentum',
                        choices=('momentum', 'adam', 'rmsprop'))
    args = parser.parse_args()
    best_prec1 = 100.
    best_epoch = 0
    err_best = 100.
    best_dice = 0

    # args.cuda=False
    # args.save = args.save or 'work/vnet.base.{}'.format(datestr())
    # ?????????????????????
    def reader_decorator(reader):
        def __reader__():
            for item in reader():
                img = np.array(item[0]).astype('float32').reshape(3, 224, 224)
                label = np.array(item[1]).astype('int64').reshape(1)
                yield img, label

        return __reader__

    weight_decay = args.weight_decay
    setproctitle.setproctitle(args.save)
    strategy = fleet.DistributedStrategy()
    fleet.init(is_collective=True, strategy=strategy)

    # paddle.manual_seed(args.seed)
    paddle.seed(args.seed)

    print("build vnet")

    model = vnet.VNet()

    model = paddle.distributed.fleet.distributed_model(model)
    batch_size = args.ngpu * args.batchSz
    # gpu_ids = range(args.ngpu)
    # model = nn.parallel.DataParallel(model, device_ids=gpu_ids)

    if args.opt == 'momentum':
        optimizer = optim.Momentum(learning_rate=0.001, momentum=0.99, parameters=model.parameters(),
                                   weight_decay=weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(learning_rate=0.001, parameters=model.parameters(), weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), weight_decay=weight_decay)

    optimizer = fleet.distributed_optimizer(optimizer)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = paddle.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            model.set_state_dict(checkpoint['state_dict'])
            optimizer.set_state_dict(checkpoint['optimizer_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)

    train = train_dice
    evale = eval_dice
    class_balance = False

    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)

    # LUNA16 dataset isotropically scaled to 2.5mm^3
    # and then truncated or zero-padded to 160x128x160
    normMu = [-300]
    normSigma = [700]
    normTransform = transforms.Normalize(normMu, normSigma)

    trainTransform = transforms.Compose([
        # transforms.ToTensor(),
        normTransform
    ])
    evalTransform = transforms.Compose([
        # transforms.ToTensor(),
        normTransform
    ])
    if ct_targets == nodule_masks:
        masks = lung_masks
    else:
        masks = None
    # import pdb
    # pdb.set_trace()
    if args.inference != '':
        if not args.resume:
            print("args.resume must be set to do inference")
            exit(1)
        kwargs = {'num_workers': 1}
        src = args.inference
        dst = args.save
        inference_batch_size = args.ngpu
        root = os.path.dirname(src)
        images = os.path.basename(src)
        dataset = LUNA16(root=root, images=images, transform=evalTransform, split=target_split, mode="infer")
        loader = DataLoader(dataset, batch_size=inference_batch_size, shuffle=False, collate_fn=noop, **kwargs)
        inference(args, loader, model, trainTransform)
        return

    kwargs = {}
    print("loading training set")

    trainSet = LUNA16(root=r'./', images=ct_images, targets=ct_targets,
                      mode="train", transform=trainTransform,
                      class_balance=class_balance, split=target_split, seed=args.seed, masks=masks)

    train_sampler = paddle.io.DistributedBatchSampler(trainSet, batch_size=batch_size, shuffle=True, drop_last=True)
    trainLoader = DataLoader(trainSet, batch_sampler=train_sampler, **kwargs)

    print("loading eval set")

    evalSet = LUNA16(root=r'./', images=ct_images, targets=ct_targets,
                     mode="eval", transform=evalTransform, seed=args.seed, masks=masks, split=target_split)
    eval_sampler = paddle.io.DistributedBatchSampler(evalSet, batch_size=1, shuffle=False)
    evalLoader = DataLoader(evalSet, batch_sampler=eval_sampler, **kwargs)


    class_weights = []

    for epoch in range(args.start_epoch + 1, args.nEpochs + 1):

        start_time = time.time()
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, model, trainLoader, optimizer, class_weights)
        print('start test nEpochs:{}'.format(epoch))

        dice = evale(args, epoch, model, evalLoader, optimizer, class_weights)
        print('cost time is {}'.format(time.time() - start_time))
        is_best = False
        # import pdb
        # pdb.set_trace()
        if best_dice < dice:
            best_epoch = epoch
            is_best = True
            best_dice = dice
        print('best_epoch is {},best_dice :{:.8f}%'.format(best_epoch, best_dice))
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'dice': dice,
                         'optimizer_dict': optimizer.state_dict()},
                        is_best, args.save, "vnet", epoch)



def train_dice(args, epoch, model, trainLoader, optimizer, weights):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):

        if ((data < -1).sum().item() + (data > 1).sum().item()) > 0:
            continue
        optimizer.clear_grad()
        output = model(data)
        loss = utils.dice_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)

        nProcessed += len(data)
        Dice_coefficient = 100. * (1. - loss.item())
        error_rate = (1 - paddle.metric.accuracy(output.transpose([0, 2, 3, 4, 1]).reshape([-1, 2]),
                                                 target.reshape([-1, 1]), k=1).item()) * 100.
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1

        print(
            'Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}\tDice_coefficient:: {:.8f}%\tErrorRate:{}%  time:{}'.format(
                partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
                loss.item(), Dice_coefficient, error_rate, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        if paddle.isnan(loss):
            print('data:{}'.format(data))
            print('model.state_dict{}'.format(model.state_dict()))
            print('output:{}'.format(output))

        del Dice_coefficient
        del error_rate
        loss.backward()
        optimizer.step()



def eval_dice(args, epoch, model, evalLoader, optimizer, weights):
    model.eval()
    eval_loss = 0
    incorrect = 0
    Dice_coefficient = 0
    error_rate = 0
    count = 0
    for data, target in evalLoader:
        # import pdb
        # pdb.set_trace()
        output = model(data)
        # import pdb
        # pdb.set_trace()
        loss = utils.dice_loss(output, target).item()

        eval_loss += loss
        Dice_coefficient += (1. - loss)

        count += 1
        error_rate += (1 - paddle.metric.accuracy(
            output[:, :, :target.shape[1], :].transpose([0, 2, 3, 4, 1]).reshape([-1, 2]),
            target.reshape([-1, 1]), k=1).item())
        # print("this dice is {}".format(1. - loss,))
    # import pdb
    # pdb.set_trace()
    eval_loss /= count  # loss function already averages over batch size
    nTotal = len(evalLoader)
    Dice_coefficient = 100. * Dice_coefficient / count
    error_rate = 100. * error_rate / count

    print('\nEval set: Average eval_loss: {:.4f}, Dice_coefficient: {}/{} ({:.0f}%,ErrorRate:{}%)\n'.format(
        eval_loss, incorrect, nTotal, Dice_coefficient, error_rate))

    return Dice_coefficient


def adjust_opt(optAlg, optimizer, epoch):
    # if optAlg == 'momentum':
    epo = epoch % 10
    if epoch <= 10:
        lr = 1e-3
    elif epoch <= 50:
        if epo == 2:
            lr = 1e-4
        elif epo == 4:
            lr = 1e-5
        elif epo == 6:
            lr = 1e-6
        elif epo == 8:
            lr = 1e-7
        elif epo == 0:
            lr = 1e-8
        else:
            return
    else:
        lr = (1e-6) * (1 - 0.15) ** (epoch - 50)
    optimizer.set_lr(lr)


if __name__ == '__main__':
    main()
