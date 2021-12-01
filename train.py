# !/usr/bin/env python3
import os

import paddle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import argparse
import numpy as np
# import torch.nn as nn
import paddle.nn as nn
# import torch.nn.init as init
# import torch.optim as optim
import paddle.optimizer as optim
# from paddleseg.utils import logger
# import torch.nn.functional as F
# from torch.autograd import Variable
import paddle.nn.functional as F
import paddle.vision.transforms as transforms
from paddle.io import DataLoader
# from torch.utils.data import DataLoader
from luna import LUNA16
# import torchbiomed.transforms as biotransforms
# import torchbiomed.loss as bioloss
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

# nodule_masks = "normalized_mask_5_0"
# lung_masks = "normalized_seg_lungs_5_0"
# ct_images = "normalized_CT_5_0"
# target_split = [1, 1, 1]
# ct_targets = nodule_masks


nodule_masks = None
# lung_masks = "inferred_seg_lungs_2_5"
lung_masks = 'seg-lungs-LUNA16'
# ct_images = "luna16_ct_normalized"
ct_images = 'imgs'
ct_targets = lung_masks
# 这个参数弃用，因为目前数据集的尺寸是不一样大的
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
    # import pdb
    # pdb.set_trace()
    prefix_save = os.path.join(path, "epoch" + str(epoch))
    name = os.path.join(prefix_save, filename)
    paddle.save(state, name)
    if is_best:
        shutil.copyfile(name, os.path.join(path, 'checkpoint_model_best.pth.tar'))


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
    parser.add_argument('--batchSz', type=int, default=8)
    parser.add_argument('--dice', action='store_true')
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--nEpochs', type=int, default=20)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-i', '--inference', default='', type=str, metavar='PATH',
                        help='run inference on data set and save results')

    # 1e-8 works well for lung masks but seems to prevent
    # rapid learning for nodule masks
    parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
                        metavar='W', help='weight decay (default: 1e-8)')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save', type=str, default='output')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='momentum',
                        choices=('momentum', 'adam', 'rmsprop'))
    args = parser.parse_args()
    best_prec1 = 100.

    # args.cuda=False
    args.save = args.save or 'work/vnet.base.{}'.format(datestr())

    weight_decay = args.weight_decay
    setproctitle.setproctitle(args.save)

    # paddle.manual_seed(args.seed)
    paddle.seed(args.seed)

    print("build vnet")
    model = vnet.VNet()
    batch_size = args.ngpu * args.batchSz
    # gpu_ids = range(args.ngpu)
    # model = nn.parallel.DataParallel(model, device_ids=gpu_ids)

    if args.opt == 'momentum':
        optimizer = optim.Momentum(learning_rate=0.0001, momentum=0.99, parameters=model.parameters(),
                                   weight_decay=weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), weight_decay=weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = paddle.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.set_state_dict(checkpoint['state_dict'])
            optimizer.set_state_dict(checkpoint['optimizer_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)

    train = train_dice
    test = test_dice
    class_balance = False

    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)

    # LUNA16 dataset isotropically scaled to 2.5mm^3
    # and then truncated or zero-padded to 160x128x160
    normMu = [-300]
    normSigma = [600]
    normTransform = transforms.Normalize(normMu, normSigma)

    trainTransform = transforms.Compose([
        # transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
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
        dataset = LUNA16(root=root, images=images, transform=testTransform, split=target_split, mode="infer")
        loader = DataLoader(dataset, batch_size=inference_batch_size, shuffle=False, collate_fn=noop, **kwargs)
        inference(args, loader, model, trainTransform)
        return

    kwargs = {'num_workers': 1}
    print("loading training set")
    # import pdb
    # pdb.set_trace()
    # trainSet = dset.LUNA16(root='luna16', images=ct_images, targets=ct_targets,
    #                        mode="train", transform=trainTransform,
    #                        class_balance=class_balance, split=target_split, seed=args.seed, masks=masks)
    trainSet = LUNA16(root=r'/home/aistudio/work', images=ct_images, targets=ct_targets,
                      mode="train", transform=trainTransform,
                      class_balance=class_balance, split=target_split, seed=args.seed, masks=masks)
    # import pdb
    # pdb.set_trace()
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, **kwargs)

    print("loading test set")
    testLoader = DataLoader(
        LUNA16(root=r'/home/aistudio/work', images=ct_images, targets=ct_targets,
               mode="test", transform=testTransform, seed=args.seed, masks=masks, split=target_split),
        batch_size=batch_size, shuffle=True, **kwargs)

    target_mean = trainSet.target_mean()
    bg_weight = target_mean / (1. + target_mean)
    fg_weight = 1. - bg_weight
    # print(bg_weight)
    # class_weights = torch.FloatTensor([bg_weight, fg_weight])
    class_weights = paddle.to_tensor([bg_weight, fg_weight])

    # import pdb
    # pdb.set_trace()

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')
    best_epoch = 0
    err_best = 100.
    for epoch in range(args.start_epoch + 1, args.nEpochs + 1):
        log_writer = LogWriter(os.path.join(args.save, "epoch" + str(epoch)))
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, model, trainLoader, optimizer, trainF, class_weights, log_writer)
        err = test(args, epoch, model, testLoader, optimizer, testF, class_weights, log_writer)
        is_best = False
        # import pdb
        # pdb.set_trace()
        if err < best_prec1:
            best_epoch = epoch
            is_best = True
            best_prec1 = err
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1,
                         'optimizer_dict': optimizer.state_dict()},
                        is_best, args.save, "vnet", epoch)
        # os.system('./plot.py {} {} &'.format(len(trainLoader), args.save))
        print('best_epoch is {},best_err is {}'.format(best_epoch, err))
        log_writer.close()

    trainF.close()
    testF.close()


def train_dice(args, epoch, model, trainLoader, optimizer, trainF, weights, log_writer):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        optimizer.clear_grad()
        output = model(data)
        loss = utils.my_dice_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        Dice_coefficient = 100. * (1. - loss.item())
        error_rate = 100. * utils.error_rate(output, target)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}\tDice_coefficient:: {:.8f}%\tErrorRate:{}%'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.item(), Dice_coefficient, error_rate))
        log_writer.add_text('TrainInfo',
                            'Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}\tDice_coefficient:: {:.8f}%\tErrorRate:{}%'.format(
                                partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
                                loss.item(), Dice_coefficient, error_rate))

        trainF.write('{},{},{},{}\n'.format(partialEpoch, loss.item(), Dice_coefficient, error_rate))
        trainF.flush()


def test_dice(args, epoch, model, testLoader, optimizer, testF, weights, log_writer):
    model.eval()
    test_loss = 0
    incorrect = 0
    Dice_coefficient = 0
    error_rate = 0
    for data, target in testLoader:
        output = model(data)
        # import pdb
        # pdb.set_trace()
        loss = utils.my_dice_loss(output, target).item()

        test_loss += loss
        Dice_coefficient += (1. - loss)
        error_rate += utils.error_rate(output, target)

    test_loss /= len(testLoader)  # loss function already averages over batch size
    nTotal = len(testLoader)
    Dice_coefficient = 100. * Dice_coefficient / nTotal
    error_rate = 100. * error_rate / nTotal
    print('\nTest set: Average test_loss: {:.4f}, Dice_coefficient: {}/{} ({:.0f}%),ErrorRate:{}%\n'.format(
        test_loss, incorrect, nTotal, Dice_coefficient, error_rate))
    log_writer.add_text('TestInfo',
                        '\nTest set: Average test_loss: {:.4f}, Dice_coefficient: {}/{} ({:.0f}%),ErrorRate:{}%\n'.format(
                            test_loss, incorrect, nTotal, Dice_coefficient, error_rate))
    testF.write('{},{},{},{}\n'.format(epoch, test_loss, Dice_coefficient, error_rate))
    testF.flush()
    return error_rate


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'momentum':
        if epoch < 5:
            lr = 1e-3
        elif epoch == 5:
            lr = 1e-4
        elif epoch == 10:
            lr = 1e-5
        elif epoch == 15:
            lr = 1e-6
        else:
            return
        optimizer.set_lr(lr)


if __name__ == '__main__':
    main()
