# !/usr/bin/env python3
import os

import paddle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from paddle.distributed import fleet
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
lung_masks = 'labels/seg-lungs-LUNA16'
# ct_images = "testimgs"
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
    # prefix_save = os.path.join(path, "epoch" + str(epoch))
    # name = os.path.join(prefix_save, filename)
    # paddle.save(state, name)
    paddle.save(state, os.path.join(path, 'checkpoint_model_new.pth.tar'))
    if is_best:
        paddle.save(state, os.path.join(path, 'checkpoint_model_best.pth.tar'))
        # shutil.copyfile(name, os.path.join(path, 'checkpoint_model_best.pth.tar'))


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

    # args.cuda=False
    # args.save = args.save or 'work/vnet.base.{}'.format(datestr())
    # 设置数据读取器
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
    # import pdb
    # pdb.set_trace()
    # trainSet = dset.LUNA16(root='luna16', images=ct_images, targets=ct_targets,
    #                        mode="train", transform=trainTransform,
    #                        class_balance=class_balance, split=target_split, seed=args.seed, masks=masks)
    trainSet = LUNA16(root=r'./', images=ct_images, targets=ct_targets,
                      mode="train", transform=trainTransform,
                      class_balance=class_balance, split=target_split, seed=args.seed, masks=masks)
    # import pdb
    # pdb.set_trace()
    # trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, **kwargs)
    # train_sampler = paddle.io.RandomSampler(trainSet)
    train_sampler = paddle.io.DistributedBatchSampler(trainSet, batch_size=batch_size, shuffle=True, drop_last=True)
    trainLoader = DataLoader(trainSet, batch_sampler=train_sampler, **kwargs)

    print("loading eval set")
    # evalLoader = DataLoader(
    #     LUNA16(root=r'/home/aistudio/work', images=ct_images, targets=ct_targets,
    #            mode="eval", transform=evalTransform, seed=args.seed, masks=masks, split=target_split),
    #     batch_size=batch_size, shuffle=True, **kwargs)
    evalSet = LUNA16(root=r'./', images=ct_images, targets=ct_targets,
                     mode="eval", transform=evalTransform, seed=args.seed, masks=masks, split=target_split)
    eval_sampler = paddle.io.DistributedBatchSampler(evalSet, batch_size=batch_size, shuffle=True)
    evalLoader = DataLoader(evalSet, batch_sampler=eval_sampler, **kwargs)

    # target_mean = trainSet.target_mean()
    # bg_weight = target_mean / (1. + target_mean)
    # fg_weight = 1. - bg_weight
    # print(bg_weight)
    class_weights = []
    # class_weights = paddle.to_tensor([bg_weight, fg_weight])

    # import pdb
    # pdb.set_trace()

    # trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    # evalF = open(os.path.join(args.save, 'eval.csv'), 'w')
    best_epoch = 0
    err_best = 100.
    best_dice = 0
    # import pdb
    # pdb.set_trace()
    # log_writer = open(os.path.join(args.save, "train_log.txt"),"a+",encoding='utf-8')
    for epoch in range(args.start_epoch + 1, args.nEpochs + 1):

        start_time = time.time()
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, model, trainLoader, optimizer, class_weights)
        print('start test nEpochs:{}'.format(epoch))

        err, dice = evale(args, epoch, model, evalLoader, optimizer, class_weights)
        print('cost time is {}'.format(time.time() - start_time))
        is_best = False
        # import pdb
        # pdb.set_trace()
        if err < best_prec1:
            best_epoch = epoch
            is_best = True
            best_prec1 = err
            best_dice = dice
        print('best ErrorRate:{:.8f}%,best_epoch is {},best_dice :{:.8f}%'.format(best_prec1, best_epoch, best_dice))
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1,
                         'dice': dice,
                         'optimizer_dict': optimizer.state_dict()},
                        is_best, args.save, "vnet", epoch)
        # os.system('./plot.py {} {} &'.format(len(trainLoader), args.save))
        # print('best_epoch is {},best_err is {}'.format(best_epoch, err))
        # log_writer.close()
    # for i in range(4):
    #     time.sleep(30)
    #     print('**********************************')
    # log_writer.close()
    # trainF.close()
    # evalF.close()


def train_dice(args, epoch, model, trainLoader, optimizer, weights):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        # print('m======{}'.format(m))
        # import pdb
        # pdb.set_trace()
        # try:
        if ((data < -1).sum().item() + (data > 1).sum().item()) > 0:
            continue
        optimizer.clear_grad()
        output = model(data)
        loss = utils.my_dice_loss(output, target)
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
        # if paddle.isnan(loss):
        #     print('when it is not a number dice_co:{},loss.item():{},loss:{},data:{},target:{},output:{},out_dict:{}'.format(Dice_coefficient,
        #     loss.item(),loss,data,target,output,out_dict))
        #     print('model.state_dict:{}'.format(model.state_dict()))
        #     break
        if paddle.isnan(loss):
            print('data:{}'.format(data))
            print('model.state_dict{}'.format(model.state_dict()))
            print('output:{}'.format(output))
        # log_writer.write('\nTrain Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}\tDice_coefficient:: {:.8f}%\tErrorRate:{}%\n'.format(
        #     partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
        #     loss.item(), Dice_coefficient, error_rate))
        # log_writer.add_scalar('TrainInfo',
        #                     'Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}\tDice_coefficient:: {:.8f}%\tErrorRate:{}%'.format(
        #                         partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
        #                         loss.item(), Dice_coefficient, error_rate),batch_idx)
        del Dice_coefficient
        del error_rate
        loss.backward()
        optimizer.step()
        # trainF.write('{},{},{},{}\n'.format(partialEpoch, loss.item(), Dice_coefficient, error_rate))
        # trainF.flush()
        # except:
        #     print('error happend data is {}'.format(data))


def eval_dice(args, epoch, model, evalLoader, optimizer, weights):
    model.eval()
    eval_loss = 0
    incorrect = 0
    Dice_coefficient = 0
    error_rate = 0
    for data, target in evalLoader:
        output = model(data)
        # import pdb
        # pdb.set_trace()
        loss = utils.paddle_dice_loss(output, target).item()

        eval_loss += loss
        Dice_coefficient += (1. - loss)
        error_rate += (1 - paddle.metric.accuracy(output.transpose([0, 2, 3, 4, 1]).reshape([-1, 2]),
                                                  target.reshape([-1, 1]), k=1).item())

    eval_loss /= len(evalLoader)  # loss function already averages over batch size
    nTotal = len(evalLoader)
    Dice_coefficient = 100. * Dice_coefficient / nTotal
    error_rate = 100. * error_rate / nTotal
    print('\nEval set: Average eval_loss: {:.4f}, Dice_coefficient: {}/{} ({:.0f}%),ErrorRate:{}%\n'.format(
        eval_loss, incorrect, nTotal, Dice_coefficient, error_rate))
    # log_writer.write('\nEval set: Average eval_loss: {:.4f}, Dice_coefficient: {}/{} ({:.0f}%),ErrorRate:{}%\n'.format(
    #     eval_loss, incorrect, nTotal, Dice_coefficient, error_rate))
    # # log_writer.add_scalar('EvalInfo',
    # #                     '\nEval set: Average eval_loss: {:.4f}, Dice_coefficient: {}/{} ({:.0f}%),ErrorRate:{}%\n'.format(
    # #                         eval_loss, incorrect, nTotal, Dice_coefficient, error_rate),epoch)
    # evalF.write('{},{},{},{}\n'.format(epoch, eval_loss, Dice_coefficient, error_rate))
    # evalF.flush()
    return error_rate, Dice_coefficient


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
