# coding: utf-8
import argparse
import datetime
import json
import os
import random
import sys
import time

import mxnet as mx
import numpy as np
from mxnet import gluon, init, autograd
from mxnet.gluon import loss as gloss

from config import Config
from utils import *

current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_directory)
sys.path.append(root_path)

# 声明配置文件对象
config = Config()


def train(net, train_iter, test_iter_All, label_num, batch_size,
          trainers: list, ctxs, num_epochs, record=0, lr_decay=0):
    """training process

    Args:
        net: nn.Block
        train_iter: DataLoader
        test_iter_All: DataLoader
        label_num: int
        batch_size: int
        trainers: Trainer
        ctxs: list
        num_epochs: int
        record: int
        lr_decay: int

    Returns:

    """
    assert len(trainers) is not 0, 'trainer is not define'
    print('training on', ctxs, flush=True)
    time_now = datetime.datetime.now()
    print(time_now.strftime('%Y-%m-%d_%H:%M:%S'), flush=True)
    print(net, flush=True)

    # loss function
    loss_function = gloss.SoftmaxCrossEntropyLoss()
    loss_function.hybridize()

    best_auc = 0
    best_mean300 = 0
    best_record = dict()
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        # 记录每个epoch的train loss、样本数以及开始时间
        train_l_sum, simple_size_global, start = 0.0, 0, time.time()
        for inputs in train_iter:
            # 记录每个step的train loss和样本数
            train_l_local = 0
            simple_size_local = 0
            # (batch,...)
            # 多GPU加载数据，按照batch_size均分
            ctx_features, ctx_labels = split_and_load(inputs, ctxs)
            # forward
            with autograd.record():
                y_hats = [net(*ctx_feature, ctx_label) for ctx_feature, ctx_label in zip(ctx_features, ctx_labels)]
                # ls = [loss_function(y_hat[0], ctx_label).sum() for y_hat, ctx_label in zip(y_hats, ctx_labels)]
                ls = [loss_function(y_hat, ctx_label).sum() for y_hat, ctx_label in zip(y_hats, ctx_labels)]

            # calculate loss and backward
            for l in ls:
                train_l_sum += l.asscalar()
                train_l_local += l.asscalar()
                l.backward()

            # update parameters
            for trainer in trainers:
                trainer.step(batch_size, ignore_stale_grad=True)

            for label in ctx_labels:
                simple_size_local += label.size
                simple_size_global += label.size
            global_step += 1
            # learning_rate decay
            if lr_decay and ((global_step % 150000) == 0):
                for trainer in trainers:
                    trainer.set_learning_rate(trainer.learning_rate * 0.1)
                    print("The learning rate is decayed. Now, lr is %f" % trainer.learning_rate, flush=True)

        batch_loss = train_l_sum / simple_size_global
        print('----------------------------------------', flush=True)
        print('epoch %d/%d, step %d, loss %.4f'
              % (epoch, num_epochs, global_step, batch_loss), flush=True)
        # inference process
        pr_curve, best_mean300, best_auc, best_record = inference(net, test_iter_All, label_num, ctx[0], epoch,
                                                                  best_mean300, best_auc, best_record,
                                                                  start)

    print('===================Best Result===================\n', flush=True)
    print(best_record, flush=True)


def inference(net, test_iter_All, label_num, ctx, epoch,
              Prebest_mean300, Prebest_auc, best_record, start):
    best_auc = Prebest_auc
    best_mean300 = Prebest_mean300
    # inference process
    # All sentence
    pr_curve, pred_res_all, auc_macro, positive_num = predict(test_iter_All, net, label_num, ctx, is_all=True)
    correct, all_pre, all_rec = PrecisionAndRecall(pred_res_all, positive_num, top_range=2000)
    mean_300_all = (all_pre[99] + all_pre[199] + all_pre[299]) / 3
    mean_500_all = (all_pre[99] + all_pre[299] + all_pre[499]) / 3

    # best performance
    if auc_macro >= Prebest_auc:
        best_auc = auc_macro
        best_mean300 = mean_300_all
        best_record['epoch'] = epoch
        best_record['correct'] = correct
        best_record['auc'] = auc_macro
        best_record['P@N'] = [all_pre[99], all_pre[199], all_pre[299], all_pre[499], mean_300_all, mean_500_all]

    # report intermediate result
    print('positive_num: %d\t correct: %d\t AUC_macro: %.4f'
          % (positive_num, correct, auc_macro), flush=True)
    print('======================All======================')
    print('P@100\t\tP@200\t\tP@300\t\tP@500\t\tP@300Mean\tP@500Mean', flush=True)
    print('%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f'
          % (all_pre[99], all_pre[199], all_pre[299], all_pre[499], mean_300_all, mean_500_all), flush=True)
    print('time %.1f sec' % (time.time() - start), flush=True)

    return pr_curve, best_mean300, best_auc, best_record


if __name__ == '__main__':

    # 随机数种子
    np.random.seed(100)
    random.seed(100)
    mx.random.seed(10000)

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_epochs', nargs=1, type=int, help='number of epochs', required=True)
    parser.add_argument('--lib', nargs=1, type=str, help='the lib include models', required=True)
    parser.add_argument('-m', '--model', nargs=1, type=str, help='the type of the model', required=True)
    parser.add_argument('-p', '--HyperParams', nargs=1, type=str, help='the HyperParams of the model')
    parser.add_argument('-e', '--encoder', nargs=1, type=str, help='the encoder of the model')
    parser.add_argument('-a', '--att', nargs=1, type=str, help='the attention module of the model')
    parser.add_argument('-b', '--batch_size', nargs=1, type=int, help='batch size')
    parser.add_argument('-o', '--optimizer', nargs=1, type=str, help='optimizer, default is AdaDelta')
    parser.add_argument('-c', '--ctx', nargs='+', type=int, help='gpu id')
    parser.add_argument('-r', '--record', nargs=1, type=int, help='record logs, default is 0')
    parser.add_argument('--data_version', nargs=1, type=str, help='choice use which version of data')
    # parser.add_argument('--nni', nargs=1, type=int, help='choice whether use nni')
    parser.add_argument('--ld', nargs=1, type=int, help='choice whether use lr decay')
    args = parser.parse_args()
    # 更新参数配置
    for k, v in args.__dict__.items():
        assert hasattr(config, k), 'config has no attribute %s' % k
        if v:
            if k == 'ctx':
                ctx = [mx.gpu(int(i)) for i in v]
                setattr(config, k, ctx)
            elif k == 'HyperParams':
                with open(os.path.join(root_path, 'codes', 'HyperParams', v[0] + '.json'), 'r') as fr:
                    HyperParams = json.load(fr)
                setattr(config, k, HyperParams)
            else:
                setattr(config, k, v[0])

    config.batch_size = config.HyperParams['bsz']
    # 载入数据
    if config.data_version == '520':
        trainDataSet, testDataSet_All = load_data(config.train_path_520K, config.testAll_path_520K)
        config.relEmbed_path = config.relEmbed_path_520K
        config.entEmbed_path = config.entEmbed_path_520K
        config.ent_num = config.ent_num_520K
    elif config.data_version == '570':
        trainDataSet, testDataSet_All = load_data(config.train_path_570K, config.testAll_path_570K)
        config.relEmbed_path = config.relEmbed_path_570K
        config.entEmbed_path = config.entEmbed_path_570K
        config.ent_num = config.ent_num_570K
    else:
        raise Exception('data version is wrong')

    trainDataLoader, testDataLoader_All = data_loader(trainDataSet, testDataSet_All, config.batch_size)
    # 初始化网络参数
    net = net_init(init.Xavier(), config)
    # 定义优化算法
    trainers = []
    if config.optimizer == 'sgd':
        trainers.append(gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': config.HyperParams['lr']}))
    elif config.optimizer == 'momentum':
        trainers.append(gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': config.HyperParams['lr'],
                                                                    'momentum': config.HyperParams['momentum']}))
    elif config.optimizer == 'AdaDelta':
        trainers.append(gluon.Trainer(net.collect_params(), 'AdaDelta', {'rho': config.HyperParams['lr']}))
    else:
        raise Exception('No such optimizer')
    # 开始训练
    train(net, trainDataLoader, testDataLoader_All, config.rel_num, config.batch_size, trainers,
          config.ctx, config.num_epochs, config.record, config.ld)
