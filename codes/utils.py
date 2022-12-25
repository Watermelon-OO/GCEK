# coding: utf-8
import os
import subprocess

import numpy as np
from mxnet import ndarray as nd
from mxnet.gluon import data as gdata, nn
from sklearn import metrics

import models
from dataProcess.NYTData import *

__all__ = ['load_data', 'data_loader', 'net_init', 'predict',
           'PrecisionAndRecall', 'split_and_load', 'get_mem']


def load_data(train_path, test_path_all):
    """load all data, transform them to DataLoader

    Args:
        train_path: string
            data path of training data
        test_path_all: string
            data path of testing data with All sentences

    Returns:
        trainDataSet: DataSet
        testDataSet_All: DataSet
        testDataSet_One: DataSet
        testDataSet_Two: DataSet
    """

    trainDataSet = np.load(train_path, allow_pickle=True)
    print(trainDataSet.shape)
    testDataSet_All = np.load(test_path_all, allow_pickle=True)
    print(testDataSet_All.shape)

    return trainDataSet, testDataSet_All


def data_loader(trainDataSet, testDataSet_All, batch_size):
    """

    Args:
        trainDataSet: DataSet
            train DataSet
        testDataSet_All: DataSet
            test DataSet with All sentences
        batch_size: int

    Returns:
        trainDataLoader: DataLoader
        testDataLoader_All: DataLoader
        testDataLoader_One: DataLoader
        testDataLoader_Two: DataLoader

    """
    batchify_fn = bag_batchify_fn
    trainDataLoader = gdata.DataLoader(trainDataSet, batch_size=batch_size,
                                       batchify_fn=batchify_fn, shuffle=True,
                                       last_batch='discard')

    testDataLoader_All = gdata.DataLoader(testDataSet_All, batch_size=256,
                                          batchify_fn=batchify_fn, shuffle=False,
                                          last_batch='keep')

    return trainDataLoader, testDataLoader_All


def net_init(init_func, config):
    """initialize the network

    Args:
        init_func:
            the method used to initialize network
        config:
    Returns:
        model: nn.Block
            the nn model after initialization
    """
    model = getattr(getattr(models, config.lib), config.model)(config)
    # bert style
    # model.collect_params(model.__class__.__name__ + '(\d+)_(?!bert)').initialize(ctx=config.ctx,
    #                                                                              init=init_func)
    model.collect_params().initialize(ctx=config.ctx, init=init_func)
    # hybridize model
    model.hybridize()
    print(config.HyperParams)
    return model




def predict(data_iter: gdata.DataLoader, net: nn.Block, rel_num: int, ctx, is_all) -> (np.ndarray, int):
    """predict probabilities of test data

    Args:
        data_iter: DataLoader
            DataLoader of gluon
        net: nn.Block
            a neural network for predict
        rel_num: int
            number of relations
        ctx: mx.gpu()
            gpu context
        is_all: boolean
            if data is All sentence version
    Returns:
        pr_curve: list
        pred_res_over1: np.ndarray
        auc: float
        positive_num: int
    """
    pred_output = []
    ground_truth = []
    sen_num = []
    for inputs in data_iter:
        features = []
        # ids, sen_num, ent_id, ent_pos, words_ids, lpos_ids, rpos_ids, masks, valid_length, labels = inputs
        for feature in inputs[1:]:
            features.append(nd.array(np.concatenate(feature, axis=0), ctx=ctx))
        # 真实标签 (batch_size, 4)
        ground_truth.append(np.array(inputs[-1]))
        sen_num.append(np.concatenate(inputs[1], axis=0))
        # softmax on dimension 1
        pred = net(*features)
        pred_output.append(pred)

    ground_truth = np.concatenate(ground_truth, axis=0)
    sen_num = np.concatenate(sen_num, axis=0)
    pred_output = nd.concat(*pred_output, dim=0)
    pred_output = nd.softmax(pred_output, axis=1).asnumpy()
    # 将每一个标签和其预测概率对齐，排除NA样本
    pred_res = []
    pred_res_over1 = []
    over1_idx = sen_num > 1
    for i in range(1, rel_num):
        # 取出当前label的预测概率
        # (batch_size, 1)
        prob = pred_output[:, i:i + 1]
        # 预测的label
        # (batch_size, 1)
        y_hat = np.ones_like(prob) * i
        # 与ground truth 对齐，标记正确的label
        # (batch_size, 1)
        is_label = np.sum(y_hat == ground_truth, axis=1, keepdims=True)
        # 对结果进行拼接 [is_label, prob]
        # (batch_size, 2)
        pred_res.append(np.concatenate((is_label, prob), axis=1))
        # bag which's total sentences is more than 1
        if is_all:
            prob_over1 = pred_output[over1_idx, i:i + 1]
            y_hat_over1 = np.ones_like(prob_over1) * i
            is_correct_over1 = np.sum(y_hat_over1 == ground_truth[over1_idx, :], axis=1, keepdims=True)
            pred_res_over1.append(np.concatenate((is_correct_over1, prob_over1), axis=1))

    # 全部预测结果  (batch_size*52, 2)
    pred_res = np.concatenate(pred_res, axis=0)
    pr_curve = metrics.precision_recall_curve(pred_res[:, 0], pred_res[:, 1])
    auc_macro = metrics.average_precision_score(pred_res[:, 0], pred_res[:, 1])
    if is_all:
        pred_res_over1 = np.concatenate(pred_res_over1, axis=0)
    else:
        pred_res_over1 = pred_res

    positive_num = np.sum(ground_truth > 0)
    return pr_curve, pred_res_over1, auc_macro, positive_num


def PrecisionAndRecall(pred_res: np.ndarray, positive_num: int, top_range: int = 2000) -> (int, float, list, list):
    """eval_metric: precision & recall

    Args:
        pred_res: np.ndarray
            the predict result of model (sentences in bag are more than one)
        positive_num: int
            the number of positive number
        top_range: int
            the range of top K

    Returns:
        correct: int
            correct number of result
        all_pre: list
            top K precision
        all_rec: list
            top K recall

    """
    # 对预测结果进行排序，从大到小
    pred_res_sort = np.array(sorted(pred_res, key=lambda x: -x[1]))
    # auc_2000 = metrics.average_precision_score(pred_res_sort[:2000, 0], pred_res_sort[:2000, 1])
    correct = 0.0
    all_pre = []
    all_rec = []
    # 默认选取前2000个结果
    for i in range(top_range):
        correct += pred_res_sort[i, 0]
        precision = correct / (i + 1)
        recall = correct / positive_num
        all_pre.append(precision)
        all_rec.append(recall)
    return correct, all_pre, all_rec


def split_and_load(data, ctx):
    """split data and load on multi-GPU

    Args:
        data: np.NDArray
        ctx: list
            list of gpu context

    Returns:
        ctx_features: nd.NDArray
        ctx_labels: nd.NDArray

    """
    ctx_features = []
    ctx_labels = []
    n, k = len(data[0]), len(ctx)
    m = n // k  # 假设可以整除
    assert m * k == n, '# examples is not divided by # devices.'
    for i in range(k):
        features = []
        # 去掉id和label
        for feature in data[1:-1]:
            features.append(nd.array(np.concatenate(feature[i * m: (i + 1) * m], axis=0), ctx=ctx[i]))
        ctx_features.append(features)
        ctx_labels.append(nd.array(data[-1][i * m: (i + 1) * m], ctx=ctx[i], dtype='int32').reshape(m))

    return ctx_features, ctx_labels


def grad_clipping(params, theta, ctx):
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def get_mem():
    res = subprocess.check_output(['ps', 'u', '-p', str(os.getpid())])
    return int(str(res).split()[15]) / 1e3
