# -*- coding: utf-8 -*-
# 整个模型训练测试验证代码，并保存最优模型，打印测试数据
import numpy as np
import scipy.io as sio
from keras.utils.np_utils import to_categorical
import time
from sklearn import preprocessing
import CapsuleNet
import os
import argparse
from dividedataset import indexToAssignment, selectNeighboringPatch, sampling
import collections
from sklearn import metrics
from Utils import modelStatsRecord, averageAccuracy, zeroPadding


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# setting the hyper parameters
parser = argparse.ArgumentParser(description="Capsule Network on KSC.")
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--nb_classes', default=13, type=int)
parser.add_argument('--input_dimension', default=176, type=int,
                    help="Number of dimensions for input datasets.")
parser.add_argument('--save_dir', default='./result/KSC-13-3%-decoder')
args = parser.parse_args()
print(args)

ITER = 3
seeds = [1334, 1335, 1336]

PATCH_LENGTH = 6
img_rows = img_cols = 13

TOTAL_SIZE = 5211
VAL_SIZE = 162
TRAIN_SIZE = 162
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
VALIDATION_SPLIT = 0.97
# 0.995 34
# 0.99 61
# 0.97 162
# 0.95 268
# 0.93 373
# 0.9  528

# 加载数据
mat_data = sio.loadmat('./datasets/ksc/KSC.mat')
data_IN = mat_data['KSC']
# 标签数据
mat_gt = sio.loadmat('./datasets/ksc/KSC_gt.mat')
gt_IN = mat_gt['KSC_gt']

new_gt_IN = gt_IN

# 对数据进行reshape处理之后，进行scale操作
data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

# 标准化操作，即将所有数据沿行沿列均归一化道0-1之间
data = preprocessing.scale(data)
print(data.shape)

# 对数据边缘进行填充操作，有点类似之前的镜像操作
data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
whole_data = data_
padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)

train_data = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, args.input_dimension))
test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, args.input_dimension))

KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, args.nb_classes))

for index_iter in range(ITER):
    print("# %d Iteration" % (index_iter + 1))

    # save the best validated model
    best_weights_path = args.save_dir + '/KSC_best_weights_' + str(index_iter + 1) + '.hdf5'

    # 通过sampling函数拿到测试和训练样本
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)

    # gt本身是标签类，从标签类中取出相应的标签 -1，转成one-hot形式
    y_train = gt[train_indices] - 1
    y_train = to_categorical(np.asarray(y_train))

    y_test = gt[test_indices] - 1
    y_test = to_categorical(np.asarray(y_test))

    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], args.input_dimension)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], args.input_dimension)

    # 在测试数据集上进行验证和测试的划分
    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]

    gt_test = gt[test_indices] - 1
    gt_test = gt_test[:-VAL_SIZE]


    ############################################################################################################
    model, eval_model = CapsuleNet.CapsnetBuilder_2D_Deconv.build_capsnet(input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
                                                                          n_class=len(np.unique(np.argmax(y_train, 1))),
                                                                          routings=args.routings,
                                                                          c=args.input_dimension,)
    model.load_weights(best_weights_path)

    # 测试
    tic2 = time.clock()
    y_pred = eval_model.predict(x_test)
    toc2 = time.clock()
    pred_test = y_pred.argmax(axis=1)
    collections.Counter(pred_test)

    overall_acc = metrics.accuracy_score(pred_test, gt_test)
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test)
    each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test)
    KAPPA.append(kappa)
    OA.append(overall_acc)
    AA.append(average_acc)

    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc
    print('Test time:', toc2 - tic2)
    print("Testing Finished.")


# 自定义输出类
modelStatsRecord.outputStats_assess_test(KAPPA, OA, AA, ELEMENT_ACC, TESTING_TIME, args.nb_classes,
                             args.save_dir + '/KSC_test.txt',
                             args.save_dir + '/KSC_test_element.txt')