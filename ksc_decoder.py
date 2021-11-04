# -*- coding: utf-8 -*-
# 整个模型训练测试验证代码，并保存最优模型，打印测试数据
import numpy as np
import scipy.io as sio
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import keras.callbacks as kcallbacks
import time
from sklearn import preprocessing
from Utils import zeroPadding
import CapsuleNet
import os
import matplotlib.pyplot as plt
import keras
from keras import optimizers
from keras import backend as K
import argparse
from dividedataset import indexToAssignment, selectNeighboringPatch, sampling

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# setting the hyper parameters
parser = argparse.ArgumentParser(description="Capsule Network on KSC.")
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--nb_classes', default=13, type=int)
parser.add_argument('--lr', default=0.001, type=float,
                    help="Initial learning rate")
parser.add_argument('-r', '--routings', default=3, type=int,
                    help="Number of iterations used in routing algorithm. should > 0")
parser.add_argument('--r_loss', default=1, type=float,
                    help="The coefficient of recoonstruction loss.")
parser.add_argument('--debug', action='store_true',
                    help="Save weights by TensorBoard")
parser.add_argument('--input_dimension', default=176, type=int,
                    help="Number of dimensions for input datasets.")
parser.add_argument('--save_dir', default='./result/KSC-13-3%-decoder')
args = parser.parse_args()
print(args)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

ITER = 3
seeds = [1334, 1335, 1336]

PATCH_LENGTH = 6  # Patch_size (6*2+1)*(6*2+1)
img_rows = img_cols = 13 # 13, 13


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

# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('capsnet_loss'))
        self.accuracy['batch'].append(logs.get('capsnet_acc'))
        self.val_loss['batch'].append(logs.get('val_capsnet_loss'))
        self.val_acc['batch'].append(logs.get('val_capsnet_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('capsnet_loss'))
        self.accuracy['epoch'].append(logs.get('capsnet_acc'))
        self.val_loss['epoch'].append(logs.get('val_capsnet_loss'))
        self.val_acc['epoch'].append(logs.get('val_capsnet_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig(args.save_dir + '/figure.png')
        # plt.show()

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

# 调用设计好的模型
def model_CAPS():
    model, eval_model = CapsuleNet.CapsnetBuilder_2D_Deconv.build_capsnet(
                                                    input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
                                                    n_class=len(np.unique(np.argmax(y_train, 1))),
                                                    routings=args.routings,
                                                    c=args.input_dimension)

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mae'],
                  loss_weights=[1., args.r_loss],
                  metrics={'capsnet': 'accuracy'})
    model.summary(positions=[.33, .61, .75, 1.])
    return model

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

    # 重构训练、验证、测试数据集
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
    model = model_CAPS()

    # 创建一个实例history
    history = LossHistory()

    # callbacks
    log = kcallbacks.CSVLogger(args.save_dir + '/log.csv')
    earlyStopping6 = kcallbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto')
    saveBestModel6 = kcallbacks.ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1,
                                                save_best_only=True,
                                                mode='auto')
    reduce_LR_On_Plateau = kcallbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, mode='auto',
                                                        verbose=1, min_lr=0)

    model.compile(optimizer=Adam(lr=args.lr),
                  loss=[margin_loss, 'mae'],
                  loss_weights=[1., args.r_loss],
                  metrics={'capsnet': 'accuracy'})
    print(x_train.shape, x_test.shape)


    # 训练
    tic1 = time.clock()
    history_3d_SEN = model.fit([x_train, y_train], [y_train, x_train],
        validation_data=[[x_val, y_val], [y_val, x_val]],
        batch_size=args.batch_size,
        epochs=args.epochs, shuffle=True, callbacks=[log, earlyStopping6, saveBestModel6, history, reduce_LR_On_Plateau])
    toc1 = time.clock()

    print('Training Time: ', toc1 - tic1)

    # 绘制acc-loss曲线
    history.loss_plot('epoch')

    print("Training Finished.")