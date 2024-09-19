import numpy as np
import pickle
from neural_net import TwoLayerNet
import matplotlib.pyplot as plt

# 加载 CIFAR-10 数据集的函数
def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

# 加载所有训练数据和测试数据
def load_CIFAR10_data():
    X_train = []
    y_train = []

    for i in range(1, 6):
        X, y = load_CIFAR_batch(f'D:/python  object/assignment_KNN_NN/assignment/cs231n/datasets/cifar-10-batches-py/data_batch_{i}')
        X_train.append(X)
        y_train.append(y)

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_test, y_test = load_CIFAR_batch('D:/python  object/assignment_KNN_NN/assignment/cs231n/datasets/cifar-10-batches-py/test_batch')

    return X_train, y_train, X_test, y_test

# 预处理数据
def preprocess_data(X_train, X_test):
    # 将图像数据从 32x32x3 展平为 1x3072 的向量
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # 简单归一化：将像素值从 [0, 255] 归一化为 [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # 减去训练数据的均值
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    return X_train, X_test

# 训练和评估模型
def main():
    # 加载数据集
    X_train, y_train, X_test, y_test = load_CIFAR10_data()

    # 预处理数据
    X_train, X_test = preprocess_data(X_train, X_test)

    # 初始化模型参数
    input_size = 32 * 32 * 3  # CIFAR-10 输入数据的维度
    hidden_size = 300         # 隐藏层神经元数量
    num_classes = 10          # CIFAR-10 类别个数
    net = TwoLayerNet(input_size, hidden_size, num_classes)

    # 设置训练参数
    learning_rate = 1e-2
    reg = 1e-2
    num_iters = 2000
    batch_size = 100
    learning_rate_decay = 0.95

    # 训练模型
    stats = net.train(X_train, y_train, X_test, y_test, learning_rate=learning_rate, reg=reg, num_iters=num_iters, batch_size=batch_size, learning_rate_decay=learning_rate_decay, verbose=True)

    # 绘制训练和验证准确率随时间变化图
    plt.subplot(2, 1, 1)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.title('Classification accuracy history')
    plt.legend()

    # 绘制损失历史
    plt.subplot(2, 1, 2)
    plt.plot(stats['loss_history'])
    plt.title('Training Loss History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    # 计算测试集准确率
    test_acc = (net.predict(X_test) == y_test).mean()
    print(f'Test accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    main()
