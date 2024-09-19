import numpy as np
import pickle


# 加载 CIFAR-10 数据集
def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


# 加载所有训练数据
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

    return X_train, X_test


# 向量化的距离计算（欧几里得距离）
def compute_distances(X_train, X_test):
    dists = np.sqrt(np.sum(X_test ** 2, axis=1).reshape(-1, 1) + np.sum(X_train ** 2, axis=1) - 2 * np.dot(X_test, X_train.T))
    return dists


# 使用 KNN 进行预测
def predict_labels(dists, y_train, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)

    for i in range(num_test):
        # 找到距离最近的 k 个训练样本
        closest_y = y_train[np.argsort(dists[i])[:k]]
        # 统计每个类别的出现次数，选择出现次数最多的类别
        y_pred[i] = np.bincount(closest_y).argmax()

    return y_pred


# 使用交叉验证来选择最佳 k 值
def cross_validate_knn(X_train, y_train, k_values, num_folds=2):
    fold_size = X_train.shape[0] // num_folds
    accuracies = np.zeros(len(k_values))

    for fold in range(num_folds):
        # 创建训练集和验证集
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size
        X_val_fold = X_train[val_start:val_end]
        y_val_fold = y_train[val_start:val_end]

        X_train_fold = np.concatenate([X_train[:val_start], X_train[val_end:]], axis=0)
        y_train_fold = np.concatenate([y_train[:val_start], y_train[val_end:]], axis=0)

        # 计算训练集和验证集之间的距离
        dists = compute_distances(X_train_fold, X_val_fold)

        for i, k in enumerate(k_values):
            y_val_pred = predict_labels(dists, y_train_fold, k)
            accuracy = np.mean(y_val_pred == y_val_fold)
            accuracies[i] += accuracy

    # 取每个 k 值的平均准确率
    accuracies /= num_folds
    best_k = k_values[np.argmax(accuracies)]

    return best_k, accuracies

# 定义 sample_data 函数，用于从原始数据集中采样
def sample_data(X, y, num_samples=1000):
    idx = np.random.choice(X.shape[0], num_samples, replace=False)
    return X[idx], y[idx]



