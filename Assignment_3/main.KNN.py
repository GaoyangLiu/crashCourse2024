from KNN_Model import *

def main():
    # 加载数据集
    X_train, y_train, X_test, y_test = load_CIFAR10_data()

    # 预处理数据
    X_train, X_test = preprocess_data(X_train, X_test)

    # 设置 k 的范围，用于自动搜索最佳 k 值
    k_values = list(range(1, 21))

    # 减少数据集大小进行调试
    # X_train_small, y_train_small = sample_data(X_train, y_train, num_samples=2000)
    # X_test_small, y_test_small = sample_data(X_test, y_test, num_samples=1000)

    # # 交叉验证找到最佳 k 值（使用较小数据集）
    # print("Finding the best k ...")
    # best_k, accuracies = cross_validate_knn(X_train_small, y_train_small, k_values)
    #
    # print(f"Best k found: {best_k}")
    # print(f"Accuracies for each k: {accuracies}")
    #
    # # 使用找到的最佳 k 值在较小的测试集上进行预测
    # print("Evaluating on the test data...")
    # dists_test = compute_distances(X_train_small, X_test_small)
    # y_test_pred = predict_labels(dists_test, y_train_small, k=best_k)
    #
    # # 计算测试集上的准确率
    # test_accuracy = accuracy_score(y_test_small, y_test_pred)
    # print(f"Test accuracy with best k = {best_k}: {test_accuracy:.4f}")

    # 计算测试集上的准确率
    #  test_accuracy = accuracy_score(y_test_small, y_test_pred)
    #  print(f"Test accuracy with best k = {best_k}: {test_accuracy:.4f}")

    #   交叉验证找到最佳 k 值
    print("Finding the best k ...")
    best_k, accuracies = cross_validate_knn(X_train, y_train, k_values)

    print(f"Best k found: {best_k}")
    print(f"Accuracies for each k: {accuracies}")

    # 使用找到的最佳 k 值在测试集上进行预测
    print("Evaluating on the test data...")
    dists_test = compute_distances(X_train, X_test)
    y_test_pred = predict_labels(dists_test, y_train, k=best_k)

    # 计算测试集上的准确率
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy with best k = {best_k}: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()