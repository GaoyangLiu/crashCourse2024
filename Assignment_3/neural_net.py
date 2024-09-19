import numpy as np

class TwoLayerNet(object):


    def __init__(self, input_size, hidden_size, output_size, std=1e-4):

        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # 前向传播
        hidden_layer = np.maximum(0, X.dot(W1) + b1)  # ReLU 激活函数
        scores = hidden_layer.dot(W2) + b2

        if y is None:
            return scores

        # 计算损失
        shifted_logits = scores - np.max(scores, axis=1, keepdims=True) # 减去样本最大得分
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True) # 所有类别的指数和，作为 Softmax 函数的分母
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs) # 计算每个类别的概率
        loss = -np.sum(log_probs[np.arange(N), y]) / N # 计算交叉熵损失

        # 加入 L2 正则化
        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        # 反向传播
        grads = {}
        dscores = probs
        dscores[np.arange(N), y] -= 1
        dscores /= N

        grads['W2'] = hidden_layer.T.dot(dscores) + reg * W2 # W2 的梯度即 loss 对 W2 的偏导
        grads['b2'] = np.sum(dscores, axis=0) # b2 的梯度即 loss 对 b2 的偏导

        dhidden = dscores.dot(W2.T)
        dhidden[hidden_layer <= 0] = 0

        grads['W1'] = X.T.dot(dhidden) + reg * W1
        grads['b1'] = np.sum(dhidden, axis=0)

        return loss, grads

    def train(self, X, y, X_val, y_val, learning_rate, learning_rate_decay, reg, num_iters, batch_size, verbose=False):

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # 使用随机梯度下降法优化参数
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            # 创建小批量数据
            batch_indices = np.random.choice(num_train, batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # 计算损失和梯度
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            # 使用梯度下降法更新参数
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print(f'Iteration {it}/{num_iters}: Loss {loss}')

            # 每个 epoch 结束后，检查训练集和验证集的准确率，并衰减学习率
            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):

        hidden_layer = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
        scores = hidden_layer.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)
        return y_pred


