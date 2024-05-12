import time
import numpy as np
from numpy.linalg import pinv
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder


class ELM(object):
    def __init__(self, n_hidden, n_input, n_output, act_func="sigmoid", type_="CLASSIFIER", seed=None):
        self.beta = None  # beta矩阵
        self.b = None  # 偏置矩阵
        self.W = None  # 权重矩阵
        self.n_input = n_input
        self.n_output = n_output
        self.seed = seed
        self.n_hidden = n_hidden  # 隐含层节点个数
        self.act_func = None  # 激活函数
        self.type_ = type_  # 极限学习机类别   :CLASSIFIER->分类， REGRESSION->回归
        self._init(act_func)

    def _init(self, act_func):
        if act_func == "sigmoid":
            self.act_func = self._sigmoid
        elif act_func == 'sine':
            self.act_func = self._sine
        elif act_func == 'cos':
            self.act_func = self._cos
        if self.seed is None:
            # self.W = np.random.uniform(-1, 1, size=(self.n_input, self.n_hidden))
            self.W = np.random.randn(self.n_input, self.n_hidden)
            # self.b = np.random.uniform(-0.4, 0.4, size=self.n_hidden)
            self.b = np.random.randn(self.n_hidden)
        else:
            np.random.seed(self.seed)
            self.W = np.random.uniform(-1, 1, size=(self.n_input, self.n_hidden))
            self.b = np.random.uniform(0, 1, size=self.n_hidden)
            np.random.default_rng()
            # self.b = np.random.randn(self.n_hidden)

    def set_w_b(self, w, b):
        self.W = w
        self.b = b

    def fit(self, X, T):
        if self.type_ == "CLASSIFIER":
            encoder = OneHotEncoder()
            # 将输入的T转换为独热编码的形式，注意这里是稀疏存储的形式
            T = encoder.fit_transform(T.reshape(-1, 1)).toarray()
        elif self.type_ == "REGRESSION":
            T = T.reshape(-1, 1)
        start = time.time()
        # 隐含层输出矩阵 n*n_hidden
        H = self.act_func(np.dot(X, self.W) + self.b)
        # 输出权重系数 n_hidden*m
        self.beta = self.calculate_beta(H, T)
        end = time.time()
        return end - start

    def set_seed(self, seed):
        self.seed = seed  # 设置seed

    def calculate_beta(self, H, T):
        # 输出权重系数 n_hidden*m，β的计算公式为：((H.T*H)^-1)*H.T*T
        # return np.dot(np.dot(pinv(np.dot(H.T, H)), H.T), T)
        return np.dot(pinv(H), T)

    def predict(self, x):
        """
            返回值为y_pred
        """
        h = self.act_func(np.dot(x, self.W) + self.b)
        res = np.dot(h, self.beta)
        if self.type_ == "REGRESSION":  # 回归预测
            return res
        elif self.type_ == "CLASSIFIER":  # 分类预测
            # 返回最大值所在位置的索引，因为最大值位置的类别恰好等于索引
            return np.argmax(res, axis=1)

    @staticmethod
    def score(y_true, y_pred):
        # 测试准确度
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def MSE(y_pred, y_true):
        # 均方误差
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def RMSE(y_pred, y_true):
        # 均方根误差
        # 均方根误差等于均方误差开根号
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _sine(x):
        return np.sin(x)

    @staticmethod
    def _cos(x):
        return np.cos(x)

    # @staticmethod
    # def _softplus(x):
    #     return np.log(1+np.exp(x))

    @staticmethod
    def _tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x), np.exp(-x))

    # @staticmethod
    # def gaussian(x):
    #     return np.exp(-x**2)
