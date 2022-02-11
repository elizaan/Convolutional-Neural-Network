from re import X
import numpy as np
from mnist import MNIST
from sklearn import metrics
import pickle
import sys

class Conv2D:
    def __init__(self, filter, kernal, stride=1, padding=0):
        self.filter = filter
        self.kernal = kernal
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        self.x = x
        self.input_shape = x.shape
        self.W = np.zeros((self.filter, self.x.shape[1], self.kernal, self.kernal))
        self.b = np.zeros((self.filter))
        n_filters, f_depth, f_height, f_width = self.W.shape
        data_len, x_depth, x_height, x_width = x.shape
        out_height = int((x_height + 2 * self.padding - f_height) / self.stride + 1)
        out_width = int((x_width + 2 * self.padding - f_width) / self.stride + 1)
        out = np.zeros((data_len, n_filters, out_height, out_width))
        self.x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding) ), 'constant')
        
        for i in range(self.filter):
            self.W[i, :, :, :] = np.random.randn(f_depth, f_height, f_width)
            self.b[i] = np.random.randn()

        for i in range(data_len):
            for j in range(n_filters):
                for k in range(out_height):
                    for l in range(out_width):
                        out[i, j, k, l] = np.sum(self.x[i, :, k * self.stride:k * self.stride + f_height,
                                                          l * self.stride:l * self.stride + f_width] * self.W[j, :, :, :]) + self.b[j]
                                                          
        
        self.output = out
        return out

    def backward(self, grad, lr):
        self.grad = grad
        n_filters, f_depth, f_height, f_width = self.W.shape
        data_len, x_depth, x_height, x_width = self.x.shape

        dout = np.zeros_like(self.x)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        
        for i in range(data_len):
            for j in range(n_filters):
                for k in range(0, self.input_shape[2] - f_height + 1, self.stride):
                    for l in range(0, self.input_shape[3] - f_width + 1, self.stride):
                        dout[i, :, k:k + f_height, l:l + f_width] = dout[i, :, k:k + f_height, l:l + f_width] + self.grad[i, j, k // self.stride, l // self.stride ] * self.W[j]
                        dW[j] += self.x[i, :, k:k + f_height, l:l + f_width] * self.grad[i, j, k // self.stride, l // self.stride]
                        db[j] += self.grad[i, j, k // self.stride, l // self.stride]


        self.W -= lr * dW
        self.b -= lr * db
        return dout
    
class MaxPool2D:
    def __init__(self, pool_size=2, stride=1):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.x = x
        data_len, x_depth, x_height, x_width = x.shape
        out_height = int((x_height - self.pool_size) / self.stride + 1)
        out_width = int((x_width - self.pool_size) / self.stride + 1)
        out = np.zeros((data_len, x_depth, out_height, out_width))
        for i in range(data_len):
            for j in range(x_depth):
                for k in range(out_height):
                    for l in range(out_width):
                        out[i, j, k, l] = np.max(x[i, j, k * self.stride:k * self.stride + self.pool_size,
                                                    l * self.stride:l * self.stride + self.pool_size])
        self.output = out
        return out

    def backward(self, grad, lr):
        data_len, x_depth, x_height, x_width = self.x.shape
        dout = np.zeros_like(self.x)
        for i in range(data_len):
            for j in range(x_depth):
                for k in range(x_height):
                    for l in range(x_width):
                        x_masked = self.x[i, j, k:k + self.pool_size, l:l + self.pool_size]
                        if np.max(x_masked) == self.x[i, j, k, l]:
                            dout[i, j, k, l] = grad[i, j, k // self.stride, l // self.stride]

        return dout


class Flatten:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        self.x_shape = x.shape
        self.output = x.reshape(x.shape[0], -1)
        return self.output

    def backward(self, grad, lr):
        return grad.reshape(self.x_shape)

class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, grad, lr):
        return grad * (self.x > 0)

class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, grad, lr):
        return grad * (self.output * (1 - self.output))


class FullConnected:
    def __init__(self, n_output):
        self.n_output = n_output

    def forward(self, x):
        self.x = x
        self.W = np.random.randn(self.x.shape[1], self.n_output)
        self.b = np.random.randn(self.n_output)
        self.output = np.dot(self.x, self.W) + self.b
        return np.dot(x, self.W) + self.b

    def backward(self, grad, lr):
        self.dW = np.dot(self.x.T, grad)
        self.db = np.sum(grad, axis=0)
        dout = np.dot(grad, self.W.T)

        self.W -= lr * self.dW
        self.b -= lr * self.db
        return dout

class Crossentropyloss:
    def __init__(self):
        pass

    def forward(self, y_true, y_pred):
        self.y_pred = y_pred
        self.y_true = y_true
        self.output = -np.sum(self.y_true * np.log(y_pred + 1e-7))
        return -np.sum(self.y_true * np.log(y_pred + 1e-7))

    def backward(self):
        return -(self.y_true / (self.y_pred + 1e-7))
        

class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        self.x = x
        for layer in self.layers:
            x = layer.forward(x)

        self.output = x
        return x

    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
        return grad

    def fit(self, x, y, lr=0.001, epochs=5, batch = 32):
        for i in range(epochs):
            for k in range(0, batch):

                x_b = x[k:k + batch]
                y_b = y[k:k + batch]

                y_pred = self.forward(x_b)
                y_new = np.zeros_like(y_pred)

                for j in range(y_pred.shape[0]):
                    y_new[j, np.argmax(y_pred[j])] = 1

                crossentropy = Crossentropyloss()
                loss = crossentropy.forward(y_new, y_pred)


                print("The loss value for epoch {} and batch {} is: {}".format(i+1, k, loss))
                grad = crossentropy.backward()
                self.backward(grad, lr)

                print("Accuracy value on train data for epoch {} and batch {} is: ".format(i+1, k), metrics.accuracy_score(y_b, np.argmax(y_pred, axis=1)))
                print("F1 score on train data for epoch {} and batch {} is: ".format(i+1, k), metrics.f1_score(y_b, np.argmax(y_pred, axis=1), average='macro'))

                print('\n')

            # y_new = np.zeros_like(y_pred)
            # for j in range(y_pred.shape[0]):
            #     y_new[j, np.argmax(y_pred[j])] = 1
            # grad = 2 * (y_pred - y_new)
            # self.backward(grad, lr)

    def predict(self, x):
        y_hat = self.forward(x)
        return y_hat




if __name__ == '__main__':

    def Random_data():

        x = np.random.randn(2, 3, 4, 4)
        y = np.random.randn(2, 3, 4, 4)
        z = np.random.randn(2, 3, 4, 4) 
        print(x.shape, y.shape, z.shape)

        model = Sequential()
        model.add(Conv2D(6, 3, 1, 1))
        model.add(ReLU())
        model.add(MaxPool2D(2, 2))
        model.add(Flatten())
        model.add(FullConnected(10))
        model.add(Softmax())
        model.fit(x, y, lr=0.01, epochs=10)
        print(model.predict(z))

    def MNIST_data_preprocessing(sample_size=2000, epochs=5, batch_size=32):

        print("Calculating loss, accuracy, and F1 Score for MNIST data...")

        data = MNIST('inputs/')
        x_train, y_train = data.load_training()
        x_test, y_test = data.load_testing()

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        x_train = x_train.reshape((x_train.shape[0], 1, 28, 28))
        x_test = x_test.reshape((x_test.shape[0], 1, 28, 28))

        y_train = y_train.reshape((y_train.shape[0], 1))
        y_test = y_test.reshape((y_test.shape[0], 1))

   

        model = Sequential()
        model.add(Conv2D(6, 5, 1, 2))
        model.add(ReLU())
        model.add(MaxPool2D(2, 2))
        model.add(Conv2D(12, 5, 1, 0))
        model.add(ReLU())
        model.add(MaxPool2D(2, 2))
        model.add(Conv2D(100, 5, 1, 0))
        model.add(ReLU())
        model.add(Flatten()) #single array conversion
        model.add(FullConnected(10))
        model.add(Softmax())
        model.fit(x_train[0:sample_size], y_train[0:sample_size], lr=0.001, epochs=epochs, batch=batch_size)
        y_pred_test = model.predict(x_test[0:sample_size])
        
        print(y_pred_test)

        print("Accuracy value on test data is: " , metrics.accuracy_score(y_test[0:sample_size], np.argmax(y_pred_test, axis=1)))
        print("F1 score on test data is: " , metrics.f1_score(y_test[0:sample_size], np.argmax(y_pred_test, axis=1), average='macro'))

    def CIFAR_data_preprocessing(sample_size=2000, epochs=5, batch_size=32):

        print("Calculating loss, accuracy, and F1 Score for CIFAR data...")
        data_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
        data_dir = 'cifar/'

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        
        for file in data_files:

            with open(data_dir+file, 'rb') as f:
                dict = pickle.load(f, encoding='bytes')
                x_train.append(dict[b'data'])
                y_train.append(dict[b'labels'])

        x_train = np.concatenate(x_train, axis= 0)
        y_train = np.concatenate(y_train, axis=0)

        data_files2 = ['test_batch']

        for file in data_files2:
            with open(data_dir+file, 'rb') as f:
                dict = pickle.load(f, encoding='bytes')
                x_test.append(dict[b'data'])
                y_test.append(dict[b'labels'])

        x_test = np.concatenate(x_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        
        x_train = x_train.reshape((x_train.shape[0], 3, 32, 32))
        x_test = x_test.reshape((x_test.shape[0], 3, 32, 32))


        model = Sequential()
        model.add(Conv2D(16, 5, 1, 2))
        model.add(ReLU())
        model.add(MaxPool2D(2, 2))
        model.add(Conv2D(20, 5, 1, 2))
        model.add(ReLU())
        model.add(MaxPool2D(2, 2))
        model.add(Conv2D(20, 5, 1, 2))
        model.add(ReLU())
        model.add(MaxPool2D(2, 2))
        model.add(Flatten()) #single array conversion
        model.add(FullConnected(10))
        model.add(Softmax())
        model.fit(x_train[0:sample_size], y_train[0:sample_size], lr=0.001, epochs=epochs, batch=batch_size)
        y_pred_test = model.predict(x_test[0:sample_size])
        
        print(y_pred_test)

        print("Accuracy value on test data is: " , metrics.accuracy_score(y_test[0:sample_size], np.argmax(y_pred_test, axis=1)))
        print("F1 score on test data is: " , metrics.f1_score(y_test[0:sample_size], np.argmax(y_pred_test, axis=1), average='macro'))


    old_stdout = sys.stdout

    log_file = open("1605089.log","w")

    sys.stdout = log_file

    

    
    
    
    CIFAR_data_preprocessing(5000, 5, 32)
    MNIST_data_preprocessing(5000, 5, 32)

    sys.stdout = old_stdout
    log_file.close()