import numpy as np
from datagenerator import DataGenerator
from matplotlib import pyplot as plt
from configparser import ConfigParser
from scipy.special import softmax as scipy_softmax

config = ConfigParser()
config.read('config.ini')
neural, layers = config['neural'], config['layers']

BATCH_SIZE = int(neural['batch_size'])
NUMBER_OF_CASES = int(neural['number_of_cases'])
SPLIT = [int(x) for x in neural['split'].split()]
SOFTMAX = bool(layers['softmax'])
DIMENSIONS = int(config['datagen']['background_size'])**2  # Width of picture
LR = float(neural['lr'])
LOSS_FUNC = neural['loss']
N_LAYERS = int(layers['n_layers'])
N_CLASSES = 4
output_act = 'relu'


class Layer():
    def __init__(self, n_nodes, act):
        self.n_nodes = n_nodes
        self.act = eval('self.'+act)
        self.weights = None  # Will be set later
        self.bias = np.ones(n_nodes)
        self.n_prev = None
        self.dotprod = None  # set in forward pass
        self.output = None  # ------||-----
        self.error = None  # set in backwards pass
        self.error_delta = None  # ------||----

    def print_layer(self):
        print(f'nodes: {self.n_nodes}, act{self.act}')

    def forward_pass(self, _input):
        #print((_input @ self.weights).shape, self.bias.shape)
        x = _input @ self.weights + self.bias
        z = self.act(x)
        self.dotprod = x
        self.output = z
        return z

    def linear(self, z, der=False):
        if(der):
            return 1
        return z

    def relu(self, z, der=False):
        if(der):
            return 1 * (z > 0)
        return z * (z > 0)

    def sigmoid(self, z, der=False):
        if(der):
            return z(1-z)
        return 1/(1+np.exp(-z))

    def tanh(self, z, der=False):
        if(der):
            return 1 - np.tanh(z) ** 2
        return np.tanh(z)


class Network:
    def __init__(self):
        self.layers = [Layer(n, act) for n, act in self.get_layer_pref()]
        self.config_layers()
        self.lr = LR
        self.loss_func = eval('self.'+LOSS_FUNC)

    def config_layers(self):
        index_range = range(len(self.layers))
        for i, layer in enumerate(self.layers):
            layer.n_prev = self.layers[i-1].n_nodes if (i-1) in index_range else DIMENSIONS
            layer.weights = np.random.uniform(-1.0, 1.0, (layer.n_prev, layer.n_nodes))

    @staticmethod
    def get_layer_pref():
        layer_p = []
        for i in range(1, N_LAYERS+1):
            n_nodes, act = layers[f'layer{i}'].split()
            n_nodes = int(n_nodes)
            layer_p.append((n_nodes, act))
        layer_p.append((N_CLASSES, output_act))
        return layer_p

    @staticmethod
    def softmax(_input, der=False):
        if(der):
            pass
        return scipy_softmax(_input, axis=1)

    @staticmethod
    def mean_square_error(predicted, trainY, der=False):
        #print(predicted.shape, trainY.shape)
        if der:
            return -2 * (trainY - predicted)
        # mse = 1/n * sum(error^2)
        mse = np.sum((trainY - predicted) ** 2) / len(trainY)
        return mse

    @staticmethod
    def cross_entropy(predicted, trainY, der=False):
        if(der):
            pass
        predicted_aslog = np.log(predicted)
        return - np.sum(trainY * predicted_aslog)

    def forward_pass(self, dataX):
        output_from_each_layer = []
        inputs_to_next = dataX
        for layer in self.layers:
            inputs_to_next = layer.forward_pass(inputs_to_next)
        outputs = self.softmax(inputs_to_next) if SOFTMAX else inputs_to_next
        return outputs

    def backwards_pass(self, trainX, pred, trainY):
        loss = self.loss_func(pred, trainY)
        #self.loss_func(pred, trainY, der=True)
        loss_delta = self.softmax(pred, der=True) if SOFTMAX else loss
        error_delta = loss_delta
        for layer in reversed(self.layers):
            layer.error = error_delta @ layer.weights.T
            print('layer.errror.shape', layer.error.shape)
            layer.error_delta = layer.error @ layer.act(layer.output, der=True)

            print('weights, trans trainX, error_delta', layer.weights.shape, trainX.T.shape, layer.error_delta.shape)
            layer.weights += self.lr * trainX.T @ layer.error_delta

            error_delta = layer.error_delta

    def training_phase(self, trainX, trainY):
        step = BATCH_SIZE
        # index used to
        i = 0
        for j in range(BATCH_SIZE, len(trainX), step):
            x, Y = trainX[i:j], trainY[i:j]
            outputs = self.forward_pass(x)
            self.backwards_pass(x, outputs, Y)
            i = j
        # Last mini batch
        x, Y = trainX[j:-1], trainY[j:-1]
        outputs = self.forward_pass(x)
        self.backwards_pass(x, outputs, Y)

    def validation_phase(self, valX, valY):
        pass

    def test_phase(self, testX, testY):
        predicted, _, loss = self.forward_pass(testX, testY)
        return loss

    def run(self):
        datagen = DataGenerator()
        trainX, trainY, validationX, validationY, testX, testY = datagen.make_cases(NUMBER_OF_CASES, SPLIT)
        # datagen.display_N_pictures(10)
        self.training_phase(trainX, trainY)
        self.validation_phase(validationX, validationY)
        self.test_phase(testX, testY)


if __name__ == '__main__':
    """ network = Network()
    trainX, trainY, *_ = DataGenerator.make_cases(NUMBER_OF_CASES, SPLIT)
    x, _ = network.forward_pass(trainX)
    x = Network.softmax(x)
    summ = network.loss_func(x, trainY)
    print(summ)
    predictions = np.array([0.9, 0.8, 1.1, 1])
    targets = np.array([1, 1, 1, 1])
    c = Network.cross_entropy(predictions, targets)
    print(c) """
    network = Network()
    network.run()
