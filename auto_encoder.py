import tensorflow as tf
import numpy as np
from dtw import *
# from tf_dtw import tf_dtw

class Encoder(tf.keras.Model):

    def __init__(self, input_shape, code_size, filters, kernel_sizes):
        super(Encoder, self).__init__()
        assert len(filters) == len(kernel_sizes)
        assert len(input_shape) == 2 # (x, y), x = # of samples, y = # of vars
        #self.input_shape = input_shape
        self.code_size = code_size

        self.convs = []
        self.norms = []
        output_len = input_shape[0]
        output_channels = input_shape[1]

        for f, k in zip(filters, kernel_sizes):
            l = tf.keras.layers.Conv1D(f, k, activation="tanh")
            b = tf.keras.layers.BatchNormalization()
            self.convs.append(l)
            self.norms.append(b)
            output_len = output_len - (k-1)
            output_channels = f

        self.last_kernel_shape = (output_len, output_channels)
        self.flatten = tf.keras.layers.Flatten()
        self.out = tf.keras.layers.Dense(code_size)

    def call(self, inputs, training=False):

        x = self.convs[0](inputs)
        x = self.norms[0](x)
        for conv, norm in zip(self.convs[1:], self.norms[1:]):
            x = conv(x)
            x = norm(x, training=training)
        assert x.shape[1:] == self.last_kernel_shape
        #print(x.shape)
        x = self.flatten(x)

        x = self.out(x)
        return x

class Decoder(tf.keras.Model):

    def __init__(self, code_size, last_kernel_shape, output_shape, filters, kernel_sizes):
        super(Decoder, self).__init__()

        assert len(last_kernel_shape) == 2
        assert len(output_shape) == 2 # (x, y) x = # of samples, y = samples n variables

        self.code_size = code_size
        self.last_kernel_shape = last_kernel_shape
        self.expected_output_shape = output_shape

        flat_len = last_kernel_shape[0] * last_kernel_shape[1]

        self.expand = tf.keras.layers.Dense(flat_len)
        self.reshape = tf.keras.layers.Reshape(last_kernel_shape)

        self.convs = []
        self.norms = []

        for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
            l = tf.keras.layers.Conv1DTranspose(f, k)
            b = tf.keras.layers.BatchNormalization()
            self.convs.append(l)
            self.norms.append(b)

    def call(self, inputs, training=False):
        x = self.expand(inputs)
        x = self.reshape(x)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x, training=training)
            x = conv(x)
        assert self.expected_output_shape == x.shape[1:]
        return x

_optimizer = tf.keras.optimizers.Nadam(learning_rate=0.00015)
_mse_loss = tf.keras.losses.MeanSquaredError()

class AutoEncoder:
    def __init__(self, **kwargs):

        input_shape     = kwargs["input_shape"]
        code_size       = kwargs["code_size"]
        filters         = kwargs["filters"]
        kernel_sizes    = kwargs["kernel_sizes"]

        if "loss" in kwargs:
            loss        = kwargs["loss"]
        else:
            loss        = _mse_loss

        if "optimizer" in kwargs:
            optimizer   = kwargs["optimizer"]
        else:
            optimizer   = _optimizer

        self.encode = Encoder(input_shape, code_size, filters, kernel_sizes)

        decoder_filters = list(filters[:len(filters)-1])
        decoder_filters.append(input_shape[1])
        last_kernel_shape = self.encode.last_kernel_shape

        self.decode = Decoder(code_size, last_kernel_shape, input_shape, decoder_filters,
                kernel_sizes)

        self.loss = loss
        self.optimizer = optimizer

def similarity_funcs(input):
    # 50 x 150 x 1
    # 50 x 150
    print("1")
    remove_last_dim = tf.reshape(input, [input.shape[0], input.shape[1]])
    # 1225 x 150
    total_distance = 0
    for i in range(remove_last_dim.shape[0]):
        for j in range(i + 1, remove_last_dim.shape[0]):
            total_distance += tf_dtw(remove_last_dim[i], remove_last_dim[j])
    return total_distance / (remove_last_dim.shape[0] * (remove_last_dim.shape[0] - 1) / 2)



def eu_code_func(codes):
    remove_last_dim = tf.reshape(codes, [codes.shape[0], codes.shape[1]])
    # 1225 x 150
    total_distance = 0
    for i in range(remove_last_dim.shape[0]):
        for j in range(i + 1, remove_last_dim.shape[0]):
            total_distance += tf.linalg.norm(remove_last_dim[i] - remove_last_dim[j])
    return total_distance / (remove_last_dim.shape[0] * (remove_last_dim.shape[0] - 1) / 2)


def tf_dtw(s, t):
    s = tf.cast(s, tf.float32)
    t = tf.cast(t, tf.float32)
    print("sb")
    n = len(s)
    m = len(t)
    # DTW = tf.fill([len(s)+1,len(t)+1], tf.tensor(np.inf))
    # DTW = [[tf.Variable(np.inf) for _ in range(len(t)+1)] for _ in range(len(s)+1)]
    DTW = np.full([n + 1, m + 1], tf.Variable(np.inf))
    DTW[0][0].assign(0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s[i - 1] - t[j - 1])
            last_min = tf.math.minimum(tf.math.minimum(DTW[i - 1][j], DTW[i][j - 1]), DTW[i - 1][j - 1])
            DTW[i][j].assign(tf.math.add(cost, last_min))
    return DTW[-1][-1]


@tf.function
def train_step(input, auto_encoder, optimizer=_optimizer, loss = _mse_loss):
    # print("!!!!!!!!!!!")
    with tf.GradientTape() as tape:
        # print("!!!!!!!!!!!")

        dtw_input = similarity_funcs(input)
        # dtw_input = tf_dtw(input, len(input))
        print(dtw_input)
        codes = auto_encoder.encode(input, training=True)
        eu_code = eu_code_func(codes)
        # print(eu_code)
        decodes = auto_encoder.decode(codes, training=True)
        reconstruction_loss = loss(input, decodes)
        # print(reconstruction_loss)
        similarity_loss = loss(eu_code, dtw_input)
        # print(similarity_loss)
        loss = (reconstruction_loss + similarity_loss) / 2
        # loss = (reconstruction_loss + eu_code) / 2
        trainables = auto_encoder.encode.trainable_variables + auto_encoder.decode.trainable_variables
    gradients = tape.gradient(loss, trainables)
    optimizer.apply_gradients(zip(gradients, trainables))
    return loss

