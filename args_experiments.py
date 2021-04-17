from comet_ml import Experiment
import sys
# sys.path.append("/Users/jiang/Desktop/2270/cs227_final_project/data/data") # path to this repository
import py_ts_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

import datetime
from auto_encoder import AutoEncoder, train_step, LAMBDA, Encoder, train_step_new, train_step_enc_sep
from preprocess import augmentation
from tqdm import tqdm
from sample_evaluation_funcs import *

import argparse
import os


EPOCHS = 50
BATCH = 50

hyperparams = {
	# for log purpose only
	"model_type": "CNN",
	"epochs": EPOCHS,
	"batch_size": BATCH
}
# experiment.log_parameters(hyperparams)


def min_max(data, feature_range=(0, 1)):
    """
    implements min-max scaler
    """
    min_v = feature_range[0]
    max_v = feature_range[1]
    max_vals = data.max(axis=1)[:, None, :]
    min_vals = data.min(axis=1)[:, None, :]
    X_std = (data - min_vals) / (max_vals - min_vals)
    return X_std * (max_v - min_v) + min_v

def normalize(data):
    """
    Z-normalize data with shape (x, y, z)
    x = # of timeseries
    y = len of each timeseries
    z = vars in each timeseres
    
    s.t. each array in [., :, .] (i.e. each timeseries variable)
    is zero-mean and unit stddev
    """
    sz, l, d = data.shape
    means = np.broadcast_to(np.mean(data, axis=1)[:, None, :], (sz, l, d))
    stddev = np.broadcast_to(np.std(data, axis=1)[:, None, :], (sz, l, d)) 
    return (data - means)/stddev

def train(ae, encoder, EPOCHS, train_dataset, suffix, experiment, lambda_p, args):
    print("training with lambda = ", lambda_p)
    loss_history = []
    sim_history = []
    re_history = []

    with experiment.train():
        for epoch in range(EPOCHS):
            total_loss = 0
            total_sim = 0
            total_re = 0
        #     for i, (input, _) in enumerate(train_dataset):
            for (input, _) in tqdm(train_dataset):
                # loss, similarity_loss, reconstruction_loss = train_step_new(input, ae, encoder, lambda_p=lambda_p)
                loss = None
                similarity_loss = None
                reconstruction_loss = None
                if args.auto:
                    loss, similarity_loss, reconstruction_loss = train_step(input, ae, lambda_p=lambda_p)
                elif args.encauto:
                    loss, similarity_loss, reconstruction_loss = train_step_new(input, ae, encoder, lambda_p=lambda_p)
                elif args.seqencauto:
                    loss, similarity_loss, reconstruction_loss = train_step_enc_sep(input, ae, encoder, lambda_p=lambda_p)
                else:
                    raise Exception("model type flag not set")
        #         if i % 100 == 0:
        #             print(loss)
                total_loss += loss
                total_sim += similarity_loss
                total_re += reconstruction_loss
                # break
                
            loss_history.append(total_loss)
            sim_history.append(total_sim)
            re_history.append(total_re)
            print("Epoch {}: {}".format(epoch, total_loss, total_sim, total_re))
        #     break

        # ae.save('mymodel')
        # suffix = "hahaha"

        plt.plot(loss_history)
        plot_name = "loss_history_{suffix}.png".format(suffix = suffix)
        plt.savefig(plot_name)
        plt.clf()
        experiment.log_image(plot_name)

        plot_name = "sim_history_{suffix}.png".format(suffix = suffix)
        plt.plot(sim_history)
        plt.savefig(plot_name)
        plt.clf()
        experiment.log_image(plot_name)

        plot_name = "re_history_{suffix}.png".format(suffix = suffix)
        plt.plot(re_history)
        plt.savefig(plot_name)
        plt.clf()
        experiment.log_image(plot_name)


def recon_eval(ae, X_test, suffix, experiment):
    # evaluate recon
    code_test = ae.encode(X_test)
    decoded_test = ae.decode(code_test)


    plt.plot(X_test[0])
    plt.plot(decoded_test[0])
    # plt.show()
    plot_name = "recon_eval_{suffix}.png".format(suffix = suffix)
    plt.savefig(plot_name)
    plt.clf()
    experiment.log_image(plot_name)

    losses = []
    for ground, predict in zip(X_test, decoded_test):
        losses.append(np.linalg.norm(ground - predict))
    print("Mean L2 distance: {}".format(np.array(losses).mean()))
    experiment.log_metric("Mean L2 distance " + suffix, np.array(losses).mean())

    return code_test

def nn_dist(x, y):
    """
    Sample distance metric, here, using only Euclidean distance
    """
    x = x.reshape((45, 2))
    y = y.reshape((45, 2))
    return np.linalg.norm(x-y)


def sim_eval(X_test, code_test, suffix, experiment):
    # evaluate similarity

    # print("000")
    nn_x_test = X_test.reshape((-1, 90))
    baseline_nn = NearestNeighbors(n_neighbors=10, metric=nn_dist).fit(nn_x_test)
    code_nn = NearestNeighbors(n_neighbors=10).fit(code_test)
    # print("111")

    # For each item in the test data, find its 11 nearest neighbors in that dataset (the nn is itself)
    baseline_11nn = baseline_nn.kneighbors(nn_x_test, 11, return_distance=False)
    code_11nn     = code_nn.kneighbors(code_test, 11, return_distance=False)

    # On average, how many common items are in the 10nn?
    result = []
    for b, c in zip(baseline_11nn, code_11nn):
        # remove the first nn (itself)
        b = set(b[1:])
        c = set(c[1:])
        result.append(len(b.intersection(c)))
        # print(result)
    print(np.array(result).mean())
    experiment.log_metric("sim_eval "+ suffix, np.array(result).mean())

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--auto", action="store_true",
                            help="autoencoder")
    parser.add_argument("-e", "--encauto", action="store_true",
                            help="encoder + autoencoder")
    parser.add_argument("-s", "--seqencauto", action="store_true",
                            help="Encoder(sim) + autoencoder(rec)")
    parser.add_argument("l")
    parser.add_argument("filter1")
    parser.add_argument("filter2")
    parser.add_argument("filter3")
    parser.add_argument("epoch")
    parser.add_argument("batch")
    args = parser.parse_args()

    m_type = None
    if args.auto:
        m_type = "autoencoder"
    elif args.encauto:
        m_type = "encoder_autoencoder"
    elif args.seqencauto:
        m_type = "Encoder_sim_autoencoder_rec"
    else:
        raise Exception("model type flag not set")

    model_type_log = "{m_type} lambda={l} filter=[{filter1}, {filter2}, {filter3}] epoch={epoch} batch={batch}".format(
        m_type = m_type,
        l = args.l,
        filter1 = args.filter1,
        filter2 = args.filter2,
        filter3 = args.filter3,
        epoch = args.epoch,
        batch = args.batch)

    filters = [int(args.filter1), int(args.filter2), int(args.filter3)]
    BATCH = int(args.batch)
    EPOCHS = int(args.epoch)
    lam = float(args.l)

    hyperparams["model_type"] = model_type_log
    hyperparams["epochs"] = EPOCHS
    hyperparams["batch_size"] = BATCH

    experiment = Experiment(log_code=False)
    experiment.log_parameters(LAMBDA)
    experiment.log_parameters(hyperparams)

    dataset_name = "GunPoint"

    X_train, y_train, X_test, y_test, info = py_ts_data.load_data(dataset_name, variables_as_channels=True)
    print("Dataset shape: Train: {}, Test: {}".format(X_train.shape, X_test.shape))


    print(X_train.shape, y_train.shape)
    X_train, y_train = augmentation(X_train, y_train)
    # X_test, y_test = augmentation(X_test, y_test)
    print(X_train.shape, y_train.shape)
    # fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    # axs[0].plot(X_train[200])
    X_train = min_max(X_train, feature_range=(-1, 1))
    # axs[1].plot(X_train[200])
    X_test = min_max(X_test, feature_range=(-1, 1))
    # plt.show()


    kwargs = {
        "input_shape": (X_train.shape[1], X_train.shape[2]),
        # "filters": [32, 64, 128],
        # "filters": [128, 64, 32],
        "filters": filters,
        # "filters": [32, 32, 32],
        # "filters": [32, 32, 16],
        "kernel_sizes": [5, 5, 5],
        "code_size": 16,
    }


    # lambda_to_test = [0.9, ]
    # for l in range(1, 10):
    #     lam = l / 10

    # lam = 0.99
    ae = AutoEncoder(**kwargs)

    input_shape = kwargs["input_shape"]
    code_size = kwargs["code_size"]
    filters = kwargs["filters"]
    kernel_sizes = kwargs["kernel_sizes"]
    encoder = Encoder(input_shape, code_size, filters, kernel_sizes)
    # training

    SHUFFLE_BUFFER = 100
    K = len(set(y_train))

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH)

    suffix = "lam={lam}".format(lam=lam)
    train(ae, encoder, EPOCHS, train_dataset, suffix, experiment, lam, args)


    code_test = recon_eval(ae, X_test, suffix, experiment)
    sim_eval(X_test, code_test, suffix, experiment)

    cwd = os.path.abspath(os.getcwd())
    metadata = "lambda_{l}_filter_{filter1}{filter2}{filter3}_epoch_{epoch}_batch_{batch}".format(
        l = args.l,
        filter1 = args.filter1,
        filter2 = args.filter2,
        filter3 = args.filter3,
        epoch = args.epoch,
        batch = args.batch)
    encoder_path = os.path.join(cwd, m_type, dataset_name, metadata, "encoder")
    ae_encoder_path = os.path.join(cwd, m_type, dataset_name, metadata, "auto_encoder")
    ae_decoder_path = os.path.join(cwd, m_type, dataset_name, metadata, "decoder")

    if not args.auto:
        encoder.save(encoder_path)
    ae.encode.save(ae_encoder_path)
    ae.decode.save(ae_decoder_path)
    sample_evaluation(ae.encode, ae.encode, ae.decode, experiment, suffix, DATA = dataset_name)

if __name__ == '__main__':
    main()

