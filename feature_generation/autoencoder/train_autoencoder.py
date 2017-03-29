from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
import numpy as np

from sys import stdin, stdout, exit, path
from os import path as ospath

from autoencoder import AutoEncoder, train_on_copy_task

# Add parent dir to path
path.append(ospath.dirname(ospath.dirname(ospath.abspath(__file__))))

from process_data import import_data, store_data, split_mon_unmon
import helpers

TEST_SIZE = 0.7

def run_model(data, in_memory=False):
    """
    Runs the autoencoder model.

    @param data is
        if in_memory == True:
            ([[size, incoming]], [webpage_label])
        else:
            A list of paths
    """
    tf.reset_default_graph()
    tf.set_random_seed(123)

    # Only print small part of array
    np.set_printoptions(threshold=10)

    with tf.Session() as session:

        model = AutoEncoder(args.layers, args.batch_size,
                            activation_func=args.activation_func,
                            learning_rate=args.learning_rate)

        session.run(tf.global_variables_initializer())

        loss_track = train_on_copy_task(session, model, data,
                           batch_size=args.batch_size,
                           batches_in_epoch=100,
                           verbose=False)


def main(_):
    paths, labels = None, None
    dirname, _ = ospath.split(ospath.abspath(__file__))

    try:
        data_dir = dirname + '/../../data/cells'
        paths, labels = import_data(data_dir=data_dir, in_memory=False, extension=args.extension)

        monitored_data, monitored_label, unmonitored_data = split_mon_unmon(paths, labels)
        monitored_data, monitored_label, unmonitored_data = np.array(monitored_data), np.array(monitored_label), np.array(unmonitored_data)

        helpers.shuffle_data(unmonitored_data)
        unmon_train, unmon_test = unmonitored_data[:int((1 - TEST_SIZE) * len(unmonitored_data))], unmonitored_data[int((1 - TEST_SIZE) * len(unmonitored_data)):]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=123)
        sss.get_n_splits(monitored_data, monitored_label)

        for train_index, test_index in sss.split(monitored_data, monitored_label):
            X_train, X_test = monitored_data[train_index], monitored_data[test_index]
            y_train, y_test = monitored_label[train_index], monitored_label[test_index]

            X_train = np.append(X_train, unmon_train)
            X_test = np.append(X_test, unmon_test)

            y_train = np.append(y_train, [-1] * len(unmon_train))
            y_test = np.append(y_test, [-1] * len(unmon_train))

            store_data(X_test, 'X_test')
            store_data(y_test, 'y_test')

            stdout.write("Training on data...\n")
            run_model(X_train, in_memory=False)
            stdout.write("Finished running model.")
            break

    except KeyboardInterrupt:
        stdout.write("Interrupted, this might take a while...\n")
        exit(0)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Runs a autoencoder model that learns to extract a fixed-length vector representation of data in the `data` directory")

    parser.add_argument('--batch_size', metavar='', type=int, help="Batch size (default 100)", default=100)
    parser.add_argument('--extension', metavar='', help="Extension of the cell files", default=".cell")
    parser.add_argument('--learning_rate', metavar='', type=float, help="Learning rate (default 0.000002)", default=0.0001)
    parser.add_argument('--activation_func', metavar='', help="Which activation function to use (sigmoid, relu or atan)", default='sigmoid')
    parser.add_argument('--layers', metavar='', nargs='+', type=int, help="List of how big the layers in the encoder are (e.g: 1000 800 600)", default=[1500, 500, 100])

    global args
    args = parser.parse_args()

    if args.activation_func == 'sigmoid':
        args.activation_func = tf.nn.sigmoid
    elif args.activation_func == 'relu':
        args.activation_func = tf.nn.relu
    elif args.activation_func == "atan":
        args.activation_func = tf.tanh
    else:
        print("Activation function not found, try again (sigmoid, rely or atan)")
        exit(0)

    tf.app.run()
