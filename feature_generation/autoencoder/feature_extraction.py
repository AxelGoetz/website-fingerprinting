"""
This script needs to be run after training the model with `train_model.py`.
Stores the automatically generated features in `../data/af_cells`.
"""
import tensorflow as tf
import numpy as np

from os import scandir, makedirs, path as ospath
from sys import stdout, path, exit

from autoencoder import AutoEncoder, get_vector_representations

dirname, _ = ospath.split(ospath.abspath(__file__))
DATA_DIR = dirname + '/../../data/cells'

def create_dir_if_not_exists(directory):
    """
    Creates a dir if it does not exists, if it exists, it deletes it and creates a new one

    @param directory is a string containing the **absolute** path to the directory
    """
    if ospath.exists(directory):
        from shutil import rmtree
        rmtree(directory)

    makedirs(directory)

def run_model(data, in_memory=False):
    """
    Runs a autoencoder model.

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

        # with bidirectional encoder, decoder state size should be
        # 2x encoder state size
        model = AutoEncoder(args.layers, args.batch_size,
                            activation_func=args.activation_func,
                            learning_rate=args.learning_rate,
                            saved_graph=args.graph_file,
                            sess=session,
                            batch_norm=args.batch_norm)

        model.set_is_training(False)

        # session.run(tf.global_variables_initializer())

        get_vector_representations(session, model, data, DATA_DIR + '/../ae_cells',
                               batch_size=args.batch_size,
                               max_batches=None,
                               batches_in_epoch=100,
                               extension=args.extension)

def main(_):
    paths, labels = [], []
    with open(DATA_DIR + '/../X_test', 'r') as f:
        paths = f.readline().split(' ')

    with open(DATA_DIR + '/../y_test', 'r') as f:
        labels = f.readline().split(' ')

    create_dir_if_not_exists(DATA_DIR + '/../ae_cells')

    run_model(paths)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Runs a autoencoder model that extracts a fixed-length vector representation of data in the `data` directory")

    parser.add_argument('--batch_size', metavar='', type=int, help="Batch size (default 100)", default=100)
    parser.add_argument('--extension', metavar='', help="Extension of the cell files", default=".cell")
    parser.add_argument('--learning_rate', metavar='', type=float, help="Learning rate (default 0.000002)", default=0.0001)
    parser.add_argument('--activation_func', metavar='', help="Which activation function to use (sigmoid, relu or atan)", default='sigmoid')
    parser.add_argument('--layers', metavar='', nargs='+', type=int, help="List of how big the layers in the encoder are (e.g: 1000 800 600)", default=[1500, 500, 100])
    parser.add_argument('--graph_file', metavar='', help="File name of the graph stores (default autoencoder_model)", default='autoencoder_model')
    parser.add_argument('--batch_norm', action="store_true", help="Whether or not to use batch normalization (default False)",)


    global args
    args = parser.parse_args()

    args.graph_file = dirname + '/../../' + args.graph_file

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
