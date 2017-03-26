"""
This script needs to be run after training the model with `train_model.py`.
Stores the automatically generated features in `../data/af_cells`.
"""
import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import LSTMCell, GRUCell
from os import scandir, makedirs, path as ospath
from sys import stdout, path, exit

from new_model import Seq2SeqModel, get_vector_representations

dirname, _ = ospath.split(ospath.abspath(__file__))
DATA_DIR = dirname + '/../data/cells'

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
    Runs a seq2seq model.

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
        model = Seq2SeqModel(encoder_cell=args.cell(args.encoder_hidden_states),
                             decoder_cell=args.cell(args.decoder_hidden_states),
                             seq_width=2,
                             batch_size=args.batch_size,
                             bidirectional=args.bidirectional,
                             reverse=args.reverse_traces,
                             learning_rate=args.learning_rate,
                             saved_graph=args.graph_file,
                             sess=session)

        session.run(tf.global_variables_initializer())

        get_vector_representations(session, model, data, DATA_DIR + '/../af_cells',
                               batch_size=args.batch_size,
                               max_batches=None,
                               batches_in_epoch=100,
                               max_time_diff=args.max_time_diff)

def main(_):
    paths, labels = [], []
    with open(DATA_DIR + '/../X_test', 'r') as f:
        paths = f.readline().split(' ')

    with open(DATA_DIR + '/../y_test', 'r') as f:
        labels = f.readline().split(' ')


    create_dir_if_not_exists(DATA_DIR + '/../af_cells')

    run_model(paths)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Runs a seq2seq model that learns to extract a fixed-length vector representation of data in the `data` directory")

    parser.add_argument('--batch_size', metavar='', type=int, help="Batch size (default 100)", default=100)
    parser.add_argument('--bidirectional', action='store_true', help="Whether to use the a bidirectional encoder or not (default not bidirectional)")
    parser.add_argument('--encoder_hidden_states', metavar='', type=int, help="Amount of encoder hidden states (output vector will be 2 * encoder_hidden_states)", default=120)
    parser.add_argument('--cell_type', metavar='', help="The cell type used (LSTM or GRU)", default='LSTM')
    parser.add_argument('--reverse_traces', action='store_true', help="If you reverse the traces for training. Do not use if bidirectional is true (default not on)")
    parser.add_argument('--max_time_diff', metavar='', type=float, help="The time at which you stop considering a packet important (default infinity)", default=float('inf'))
    parser.add_argument('--extension', metavar='', help="Extension of the cell files", default=".cell")
    parser.add_argument('--graph_file', metavar='', help="File name of the graph stores (default seq2seq_model)", default='seq2seq_model')
    parser.add_argument('--learning_rate', metavar='', type=float, help="Learning rate (default 0.000002)", default=0.000002)

    global args
    args = parser.parse_args()

    args.graph_file = dirname + '/../' + args.graph_file

    args.decoder_hidden_states = 2 * args.encoder_hidden_states if args.bidirectional else args.encoder_hidden_states

    if args.cell_type == 'LSTM':
        args.cell = LSTMCell
    elif args.cell_type == 'GRU':
        args.cell = GRUCell
    else:
        print("Cell type not found, try again (LSTM or GRU)")
        exit(0)

    tf.app.run()
