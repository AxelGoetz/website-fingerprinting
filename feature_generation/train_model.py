from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.contrib.rnn import LSTMCell, GRUCell
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from sys import stdin, stdout, exit
from os import path

from seq2seq import Seq2SeqModel, train_on_copy_task, get_vector_representations
from process_data import import_data, store_data, split_mon_unmon

import helpers

TEST_SIZE = 0.7

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
                             learning_rate=args.learning_rate)

        session.run(tf.global_variables_initializer())

        loss_track = train_on_copy_task(session, model, data,
                           batch_size=args.batch_size,
                           batches_in_epoch=100,
                           max_time_diff=args.max_time_diff,
                           verbose=True)

        # plt.plot(loss_track)

def main(_):
    paths, labels = None, None
    dirname, _ = path.split(path.abspath(__file__))

    try:
        data_dir = dirname + '/../data/cells'
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

    parser = argparse.ArgumentParser(description="Runs a seq2seq model that learns to extract a fixed-length vector representation of data in the `data` directory")

    parser.add_argument('--batch_size', metavar='', type=int, help="Batch size (default 100)", default=100)
    parser.add_argument('--bidirectional', action='store_true', help="Whether to use the a bidirectional encoder or not (default not bidirectional)")
    parser.add_argument('--encoder_hidden_states', metavar='', type=int, help="Amount of encoder hidden states (output vector will be 2 * encoder_hidden_states)", default=120)
    parser.add_argument('--cell_type', metavar='', help="The cell type used (LSTM or GRU)", default='LSTM')
    parser.add_argument('--reverse_traces', action='store_true', help="If you reverse the traces for training. Do not use if bidirectional is true (default not on)")
    parser.add_argument('--max_time_diff', metavar='', type=float, help="The time at which you stop considering a packet important (default infinity)", default=float('inf'))
    parser.add_argument('--extension', metavar='', help="Extension of the cell files", default=".cell")
    parser.add_argument('--learning_rate', metavar='', type=float, help="Learning rate (default 0.000002)", default=0.000002)


    global args
    args = parser.parse_args()
    args.decoder_hidden_states = 2 * args.encoder_hidden_states if args.bidirectional else args.encoder_hidden_states

    if args.cell_type == 'LSTM':
        args.cell = LSTMCell
    elif args.cell_type == 'GRU':
        args.cell = GRUCell
    else:
        print("Cell type not found, try again (LSTM or GRU)")
        exit(0)

    tf.app.run()
