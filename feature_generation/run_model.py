from tensorflow.contrib.rnn import LSTMCell, GRUCell
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from importlib import reload
from sys import stdin, stdout, exit
from os import path

from new_model import Seq2SeqModel, train_on_copy_task, get_vector_representations
from process_data import import_data

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
                             reverse=args.reverse_traces)

        session.run(tf.global_variables_initializer())

        loss_track = train_on_copy_task(session, model, data,
                           batch_size=args.batch_size,
                           batches_in_epoch=100,
                           max_time_diff=args.max_time_diff,
                           verbose=True,
                           in_memory=in_memory)

        plt.plot(loss_track)

def main(_):
    cache_data, labels = None, None
    dirname, _ = path.split(path.abspath(__file__))

    try:
        data_dir = dirname + '/../data/cells'
        cache_data = import_data(data_dir=data_dir, in_memory=False, extension=args.extension)

        stdout.write("Training on data...\n")
        run_model(cache_data, in_memory=False)
        stdout.write("Finished running model.")

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

    print(args)

    tf.app.run()
