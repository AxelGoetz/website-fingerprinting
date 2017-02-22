from tensorflow.contrib.rnn import LSTMCell, GRUCell
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from importlib import reload
from sys import stdin, stdout
from os import path

from new_model import Seq2SeqModel, train_on_copy_task
from process_data import import_data

def run_model(data, in_memory=True):
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

    batch_size = 10
    bidirectional = False

    encoder_hidden_states = 120
    decoder_hidden_states = 2 * encoder_hidden_states if bidirectional else encoder_hidden_states

    cell = LSTMCell

    reverse_traces = False

    # Responsible for shortening the cells
    max_time_diff = 2

    with tf.Session() as session:

        # with bidirectional encoder, decoder state size should be
        # 2x encoder state size
        model = Seq2SeqModel(encoder_cell=cell(encoder_hidden_states),
                             decoder_cell=cell(decoder_hidden_states),
                             seq_width=2,
                             batch_size=batch_size,
                             bidirectional=bidirectional,
                             reverse=reverse_traces)

        session.run(tf.global_variables_initializer())

        loss_track = train_on_copy_task(session, model, data,
                           batch_size=batch_size,
                           batches_in_epoch=100,
                           max_time_diff=max_time_diff,
                           verbose=True,
                           in_memory=in_memory)

        plt.plot(loss_track)

def main(_):
    cache_data, labels = None, None
    dirname, _ = path.split(path.abspath(__file__))

    data_dir = dirname + '/../data/cells'
    in_memory = False

    try:
        if in_memory:
            cache_data, labels = import_data(data_dir=data_dir, in_memory=in_memory)
        else:
            cache_data = import_data(data_dir=data_dir, in_memory=in_memory)

        stdout.write("Training on data...\n")
        run_model(cache_data, in_memory=in_memory)
        stdout.write("Finished running model.")

    except KeyboardInterrupt:
        stdout.write("Interrupted, this might take a while...\n")
        exit(0)

if __name__ == '__main__':
    tf.app.run()
