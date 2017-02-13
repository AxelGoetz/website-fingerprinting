"""
Includes all of the helper functions such as:
- Getting new batches of data
- Converting data to time-major format
"""

import numpy as np

def time_major(data):
    """
    Return the data in time-major form (Most often used to turn a single batch into this form)

    @param data is an array of sequences
    @return batch contains the batch in **time-major** form [max_time, batch_size] padded with the PAD symbol
    """
    # Swap axis
    data_time_major = data_time_major.swapaxes(0, 1)

    return (data_time_major, sequence_lengths)

def shuffle_data(data, seed=123):
    """
    Shuffles an array-like object
    """
    np.random.seed(seed)
    np.random.shuffle(data)


def get_batches(data, sequence_lengths, labels, batch_size=100):
    """
    Divides the data up into batches

    @param data is an array of sequences
    @param batch_size is the size of each batch

    @return an iterator with the batch sizes
    """
    # TODO: Perhaps we can take the class distribution in each class into consideration

    # Randomly shuffle the data
    shuffle_data(data)
    shuffle_data(sequence_lengths)
    shuffle_data(labels)

    data_length = len(data)

    for i in range(0, data_length, batch_size):
        start_index = i
        end_index = i + batch_size if i + batch_size < data_length else data_length - 1

        yield (data[start_index: end_index], sequence_lengths[start_index: end_index], labels[start_index, end_index])
