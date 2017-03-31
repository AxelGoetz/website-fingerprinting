"""
Includes all of the helper functions such as:
- Getting new batches of data
- Converting data to time-major format
"""

import numpy as np
TRACE_DELIMITER = '\t'

from os import path as ospath

def add_EOS(data, sequence_lengths, EOS=-1):
    """
    For the decoder targets, we want to add a EOS symbol after the end of each sequence

    @param data in **batch-major** format
    @param sequence_lengths is a array-like object with the sequence lengths of the elements in the data object
    @param EOS is the end-of-sequence marker
    """
    for i, sequence in enumerate(data):
        sequence[sequence_lengths[i]] = EOS

    return data

def time_major(data):
    """
    Return the data in time-major form (Most often used to turn a single batch into this form)
    *(Assumes the traces are already padded)*

    @param data is an array of sequences
    @return batch contains the batch in **time-major** form [max_time, batch_size] padded with the PAD symbol
    """
    # Swap axis
    return data.swapaxes(0, 1)

def shuffle_data(data, seed=123):
    """
    Shuffles an array-like object, we perform it in here to reset the seed
    and get consistent shuffles.
    """
    np.random.seed(seed)
    np.random.shuffle(data)


def get_batches(data, batch_size=100):
    """
    Divides the data up into batches

    @param data is an array of sequences or an array of paths to files
    @param batch_size is the size of each batch

    @return an iterator with the batch sizes
    """
    # Randomly shuffle the data
    shuffle_data(data)

    data_length = len(data)

    for i in range(0, data_length, batch_size):
        if i + batch_size >= data_length:
            return

        start_index = i
        end_index = i + batch_size

        yield data[start_index: end_index]


def pad_traces(data, extra_padding=1, reverse=False, seq_width=1):

    """
    Pad the traces such that they have the same length

    @param data is an 2D matrix in the following format: [[size, incoming]]
    @param extra_padding determines how much extra padding is added to the elements
    @param reverse is a boolean value that if true, reverses the traces

    @return a tuple where the first element is the padded matrix and the second the lengths
    """
    sequence_lengths = [len(seq) for seq in data]
    batch_size = len(data)

    if reverse:
        for i, seq in enumerate(data):
            data[i] = list(reversed(seq))

    max_sequence_length = max(sequence_lengths) + extra_padding

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length, seq_width], dtype=np.float32)

    for i, seq in enumerate(data):
        for j, element in enumerate(seq):
            inputs_batch_major[i][j] = element

    return inputs_batch_major, sequence_lengths

def read_cell_file(path, max_time_diff=float("inf")):
    """
    For a file, reads its contents and returns them in the appropriate format

    @param path is a path to the file
    @param max_time_diff is the maximum difference between the time in the first cell and the last.
        cuts all the entries after that

    @return a list of (size, incoming pairs)
    """
    contents = []
    with open(path, 'r') as open_file:
        for line in open_file:
            line = line[:-1] # Get rid of newline

            split = line.split(TRACE_DELIMITER)
            split[0] = float(split[0]) # Time
            split[1] = int(split[1]) # Incoming/outcoming

            if len(contents) > 0 and split[0] - contents[0][0] > max_time_diff:
                break

            contents.append(split)

    return contents

def save_object(obj, file_name):
    """
    Used to save an object to a file

    @param obj is the object you want to save
    @param file_name is the name that you are saving the object to *(Should have a `.pkl` extension)*
    """
    from pickle import dump

    with open(file_name, "wb") as f:
        dump(obj, f) # Use the default protocol

def load_object(file_name):
    """
    Loads an object from a file

    @param file_name is the file name of the file *(with the `.pkl` extension)
    """
    from pickle import load

    with open(file_name, "rb") as f:
        return load(f)

def write_to_file(content, save_dir, file_name, new_extension=""):
    """
    Writes the features to a file in a space separated fashion

    @param content is a 1D list of features
    @param save_dir is the absolute path to the directory where we save the traces
    @param file_name is the file name to store the features in
    """
    content = list(map(lambda x: str(x), content))

    with open(save_dir + '/' + file_name + new_extension, 'w') as f:
        f.write(' '.join(content))

def extract_filename_from_path(path, extension):
    """
    Extracts a file name from a given path. (Without the given extension)
    """
    if len(extension) > 0 and extension[0] != '.':
        extension = '.' + extension

    file_name = ospath.basename(path)
    return file_name.replace(extension, "")
