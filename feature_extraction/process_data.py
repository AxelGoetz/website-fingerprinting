"""
Pulls all of the data in the ../data directory into memory
"""

from os import scandir, path
from sys import stdout, exit

import numpy as np

# CONSTANTS
dirname, _ = path.split(path.abspath(__file__))
DATA_DIR = dirname + '/../data/test-cells'
UNKNOWN_WEBPAGE = -1
TRACE_DELIMITER = '\t'

def update_progess(total, current):
    """Prints a percentage of how far the process currently is"""
    stdout.write("{:.2f} %\r".format((current/total) * 100))
    stdout.flush()

def read_cell_file(f):
    """
    For a file, reads its contents and returns them in the appropriate format

    @param f is a `posix.DirEntry` object
    @return a list of (size, incoming pairs)
    """
    contents = []
    with open(f.path, 'r') as open_file:
        for line in open_file:
            line = line[:-1] # Get rid of newline

            split = line.split(TRACE_DELIMITER)
            split[0] = float(split[0])
            split[1] = int(split[1])

            contents.append(split)

    return contents

def import_data(data_dir=DATA_DIR):
    """
    Reads all of the files in the `data_dir` and returns all of the contents in a variable.

    @param data_dir is a string with the name of the data directory
    @return data is a tuple with the following format: ([[size, incoming]], [webpage_label])
        where incoming is 1 is outgoing and -1 is incoming
    """
    stdout.write("Starting data import\n")
    data = []
    labels = []

    total_files = len([f for f in scandir(data_dir) if f.is_file()])

    for i, f in enumerate(scandir(data_dir)):
        if f.is_file():
            name = f.name
            name = name.replace(".cell", "")

            name_split = name.split('-') # Contains [webpage, index (OPTIONAL)]

            webpage_label = None
            # If the length is 1, classify as unknown webpage
            if len(name_split) == 1:
                webpage_label = UNKNOWN_WEBPAGE
            else:
                webpage_label = int(name_split[0])

            labels.append(webpage_label)

            data.append(read_cell_file(f))

            if i % 100 == 0:
                update_progess(total_files, i)

    stdout.write("Finished importing data\n")
    return (data, labels)

if __name__ == '__main__':
    import_data()
