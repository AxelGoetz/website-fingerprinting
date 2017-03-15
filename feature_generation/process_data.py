"""
Pulls all of the data in the ../data directory into memory
"""

from os import scandir, path
from sys import stdout, exit

from helpers import read_cell_file

import numpy as np

# CONSTANTS
dirname, _ = path.split(path.abspath(__file__))
DATA_DIR = dirname + '/../data/cells'
UNKNOWN_WEBPAGE = -1

def update_progess(total, current):
    """Prints a percentage of how far the process currently is"""
    stdout.write("{:.2f} %\r".format((current/total) * 100))
    stdout.flush()

def pull_data_in_memory(data_dir, extension=".cell"):
    """
    Gets the content of all the data in the `data_dir` directory in memory.

    @return is a tuple with the following format: ([[size, incoming]], [webpage_label])
        where incoming is 1 is outgoing and -1 is incoming
    """
    data = []
    labels = []

    total_files = len([f for f in scandir(data_dir) if f.is_file()])

    for i, f in enumerate(scandir(data_dir)):
        if f.is_file() and f.name[-len(extension):] == extension:
            name = f.name
            name = name.replace(extension, "")

            name_split = name.split('-') # Contains [webpage, index (OPTIONAL)]

            webpage_label = None
            # If the length is 1, classify as unknown webpage
            if len(name_split) == 1:
                webpage_label = UNKNOWN_WEBPAGE
            else:
                webpage_label = int(name_split[0])

            labels.append(webpage_label)

            data.append(read_cell_file(f.path))

            if i % 100 == 0:
                update_progess(total_files, i)

    stdout.write("Finished importing data\n")
    return (data, labels)

def get_files(data_dir, extension=".cell"):
    """
    Gets the path of all the files in the `data_dir` directory with a `extension` extension.

    @return a tuple of lists (paths, label)
    """
    files = []
    labels = []

    total_files = len([f for f in scandir(data_dir) if f.is_file()])

    for i, f in enumerate(scandir(data_dir)):
        if f.is_file() and f.name[-len(extension):] == extension:
            files.append(f.path)

            name = f.name
            name = name.replace(extension, "")

            name_split = name.split('-') # Contains [webpage, index (OPTIONAL)]

            webpage_label = None
            # If the length is 1, classify as unknown webpage
            if len(name_split) == 1:
                webpage_label = UNKNOWN_WEBPAGE
            else:
                webpage_label = int(name_split[0])

            labels.append(webpage_label)

        if i % 100 == 0:
            update_progess(total_files, i)

    stdout.write("Finished importing data\n")
    return (files, labels)

def import_data(data_dir=DATA_DIR, in_memory=True, extension=".cell"):
    """
    Reads all of the files in the `data_dir` and returns all of the contents in a variable.

    @param data_dir is a string with the name of the data directory
    @param in_memory is a boolean value. If true, it pulls all the data into memory

    @return
        if in_memory == True:
            is a tuple with the following format: ([[size, incoming]], [webpage_label])
                where outgoing is 1 is incoming and -1
        else:
            a tuple with the following format: ([paths], [webpage_label])

    """
    stdout.write("Starting data import\n")

    if in_memory:
        return pull_data_in_memory(data_dir, extension)
    else:
        return get_files(data_dir, extension)

def store_data(data, file_name):
    """
    Takes a 1D array and stores in in the `../data/file_name` directory
    using a space-separated fashion
    """
    with open(DATA_DIR + '/../' + file_name, 'w') as f:
        string = " ".join([str(x) for x in list(data)])
        f.write(string)

if __name__ == '__main__':
    import_data()
