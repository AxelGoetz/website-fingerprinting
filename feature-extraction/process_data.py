"""
Pulls all of the data in the ../data directory into memory
"""

from os import scandir
from sys import stdout, exit

# CONSTANTS
DATA_DIR = '../data/cells'
UNKNOWN_WEBPAGE = -1
TRACE_DELIMITER = '\t'

def update_progess(total, current):
    """Prints a percentage of how far the process currently is"""
    stdout.write("{:.2f} %\r".format((current/total) * 100))
    stdout.flush()

def add_key_to_dict(dictionary, key, init):
    """
    Given a dictionary, checks if the key already exists.
    If not, it initializes the entry with an init value.
    """
    if key not in dictionary:
        dictionary[key] = init

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
    @return data is a dictionary with the following format: {website_index: [size, incoming]}
        where incoming is 1 is outgoing and -1 is incoming
    """
    stdout.write("Starting data import\n")
    data = {}

    total_files = len([f for f in scandir(data_dir) if f.is_file()])

    for i, f in enumerate(scandir(data_dir)):
        if f.is_file():
            name = f.name
            name = name.replace(".cell", "")

            name_split = name.split('-') # Contains [webpage, index (OPTIONAL)]

            key = None
            # If the length is 1, classify as unknown webpage
            if len(name_split) == 1:
                key = UNKNOWN_WEBPAGE
            else:
                key = int(name_split[0])

            add_key_to_dict(data, key, [])
            data[key].append(read_cell_file(f))

            if i % 100 == 0:
                update_progess(total_files, i)

    stdout.write("Finished importing data\n")
    return data

if __name__ == '__main__':
    import_data()
