from process_data import import_data
# import model
import temp

import numpy as np

from importlib import reload
from sys import stdin, stdout

cache_data, sequence_lengths, labels = None, None, None
if __name__ == '__main__':
    stdout.write("To re-run the model, press enter and to exit press CTRL-C\n")
    try:
        while True:
            if cache_data is None:
                cache_data, labels  = import_data()

            stdout.write("Training on data...\n")
            temp.temp(cache_data, sequence_lengths, labels)
            stdout.write("Finished running model.")

            # Wait for enter
            stdin.readline()

            # Reload the source code
            # reload(model)
            reload(temp)

    except KeyboardInterrupt:
        stdout.write("Interrupted, this might take a while...\n")
        exit(0)
