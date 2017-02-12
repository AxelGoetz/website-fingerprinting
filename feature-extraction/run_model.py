from process_data import import_data
import model

from importlib import reload
from sys import stdin, stdout

cache_data = None
if __name__ == '__main__':
    stdout.write("To re-run the model, press enter and to exit press CTRL-C\n")
    try:
        while True:
            if cache_data is None:
                cache_data = import_data()

            stdout.write("Training on data...\n")
            # TODO: Perform training etc. Feed the data as an argument

            # Wait for enter
            stdin.readline()

            # Reload the source code
            reload(model)

    except KeyboardInterrupt:
        stdout.write("Interrupted, this might take a while...\n")
        exit(0)
