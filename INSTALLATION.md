# Installation

In this project, we use Keras with a Tensorflow backend.

These are a set of simple instructions to get your environment up and running.

First you will need to install a python virtual environment using:
```
pip install virtualenv
```

Make sure you are then in the main directory of this project and run:
```
source venv/bin/activate
```
to activate the virtual environment. Once you are in this environment, you will need to install the appropriate packages by running:
```
pip3 install -r requirements.txt
```

## MacOS instructions
If you are running MacOS, you will also need  to install [homebrew](http://brew.sh/) and use it to install bazel:
```
brew install bazel
```

### GPU Setup

If you plan to use the GPU support, you will also need to run some additional instructions, all of which can be found [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#test-the-tensorflow-installation) and you will need to install the GPU enabled tensorflow instead..
