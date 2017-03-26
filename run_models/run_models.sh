#!/bin/bash

# Normal
echo "Training with hand-picked features"
echo "kNN"
python run_models/run_model.py --model "kNN"

echo "Random forest"
python run_models/run_model.py --model "random_forest"

echo "SVC 1"
python run_models/run_model.py --model "svc1"

echo "SVC 2"
python run_models/run_model.py --model "svc2"


# Automatically generated
echo "Now training with automatically generated features:"
echo "kNN"
python run_models/run_model.py --model "kNN" --handpicked

echo "Random forest"
python run_models/run_model.py --model "random_forest" --handpicked

echo "SVC 1"
python run_models/run_model.py --model "svc1" --handpicked

echo "SVC 2"
python run_models/run_model.py --model "svc2" --handpicked

# Binary classification
echo "Now trying as a binary classification task"
echo "kNN"
python run_models/run_model.py --model "kNN" --is_multiclass

echo "Random forest"
python run_models/run_model.py --model "random_forest" --is_multiclass

echo "SVC 1"
python run_models/run_model.py --model "svc1" --is_multiclass

echo "SVC 2"
python run_models/run_model.py --model "svc2" --is_multiclass

# Automatically generated
echo "Binary classification with automatically generated features"
echo "kNN"
python run_models/run_model.py --model "kNN" --handpicked --is_multiclass

echo "Random forest"
python run_models/run_model.py --model "random_forest" --handpicked --is_multiclass

echo "SVC 1"
python run_models/run_model.py --model "svc1" --handpicked --is_multiclass

echo "SVC 2"
python run_models/run_model.py --model "svc2" --handpicked --is_multiclass
