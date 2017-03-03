# Primitive Machine Learning Models

In this folder we create the machine learning models that are used to benchmark our automatic feature generation process.
All of these have been carefully selected according to importance in the website fingerprinting community.
In addition to that, we also selected a variety of different models such that we could compare the effectiveness of automatic feature learning.

The models outlined all need to have two basic functions:
- `fit(X, y)` - trains the model to fit the relationship between matrix x and labels y
- `predict(X)` - Uses the previously trained model to predict the labels

In addition, the constructors also need to have a flag, called `is_multiclass`, which determines if we are training a multiclass or binary classification model.
