# ML algorithms project

## A Naive Bayes and a Random Forest algorithm, which predicts if a movie review is positive/negative

The dataset is consisted of revies from IMDB. There are 50000 reviews, 25000 for training and 25000 for testing. I use part of the training data as development data.

The folder consists of: 
- main.py : where every call of algorithm is made
- preprocessing.py : loads the data using Keras Dataset API and preprocess the reviews' texts
- vocab.py : created the vocabulary which the algorithms use, by discarding certain words depending on the superparameters
- evaluaton.py : creates the metrics and the learning curves for the algorithms
- algorithms
  - naive_bayes.py : self explanatory
  - random_forest.py : self explanatory
- RNN
  - rnn_model.py
  - rnn_dataset.py
  - rnn_utils.py
