import os
import glob
import re
from keras.api.datasets import imdb

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def load_data(directory):
    texts = []
    labels = []
    for label, folder in [('pos', 'pos'), ('neg', 'neg')]:
        path = os.path.join(directory, folder, '*.txt')
        for filename in glob.glob(path):
            with open(filename, encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1 if label == 'pos' else 0)
    return texts, labels

def load_imdb_data(num_words=10000, dev_size=10000):
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)
    
    word_index = imdb.get_word_index()
    index_word = {i+3: word for word, i in word_index.items()}
    index_word[0] = "<PAD>"
    index_word[1] = "<START>"
    index_word[2] = "<UNK>"
    
    train_texts = [" ".join([index_word.get(i, "?") for i in review]) for review in train_data]
    test_texts = [" ".join([index_word.get(i, "?") for i in review]) for review in test_data]
    
    dev_texts = train_texts[-dev_size:]
    dev_labels = train_labels[-dev_size:]
    train_texts = train_texts[:-dev_size]
    train_labels = train_labels[:-dev_size]
    
    return (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels)
