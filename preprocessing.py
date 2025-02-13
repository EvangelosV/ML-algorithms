import os
import glob
import re
from tensorflow.keras.datasets import imdb

def tokenize(text):
    """ Επιστρέφει μια λίστα λέξεων (σε πεζά) από το κείμενο. """
    return re.findall(r'\b\w+\b', text.lower())

def load_data(directory):
    """
    Εφόσον τα δεδομένα είναι σε φακέλους (π.χ., data/train/pos, data/train/neg),
    αυτή η συνάρτηση διαβάζει τα αρχεία και επιστρέφει κείμενα και ετικέτες.
    """
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
    """
    Φορτώνει το IMDB dataset χρησιμοποιώντας το API του Keras.
    
    - num_words: Το μέγιστο πλήθος λέξεων (βάσει της συχνότητας) που θα διατηρηθούν.
    - dev_size: Πλήθος παραδειγμάτων από το training set που θα χρησιμοποιηθούν ως
      development set.
      
    Η συνάρτηση επιστρέφει τρία ζεύγη: (train_texts, train_labels), (dev_texts, dev_labels) και (test_texts, test_labels)
    """
    # Φόρτωση δεδομένων από το Keras IMDB dataset
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)
    
    # Λήψη του λεξιλογίου που αντιστοιχεί σε λέξεις-κλειδιά (word_index)
    word_index = imdb.get_word_index()
    # Δημιουργία αντιστρόφου λεξικού: index -> word. Οι δείκτες 0, 1, 2 είναι κρατημένοι για ειδικές τιμές.
    index_word = {i+3: word for word, i in word_index.items()}
    index_word[0] = "<PAD>"
    index_word[1] = "<START>"
    index_word[2] = "<UNK>"
    
    # Μετατροπή των ακολουθιών από ακέραιους σε κείμενο (string)
    train_texts = [" ".join([index_word.get(i, "?") for i in review]) for review in train_data]
    test_texts = [" ".join([index_word.get(i, "?") for i in review]) for review in test_data]
    
    # Δημιουργία του development set από το training set (π.χ., τα τελευταία dev_size παραδείγματα)
    dev_texts = train_texts[-dev_size:]
    dev_labels = train_labels[-dev_size:]
    train_texts = train_texts[:-dev_size]
    train_labels = train_labels[:-dev_size]
    
    return (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels)
