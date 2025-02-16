from preprocessing import load_imdb_data
from vocab import build_vocabulary, transform_texts
from evaluations import *
from algorithms.naive_bayes import NaiveBayes
from algorithms.random_forest import RandomForest
import numpy as np
from sklearn.naive_bayes import BernoulliNB as SKNaiveBayes
from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from sklearn.neural_network import MLPClassifier as SKMLP
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from RNN.rnn_dataset import TextDataset
from RNN.rnn_model import StackedBiRNN
from RNN.rnn_utils import train_model, evaluate_model, plot_loss_curves


def main():
    # ------------------- Yperparametroi --------------------
    num_words = 10000  # orio leksewn IMDB dataset 
    dev_size = 10000   # sample size gia development set
    n = 30            # n pio sixnes lekseis gia aporipsi
    k = 30            # k pio spanies lekseis gia aporipsi
    m = 1000          # vocab size 
    alpha = 1.0       # LaPlace smoothing gia NB
    n_trees = 30      # num of trees Random Forest / 50 TOO BIG 35 TOO BIG
    max_depth = 15    # 10 TOO SMALL 

    # ---------------------------------------------------------------------

    # Loading IMDB data
    print(" \n*** Booting up *** \n")
    print("Τιμές υπερπαραμέτρων:\nΌριο λέξεων:", num_words, "\nΜέγεθος dev set:", dev_size, "\nΠιο συχνές λέξεις(discard):", n,
           "\nΠιο σπάνιες λέξεις(discard):", k, "\nΜέγεθος λεξιλογίου:", m, "\n(Naive Bayhes)Laplace Smoothing (α):",
             alpha, "\n(Random Forest)Δέντρα:", n_trees, "\n(Random Forest)Μέγιστο βάθος:", max_depth)
    print("Loading IMDB data Keras API...")
    (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels) = load_imdb_data(num_words=num_words, dev_size=dev_size)
    print("\nNumber of training examples:", len(train_texts))
    
    # Calling vocab functions
    print("Creating the vocabulary...")
    vocabulary = build_vocabulary(train_texts, train_labels, n, k, m)
    print("Vocabulary size:", len(vocabulary))
    
    # Creating vectors
    X_train = transform_texts(train_texts, vocabulary)
    X_dev = transform_texts(dev_texts, vocabulary)
    X_test = transform_texts(test_texts, vocabulary)
    
    train_sizes = np.linspace(100, len(X_train), 5, dtype=int)
    
    # ------------------ Naive Bayes ------------------
    print("\n--- Naive Bayes ---")
    nb = NaiveBayes(alpha=alpha)
    nb.fit(X_train, train_labels)
    nb_train_preds = nb.predict(X_train)
    nb_train_metrics = compute_metrics(train_labels, nb_train_preds)
    
    print("* Metrics for Naive Bayes Algorithm: Training & Development Data *")
    for metric_name, value in nb_train_metrics.items():
        formatted_value = {k: round(v, 3) for k, v in value.items()}
        print(f"{metric_name}: {formatted_value}")
    
    # Learning curves apo metrics (precision, recall, f1) positive category (π.χ. class 1)
    print("\nNAIVE BAYES - Creating learning curves only for positive category (cls=1)...")
    plot_learning_curves_for_category(NaiveBayes, X_train, train_labels, X_dev, dev_labels,
        train_sizes, classifier_params={'alpha': alpha},category=1)

    nb_test_preds = nb.predict(X_test)
    nb_test_metrics = compute_metrics(test_labels, nb_test_preds)
    print("\n* Metrics for Naive Bayes Algorithm: Testing Data *")
    for metric_name, value in nb_test_metrics.items():
        formatted_value = {k: round(v, 3) for k, v in value.items()}
        print(f"{metric_name}: {formatted_value}")
    
    # ------------------ Random Forest ------------------
    print("\n--- Random Forest ---")
    rf = RandomForest(n_trees=n_trees, max_depth=max_depth, max_features='sqrt')
    rf.fit(X_train, train_labels)
    rf_train_preds = rf.predict(X_train)
    rf_train_metrics = compute_metrics(train_labels, rf_train_preds)
    
    print("* Metrics for Random Forest Algorithm: Training & Development Data *")
    for metric_name, value in rf_train_metrics.items():
        formatted_value = {k: round(v, 3) for k, v in value.items()}
        print(f"{metric_name}: {formatted_value}")
   
    # Learning curves apo metrics (precision, recall, f1) positive category (π.χ. class 1)
    print("\nLearning curves may take a while to generate...")
    print("RANDOM FOREST - Creating learning curves only for positive category (cls=1)...")
    plot_learning_curves_for_category(RandomForest, X_train, train_labels, X_dev, dev_labels,
        train_sizes, classifier_params={'n_trees': n_trees, 'max_depth': max_depth, 'max_features': 'sqrt'},category=1)

    
    rf_test_preds = rf.predict(X_test)
    rf_test_metrics = compute_metrics(test_labels, rf_test_preds)
    print("\n* Metrics for Random Forest Algorithm: Testing Data *")
    for metric_name, value in rf_test_metrics.items():
        formatted_value = {k: round(v, 3) for k, v in value.items()}
        print(f"{metric_name}: {formatted_value}")
   

    # ------------------ Scikit-Learn Naive Bayes ------------------
    print("\n--- Scikit-Learn Naive Bayes ---")
    sk_nb = SKNaiveBayes(alpha=alpha)
    sk_nb.fit(X_train, train_labels)
    sk_nb_preds = sk_nb.predict(X_train)
    sk_nb_metrics = compute_metrics(train_labels, sk_nb_preds)
    
    
    print("\n* Metrics for Scikit-Learn Naive Bayes: Training & Development Data *")
    for metric_name, value in sk_nb_metrics.items():
        formatted_value = {k: round(v, 3) for k, v in value.items()}
        print(f"{metric_name}: {formatted_value}")
    
    # Learning curves apo metrics (precision, recall, f1) positive category (π.χ. class 1)
    print("\nScikit-Learn NB - Learning curves for positive category (cls=1)...")
    plot_learning_curves_for_category(SKNaiveBayes, X_train, train_labels, X_dev, dev_labels,
                                      train_sizes, classifier_params={'alpha': alpha}, category=1)

    sk_nb_test_preds = sk_nb.predict(X_test)
    sk_nb_test_metrics = compute_metrics(test_labels, sk_nb_test_preds)
    print("\n* Metrics for Scikit-Learn Naive Bayes: Testing Data *")
    for metric_name, value in sk_nb_test_metrics.items():
        formatted_value = {k: round(v, 3) for k, v in value.items()}
        print(f"{metric_name}: {formatted_value}")
    

    # ------------------ Scikit-Learn Random Forest ------------------
    print("\n--- Scikit-Learn Random Forest ---")
    sk_rf = SKRandomForest(n_estimators=n_trees, max_depth=max_depth)
    sk_rf.fit(X_train, train_labels)
    sk_rf_preds = sk_rf.predict(X_train)
    sk_rf_metrics = compute_metrics(train_labels, sk_rf_preds)
    
    print("\n* Metrics for Scikit-Learn Random Forest: Training & Development Data *")
    for metric_name, value in sk_rf_metrics.items():
        formatted_value = {k: round(v, 3) for k, v in value.items()}
        print(f"{metric_name}: {formatted_value}")
    
    # Learning curves apo metrics (precision, recall, f1) positive category (π.χ. class 1)
    print("\nScikit-Learn RF - Learning curves for positive category (cls=1)...")
    plot_learning_curves_for_category(SKRandomForest, X_train, train_labels, X_dev, dev_labels,
                                      train_sizes, classifier_params={'n_estimators': n_trees, 'max_depth': max_depth}, category=1)
    
    sk_rf_test_preds = sk_rf.predict(X_test)
    sk_rf_test_metrics = compute_metrics(test_labels, sk_rf_test_preds)
    print("\n* Metrics for Scikit-Learn Random Forest: Testing Data *")
    for metric_name, value in sk_rf_test_metrics.items():
        formatted_value = {k: round(v, 3) for k, v in value.items()}
        print(f"{metric_name}: {formatted_value}")


    # ------------------ Stacked Bidirectional RNN me PyTorch ------------------
    print("\n--- Stacked Bidirectional RNN ---\n")
    #Yperparametroi gia to RNN
    embed_dim = 100     # embeddings dimension
    hidden_dim = 128    # hidden space size LSTM
    output_dim = 2      # binary
    n_layers = 2        
    dropout = 0.5
    num_epochs = 7     # plithos epochs
    learning_rate = 0.001
    max_length = 100    # max length keimenou

    print("Υπερπαραμέτροι RNN:")
    print("Embedding dim:", embed_dim)
    print("Hidden dim:", hidden_dim)
    print("Αριθμός στοιβαγμένων επιπέδων:", n_layers)
    print("Dropout:", dropout)
    print("Learning rate:", learning_rate)
    print("Αριθμός εποχών:", num_epochs)
    
    
    # Create datasets gia RNN 
    
    train_dataset = TextDataset(train_texts, train_labels, vocabulary, max_length=max_length)
    dev_dataset = TextDataset(dev_texts, dev_labels, vocabulary, max_length=max_length)
    test_dataset = TextDataset(test_texts, test_labels, vocabulary, max_length=max_length)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    vocab_size = len(vocabulary) + 1  # +1 gia padding (index 0)
    
    
    #Create RNN model
    model = StackedBiRNN(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
                         output_dim=output_dim, n_layers=n_layers, dropout=dropout, bidirectional=True)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_losses, dev_losses, best_epoch = train_model(model, train_loader, dev_loader, optimizer, criterion, num_epochs, device)
    print(f"Best Epoch: {best_epoch}")
    
    print("\nLoss Curves for RNN:")
    plot_loss_curves(train_losses, dev_losses)
    print(" ! Saved as Loss_Curves_RNN.png in the folder !")
    
    test_true, test_preds = evaluate_model(model, test_loader, device)
    rnn_test_metrics = compute_metrics(test_true, test_preds)
    print("\nTesting Data Metrics RNN:")
    for metric_name, value in rnn_test_metrics.items():
        formatted_value = {k: round(v, 3) for k, v in value.items()}
        print(f"{metric_name}: {formatted_value}")


if __name__ == '__main__':
    main()
