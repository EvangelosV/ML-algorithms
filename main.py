from preprocessing import load_imdb_data
from vocab import build_vocabulary, transform_texts
from evaluations import *
from algorithms.naive_bayes import NaiveBayes
from algorithms.random_forest import RandomForest
import numpy as np
from sklearn.naive_bayes import BernoulliNB as SKNaiveBayes
from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from sklearn.neural_network import MLPClassifier as SKMLP


def main():
    # ------------------- Ρυθμίσεις & Υπερπαράμετροι --------------------
    num_words = 10000  # Όριο λέξεων για το IMDB dataset (απ' το API του Keras)
    dev_size = 10000   # Πλήθος παραδειγμάτων για το development set
    n = 30            # n πιο συχνές λέξεις για απόρριψη
    k = 30            # k πιο σπάνιες λέξεις για απόρριψη
    m = 1000          # Μέγεθος λεξιλογίου μετά την επιλογή
    alpha = 1.0       # Smoothing για τον NB
    n_trees = 35      # Πλήθος δέντρων για το Random Forest 50 TOO BIG
    max_depth = 20    # Μέγιστο βάθος των δέντρων 10 TOO SMALL
    # ---------------------------------------------------------------------

    # Φόρτωση δεδομένων από το Keras IMDB dataset
    print(" \n*** Booting up *** \n")
    print("Τιμές υπερπαραμέτρων:\nΌριο λέξεων:", num_words, "\nΜέγεθος dev set:", dev_size, "\nΠιο συχνές λέξεις(discard):", n,
           "\nΠιο σπάνιες λέξεις(discard):", k, "\nΜέγεθος λεξιλογίου:", m, "\n(Naive Bayhes)Laplace Smoothing (α):",
             alpha, "\n(Random Forest)Δέντρα:", n_trees, "\n(Random Forest)Μέγιστο βάθος:", max_depth)
    print("Loading IMDB data Keras API...")
    (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels) = load_imdb_data(num_words=num_words, dev_size=dev_size)
    print("\nNumber of training examples:", len(train_texts))
    
    # Δημιουργία λεξιλογίου με βάση τα training δεδομένα
    print("Creating the vocabulary...")
    vocabulary = build_vocabulary(train_texts, train_labels, n, k, m)
    print("Vocabulary size:", len(vocabulary))
    
    # Μετατροπή των κειμένων σε διανύσματα χαρακτηριστικών
    X_train = transform_texts(train_texts, vocabulary)
    X_dev = transform_texts(dev_texts, vocabulary)
    X_test = transform_texts(test_texts, vocabulary)
    
    train_sizes = np.linspace(100, len(X_train), 5, dtype=int)
    
    # ------------------ Εκπαίδευση & Αξιολόγηση Bernoulli Naive Bayes ------------------
    print("\n--- Naive Bayes ---")
    nb = NaiveBayes(alpha=alpha)
    nb.fit(X_train, train_labels)
    nb_train_preds = nb.predict(X_train)
    nb_train_metrics = compute_metrics(train_labels, nb_train_preds)
    
    print("* Metrics for Naive Bayes Algorithm: *")
    # Χρησιμοποιούμε pprint για πιο καθαρή εμφάνιση (χωρίς np.float64/np.int64)
    #print(nb_train_metrics)
    for metric_name, value in nb_train_metrics.items():
        formatted_value = {k: round(v, 3) for k, v in value.items()}
        print(f"{metric_name}: {formatted_value}")
    
    # Learning curves μόνο για τα metrics (precision, recall, f1) της θετικής κατηγορίας (π.χ. class 1)
    print("\nNAIVE BAYES - Creating learning curves only for positive category (cls=1)...")
    plot_learning_curves_for_category(NaiveBayes, X_train, train_labels, X_dev, dev_labels,
        train_sizes, classifier_params={'alpha': alpha},category=1)
    
    print("NAIVE BAYES - Creating Micro & Macro average learning curves...")
    plot_learning_curves_micro_macro(NaiveBayes, X_train, train_labels, X_dev, dev_labels,
        train_sizes, classifier_params={'alpha': alpha})

    
    # ------------------ Εκπαίδευση & Αξιολόγηση Random Forest ------------------
    print("\n--- Random Forest ---")
    rf = RandomForest(n_trees=n_trees, max_depth=max_depth, max_features='sqrt')
    rf.fit(X_train, train_labels)
    rf_train_preds = rf.predict(X_train)
    rf_train_metrics = compute_metrics(train_labels, rf_train_preds)
    
    print("* Metrics for Random Forest Algorithm: *")
    for metric_name, value in rf_train_metrics.items():
        formatted_value = {k: round(v, 3) for k, v in value.items()}
        print(f"{metric_name}: {formatted_value}")
   
    print("Learning curves may take a while to generate...")
    print("\nRANDOM FOREST - Creating learning curves only for positive category (cls=1)...")
    plot_learning_curves_for_category(RandomForest, X_train, train_labels, X_dev, dev_labels,
        train_sizes, classifier_params={'n_trees': n_trees, 'max_depth': max_depth, 'max_features': 'sqrt'},category=1)

    # Learning curves για τα micro & macro metrics (συμπεριλαμβάνουν όλες τις κατηγορίες)
    print("RANDOM FOREST - Creating Micro & Macro learning curves Random Forest...")
    plot_learning_curves_micro_macro(RandomForest, X_train, train_labels, X_dev, dev_labels,
        train_sizes, classifier_params={'n_trees': n_trees, 'max_depth': max_depth, 'max_features': 'sqrt'})
    

    # ------------------ Scikit-Learn Naive Bayes ------------------
    print("\n=== Scikit-Learn Naive Bayes ===")
    sk_nb = SKNaiveBayes(alpha=alpha, binarize=0.0)
    sk_nb.fit(X_train, train_labels)
    sk_nb_preds = sk_nb.predict(X_train)
    sk_nb_metrics = compute_metrics(train_labels, sk_nb_preds)
    
    print("\n* Metrics for Scikit-Learn Naive Bayes: *")
    for metric_name, value in sk_nb_metrics.items():
        formatted_value = {k: round(v, 3) for k, v in value.items()}
        print(f"{metric_name}: {formatted_value}")
    
    print("\nScikit-Learn NB - Learning curves for positive category (cls=1)...")
    plot_learning_curves_for_category(SKNaiveBayes, X_train, train_labels, X_dev, dev_labels,
                                      train_sizes, classifier_params={'alpha': alpha, 'binarize': 0.0}, category=1)
    
    print("Scikit-Learn NB - Learning curves for Micro & Macro averages...")
    plot_learning_curves_micro_macro(SKNaiveBayes, X_train, train_labels, X_dev, dev_labels,
                                     train_sizes, classifier_params={'alpha': alpha, 'binarize': 0.0})
    
    # ------------------ Scikit-Learn Random Forest ------------------
    print("\n=== Scikit-Learn Random Forest ===")
    sk_rf = SKRandomForest(n_estimators=n_trees, max_depth=max_depth)
    sk_rf.fit(X_train, train_labels)
    sk_rf_preds = sk_rf.predict(X_train)
    sk_rf_metrics = compute_metrics(train_labels, sk_rf_preds)
    
    print("\n* Metrics for Scikit-Learn Random Forest: *")
    for metric_name, value in sk_rf_metrics.items():
        formatted_value = {k: round(v, 3) for k, v in value.items()}
        print(f"{metric_name}: {formatted_value}")
    
    print("\nScikit-Learn RF - Learning curves for positive category (cls=1)...")
    plot_learning_curves_for_category(SKRandomForest, X_train, train_labels, X_dev, dev_labels,
                                      train_sizes, classifier_params={'n_estimators': n_trees, 'max_depth': max_depth}, category=1)
    
    print("Scikit-Learn RF - Learning curves for Micro & Macro averages...")
    plot_learning_curves_micro_macro(SKRandomForest, X_train, train_labels, X_dev, dev_labels,
                                     train_sizes, classifier_params={'n_estimators': n_trees, 'max_depth': max_depth})

if __name__ == '__main__':
    main()
