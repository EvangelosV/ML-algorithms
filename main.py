from preprocessing import load_imdb_data
from vocab import build_vocabulary, transform_texts
from evaluations import *
from algorithms.naive_bayes import BernoulliNB
from algorithms.random_forest import RandomForest
import numpy as np

def main():
    # ------------------- Ρυθμίσεις & Υπερπαράμετροι --------------------
    num_words = 10000  # Όριο λέξεων για το IMDB dataset (απ' το API του Keras)
    dev_size = 10000   # Πλήθος παραδειγμάτων για το development set
    n = 20            # n πιο συχνές λέξεις για απόρριψη
    k = 20            # k πιο σπάνιες λέξεις για απόρριψη
    m = 1000          # Μέγεθος λεξιλογίου μετά την επιλογή
    alpha = 1.0       # Smoothing για τον NB
    n_trees = 10      # Πλήθος δέντρων για το Random Forest
    max_depth = 10    # Μέγιστο βάθος των δέντρων
    # ---------------------------------------------------------------------

    # Φόρτωση δεδομένων από το Keras IMDB dataset
    print(" \n*** Booting up *** \n")
    print("Τιμές υπερπαραμέτρων:\nΌριο λέξεων:", num_words, "\nΜέγεθος dev set:", dev_size, "\nΠιο συχνές λέξεις(discard):", n,
           "\nΠιο σπάνιες λέξεις(discard):", k, "\nΜέγεθος λεξιλογίου:", m, "\nLaplace Smoothing (α):",
             alpha, "\n(Random Forest)Δέντρα:", n_trees, "\n(Random Forest)Μέγιστο βάθος:", max_depth)
    print("Φόρτωση δεδομένων IMDB από το Keras API...")
    (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels) = load_imdb_data(num_words=num_words, dev_size=dev_size)
    print("Αριθμός εκπαιδευτικών παραδειγμάτων:", len(train_texts))
    
    # Δημιουργία λεξιλογίου με βάση τα training δεδομένα
    print("Δημιουργία λεξιλογίου...")
    vocabulary = build_vocabulary(train_texts, train_labels, n, k, m)
    print("Μέγεθος λεξιλογίου:", len(vocabulary))
    
    # Μετατροπή των κειμένων σε διανύσματα χαρακτηριστικών
    print("Μετατροπή κειμένων σε διανύσματα χαρακτηριστικών...")
    X_train = transform_texts(train_texts, vocabulary)
    X_dev = transform_texts(dev_texts, vocabulary)
    X_test = transform_texts(test_texts, vocabulary)
    
    # ------------------ Εκπαίδευση & Αξιολόγηση Bernoulli Naive Bayes ------------------
    print("\n--- Εκπαίδευση Bernoulli Naive Bayes ---")
    nb = BernoulliNB(alpha=alpha)
    nb.fit(X_train, train_labels)
    nb_train_preds = nb.predict(X_train)
    nb_train_metrics = compute_metrics(train_labels, nb_train_preds)
    
    print("Μετρικές Εκπαίδευσης (BernoulliNB):")
    # Χρησιμοποιούμε pprint για πιο καθαρή εμφάνιση (χωρίς np.float64/np.int64)
    #print(nb_train_metrics)
    for metric_name, value in nb_train_metrics.items():
        print(f"{metric_name}: {value}")
    
    # Learning curves μόνο για τα metrics (precision, recall, f1) της θετικής κατηγορίας (π.χ. class 1)
    train_sizes = np.linspace(100, len(X_train), 5, dtype=int)
    print("\nΠαραγωγή learning curves για την θετική κατηγορία (BernoulliNB)...")
    plot_learning_curves_for_category(BernoulliNB, X_train, train_labels, X_dev, dev_labels,
        train_sizes, classifier_params={'alpha': alpha},category=1)
    
    # ------------------ Εκπαίδευση & Αξιολόγηση Random Forest ------------------
    print("\n--- Εκπαίδευση Random Forest ---")
    rf = RandomForest(n_trees=n_trees, max_depth=max_depth, max_features='sqrt')
    rf.fit(X_train, train_labels)
    rf_train_preds = rf.predict(X_train)
    rf_train_metrics = compute_metrics(train_labels, rf_train_preds)
    
    print("Μετρικές Εκπαίδευσης (Random Forest):")
    #print(rf_train_metrics)
    for metric_name, value in nb_train_metrics.items():
        print(f"{metric_name}: {value}")
    
    # Learning curves για τα micro & macro metrics (συμπεριλαμβάνουν όλες τις κατηγορίες)
    print("\nΠαραγωγή learning curves (Micro & Macro) για τον Random Forest...")
    plot_learning_curves_micro_macro(RandomForest, X_train, train_labels, X_dev, dev_labels,
        train_sizes, classifier_params={'n_trees': n_trees, 'max_depth': max_depth, 'max_features': 'sqrt'})
    
    # Μπορείς επίσης να αξιολογήσεις σε dev και test σύνολα

if __name__ == '__main__':
    main()
