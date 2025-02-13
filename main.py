from preprocessing import load_imdb_data
from vocab import build_vocabulary, transform_texts
from evaluations import compute_metrics, plot_learning_curve
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
    
    # ==================== Εκπαίδευση και Αξιολόγηση Bernoulli Naive Bayes ====================
    print("\n--- Εκπαίδευση Bernoulli Naive Bayes ---")
    nb = BernoulliNB(alpha=alpha)
    nb.fit(X_train, train_labels)
    train_preds = nb.predict(X_train)
    nb_train_metrics = compute_metrics(train_labels, train_preds)
    print("Μετρικές εκπαίδευσης (NB):", nb_train_metrics)
    
    # Παράδειγμα καμπύλης μάθησης για τον NB
    train_sizes = np.linspace(100, len(X_train), 10, dtype=int)
    plot_learning_curve(BernoulliNB, X_train, train_labels, X_dev, dev_labels, 'f1', train_sizes, classifier_params={'alpha': alpha})
    
    # ==================== Εκπαίδευση και Αξιολόγηση Random Forest ====================
    print("\n--- Εκπαίδευση Random Forest ---")
    rf = RandomForest(n_trees=n_trees, max_depth=max_depth, max_features='sqrt')
    rf.fit(X_train, train_labels)
    train_preds_rf = rf.predict(X_train)
    rf_train_metrics = compute_metrics(train_labels, train_preds_rf)
    print("Μετρικές εκπαίδευσης (RF):", rf_train_metrics)
    
    # Μπορείς επίσης να αξιολογήσεις σε dev και test σύνολα

if __name__ == '__main__':
    main()
