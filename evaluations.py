# evaluation.py
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(y_true, y_pred):
    """
    Υπολογίζει precision, recall και F1 για κάθε κατηγορία, καθώς και τους micro και macro μέσους όρους.
    Επιστρέφει ένα λεξικό με τα αποτελέσματα, όπου όλα τα νούμερα είναι κανονικοί τύποι Python.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    metrics = {}
    classes = np.unique(y_true)
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[int(cls)] = {'Precision': float(precision), 'Recall': float(recall), 'F1': float(f1)}
    # Micro-average
    tp_total = sum(np.sum((y_true == cls) & (y_pred == cls)) for cls in classes)
    fp_total = sum(np.sum((y_true != cls) & (y_pred == cls)) for cls in classes)
    fn_total = sum(np.sum((y_true == cls) & (y_pred != cls)) for cls in classes)
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    metrics['micro'] = {'Precision': float(micro_precision), 'Recall': float(micro_recall), 'F1': float(micro_f1)}
    # Macro-average
    macro_precision = np.mean([metrics[int(cls)]['Precision'] for cls in classes])
    macro_recall = np.mean([metrics[int(cls)]['Recall'] for cls in classes])
    macro_f1 = np.mean([metrics[int(cls)]['F1'] for cls in classes])
    metrics['macro'] = {'Precision': float(macro_precision), 'Recall': float(macro_recall), 'F1': float(macro_f1)}
    return metrics

def plot_learning_curves_for_category(classifier_class, X_train, y_train, X_dev, y_dev, train_sizes, classifier_params={}, category=1):
    """
    Γενική συνάρτηση που δημιουργεί learning curves (precision, recall, f1) για μια συγκεκριμένη κατηγορία.
    Δημιουργεί 3 subplots για τα metrics, χρησιμοποιώντας τα δεδομένα του training set (με διάφορα μεγέθη)
    και του development set.
    
    Parameters:
      - classifier_class: η κλάση του ταξινομητή (π.χ. BernoulliNB)
      - X_train, y_train: δεδομένα εκπαίδευσης
      - X_dev, y_dev: δεδομένα ανάπτυξης
      - train_sizes: λίστα με μεγέθη training set για τα οποία θα γίνει το πείραμα
      - classifier_params: dictionary υπερπαραμέτρων για τον ταξινομητή
      - category: η κατηγορία για την οποία θέλουμε τα metrics (π.χ. 1 για τις θετικές κριτικές)
    """
    metrics_list = ['Precision', 'Recall', 'F1']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(metrics_list):
        train_vals = []
        dev_vals = []
        for size in train_sizes:
            indices = np.random.choice(len(X_train), size=size, replace=False)
            X_subset = [X_train[i] for i in indices]
            y_subset = [y_train[i] for i in indices]
            clf = classifier_class(**classifier_params)
            clf.fit(X_subset, y_subset)
            y_train_pred = clf.predict(X_subset)
            y_dev_pred = clf.predict(X_dev)
            metrics_train = compute_metrics(y_subset, y_train_pred)
            metrics_dev = compute_metrics(y_dev, y_dev_pred)
            train_vals.append(metrics_train[category][metric])
            dev_vals.append(metrics_dev[category][metric])
        axes[idx].plot(train_sizes, train_vals, marker='o', label='Training '+metric)
        axes[idx].plot(train_sizes, dev_vals, marker='s', label='Dev '+metric)
        axes[idx].set_xlabel('Training set size')
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(f"{metric.capitalize()} for Category {category}")
        axes[idx].legend()
        axes[idx].grid(True)
    plt.tight_layout()
    filename = "NB_positive_learning_curves.png" if classifier_class.__name__ == 'NaiveBayes' else "RF_positive_learning_curves.png"
    plt.savefig(filename)
    print(" ! Saved as "+filename+" in the folder !")
    #plt.ion()
    #plt.show()

def plot_learning_curves_micro_macro(classifier_class, X_train, y_train, X_dev, y_dev, train_sizes, classifier_params={}):
    """
    Δημιουργεί learning curves για τα micro και macro metrics (precision, recall, f1), 
    όπου υπολογίζονται λαμβάνοντας υπόψη όλες τις κατηγορίες.
    Δημιουργεί 3 subplots για precision, recall και f1, με δύο γραμμές για το training (micro, macro)
    και δύο για το development set (micro, macro).
    """
    metrics_list = ['Precision', 'Recall', 'F1']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(metrics_list):
        micro_train = []
        micro_dev = []
        macro_train = []
        macro_dev = []
        for size in train_sizes:
            indices = np.random.choice(len(X_train), size=size, replace=False)
            X_subset = [X_train[i] for i in indices]
            y_subset = [y_train[i] for i in indices]
            clf = classifier_class(**classifier_params)
            clf.fit(X_subset, y_subset)
            y_train_pred = clf.predict(X_subset)
            y_dev_pred = clf.predict(X_dev)
            metrics_train = compute_metrics(y_subset, y_train_pred)
            metrics_dev = compute_metrics(y_dev, y_dev_pred)
            micro_train.append(metrics_train['micro'][metric])
            micro_dev.append(metrics_dev['micro'][metric])
            macro_train.append(metrics_train['macro'][metric])
            macro_dev.append(metrics_dev['macro'][metric])
        axes[idx].plot(train_sizes, micro_train, marker='o', label='Training Micro')
        axes[idx].plot(train_sizes, micro_dev, marker='s', label='Dev Micro')
        axes[idx].plot(train_sizes, macro_train, marker='^', label='Training Macro')
        axes[idx].plot(train_sizes, macro_dev, marker='x', label='Dev Macro')
        axes[idx].set_xlabel('Training set size')
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(f"{metric.capitalize()} (Micro & Macro)")
        axes[idx].legend()
        axes[idx].grid(True)
    plt.tight_layout()
    filename = "NB_micro_macro_learning_curves.png" if classifier_class.__name__ == 'NaiveBayes' else "RF_micro_macro_learning_curves.png"
    plt.savefig(filename)
    print(" ! Saved as "+filename+" in the folder !")
    #plt.ion()
    #plt.show()
