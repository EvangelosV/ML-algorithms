import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def compute_metrics(y_true, y_pred):
    """
    Υπολογίζει precision, recall και F1 για κάθε κατηγορία και τους micro/macro μέσους όρους.
    Επιστρέφει λεξικό με τα αποτελέσματα.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    metrics = {}
    classes = np.unique(y_true)
    for cls in classes:
        tp = sum((y_true == cls) & (y_pred == cls))
        fp = sum((y_true != cls) & (y_pred == cls))
        fn = sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics[cls] = {'precision': precision, 'recall': recall, 'f1': f1}
    # Micro και Macro
    tp_total = sum([sum((y_true == cls) & (y_pred == cls)) for cls in classes])
    fp_total = sum([sum((y_true != cls) & (y_pred == cls)) for cls in classes])
    fn_total = sum([sum((y_true == cls) & (y_pred != cls)) for cls in classes])
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    metrics['micro'] = {'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1}
    macro_precision = np.mean([metrics[cls]['precision'] for cls in classes])
    macro_recall = np.mean([metrics[cls]['recall'] for cls in classes])
    macro_f1 = np.mean([metrics[cls]['f1'] for cls in classes])
    metrics['macro'] = {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1}
    return metrics

def plot_learning_curve(classifier_class, X_train, y_train, X_dev, y_dev, metric_name, train_sizes, classifier_params={}):
    """
    Δημιουργεί καμπύλες μάθησης για ένα συγκεκριμένο metric (π.χ. 'f1') για την θετική κατηγορία.
    """
    train_metrics = []
    dev_metrics = []
    for size in train_sizes:
        indices = np.random.choice(len(X_train), size=size, replace=False)
        X_train_subset = [X_train[i] for i in indices]
        y_train_subset = [y_train[i] for i in indices]
        classifier = classifier_class(**classifier_params)
        classifier.fit(X_train_subset, y_train_subset)
        y_train_pred = classifier.predict(X_train_subset)
        y_dev_pred = classifier.predict(X_dev)
        metrics_train = compute_metrics(y_train_subset, y_train_pred)
        metrics_dev = compute_metrics(y_dev, y_dev_pred)
        # Για παράδειγμα, επιλέγουμε την κατηγορία 1:
        train_metrics.append(metrics_train[1][metric_name])
        dev_metrics.append(metrics_dev[1][metric_name])
    
    plt.figure()
    plt.plot(train_sizes, train_metrics, marker='o', label='Training ' + metric_name)
    plt.plot(train_sizes, dev_metrics, marker='s', label='Development ' + metric_name)
    plt.xlabel('Training set size')
    plt.ylabel(metric_name)
    plt.title('Learning Curve (' + metric_name + ')')
    plt.legend()
    plt.grid(True)
    plt.show()
