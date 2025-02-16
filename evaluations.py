# evaluation.py
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(y_true, y_pred):
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
    filename = classifier_class.__name__+"_positive_learning_curves.png" 
    plt.savefig(filename)
    print(" ! Saved as "+filename+" in the folder !")


