import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_conf_matrix(trainer, dataset):
    predictions = trainer.predict(dataset)
    y_true = dataset['labels']
    y_pred = np.argmax(predictions.predictions, axis=1)
    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(trainer, dataset, labels, title='Confusion Matrix'):
    cm = calculate_conf_matrix(trainer, dataset)
    plt.figure()
    cm_percentage = cm / cm.sum() * 100
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig('confusion_matrix.png')
