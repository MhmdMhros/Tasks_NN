import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_decision_boundary(x1, x2, y, w, c1, c2, title):
    class_name_to_target = {c1: -1, c2: 1}
    b = w[0]
    w1 = w[1]
    w2 = w[2]

    fig, ax = plt.subplots()

    # Scatter points for class c1
    class1 = y[y == class_name_to_target[c1]]
    ax.scatter(x1[class1.index], x2[class1.index], color='blue', label=f'Class {c1}')

    # Scatter points for class c2
    class2 = y[y == class_name_to_target[c2]]
    ax.scatter(x1[class2.index], x2[class2.index], color='red', label=f'Class {c2}')

    # Plot the decision boundary
    x_min, x_max = x1.min() - 1, x1.max() + 1
    x11, x22 = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(x2.min() - 1, x2.max() + 1, 0.01))
    eq = w1 * x11 + w2 * x22 + b
    ax.contour(x11, x22, eq, levels=[0], colors='black')

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    ax.legend()
    plt.show()


def plot_confusion_matrix(tp, tn, fp, fn, c1, c2, accuracy, cnt_epochs):
    confusion_matrix = np.array([[tp, fp], [fn, tn]])

    plt.figure(figsize=(6, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d', cbar=False)

    class_names = [c1, c2]
    plt.xticks(ticks=range(len(class_names)), labels=class_names)
    plt.yticks(ticks=range(len(class_names)), labels=class_names)
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('Actual', fontsize=15)

    plt.figtext(0.5, 0.01, f"Accuracy: {accuracy:.2f}%,      Min Epochs: {cnt_epochs}", fontsize=12, ha="center")

    plt.show()
