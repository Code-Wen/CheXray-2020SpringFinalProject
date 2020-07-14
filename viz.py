import numpy as np
import matplotlib.pyplot as plt


def loss_plot(train_loss, val_loss, fn='train_val_loss_plot'):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(16, 10))
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig(fn + '.png')


def label_stats(labels, class_names, figure_name):
    class_counter = [0] * len(class_names)

    for sample_label in labels:
        for i in range(len(sample_label)):
            if sample_label[i] == 1:
                class_counter[i] += 1

    plt.figure(figsize=[15, 15])
    plt.title(figure_name)
    label_data = np.arange(len(class_names))
    plt.bar(label_data, class_counter, align='center')
    plt.xticks(label_data, class_names, rotation=80)
    plt.savefig('plots/' + figure_name + '.png')
