import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.densenet import DenseNet121
from keras.callbacks import EarlyStopping, ModelCheckpoint  # , TensorBoard
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score


N_CLASSES = 14
BATCH_SIZE = 32
EPOCHS = 3
INPUT_SIZE = (224, 224)


def mkdir_if_not_exist(dir):
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass


def clean_df(df):
    # replace Nan with 0, drop useless columns
    df = df.fillna(0)
    df = df.drop(['Sex', 'Age', 'Frontal/Lateral', 'AP/PA'], axis=1)
    # Binary Mapping -> U-Ones model
    df = df.replace(-1, 1)

    return df


def data_generator(df, input_size, batch_size, shuffle=True):
    """df columns[1:15] are classes with labels"""
    data_gen = ImageDataGenerator(rescale=1 / 255.)
    data_generator = data_gen.flow_from_dataframe(df, directory=None, x_col='Path', y_col=list(df.columns[1:]),
                                                  class_mode='raw', target_size=input_size, batch_size=batch_size,
                                                  shuffle=shuffle)

    return data_generator


def my_generator(df, input_size, batch_size):
    """
    Do rescale, resizing
    """
    pass


def build_densenet(transfer_learning=True):
    """
    include_top=False will take away the last fc layer and the GlobalAveragePooling2D before it
    In order to preserve the original architecture while matching the number of classes,
    we need to add pooling layer back
    """
    model = DenseNet121(include_top=False)
    if transfer_learning:
        for layer in model.layers:
            layer.trainable = False
    x = GlobalAveragePooling2D()(model.output)
    x = Dense(14, activation='sigmoid')(x)

    return Model(model.input, x, name='CheXPert')


def train_nn(model, train_generator, valid_generator, epochs=5, compile=True,
             save_best_only=True, save_model_path="models/CheXpert_keras.h5",
             save_log_path='log_dir/'):
    callbacks_list = [
        EarlyStopping(
            monitor='val_acc',
            patience=2,
        ),
        ModelCheckpoint(
            filepath=save_model_path,
            monitor='val_acc',
            save_best_only=save_best_only,
        ),
        # TensorBoard(
        #     log_dir=log_dir,
        #     histogram_freq=100,
        #     batch_size=256,
        #     write_graph=True,
        #     write_grads=True,
        #     write_images=True,
        # )
    ]

    if compile:
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

    train_steps = train_generator.n // train_generator.batch_size
    valid_steps = valid_generator.n // valid_generator.batch_size

    print('batch_size: {}'.format(train_generator.batch_size))
    print('train steps: {}'.format(train_steps))

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_steps,
                                  validation_data=valid_generator,
                                  validation_steps=valid_steps,
                                  epochs=epochs,
                                  callbacks=callbacks_list,
                                  max_queue_size=16,
                                  workers=8,
                                  use_multiprocessing=True)

    return history


def eval_model(df, model, data_generator, steps):
    class_names = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
                   "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
                   "Pleural Effusion", "Pleural Other", "Support Devices"]
    data_generator.reset()
    y_score = model.predict_generator(data_generator, steps=steps)
    y_true = df.to_numpy()[:, 1:].astype(int)

    y_score = np.delete(y_score, 12, 1)
    y_true = np.delete(y_true, 12, 1)
    roc_auc_scores = {}

    for i, class_name in enumerate(class_names):
        roc_auc_scores[class_name] = roc_auc_score(y_true[:, i], y_score[:, i])

    return roc_auc_scores


def loss_plot(history, save_plot_path, model_name):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    mkdir_if_not_exist(save_plot_path)
    plt.figure(figsize=(16, 10))

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig(save_plot_path + model_name + '.png')


def run(file_path,
        save_model_path='models/CehXPert_keras.h5',
        save_model_plot_path='arch/',
        save_plot_path='plots/',
        train=True,
        transfer_learning=True,
        compile=True,
        plot_loss=True,
        save_history=True):
    valid_df_path = file_path + 'valid.csv'
    vdf = pd.read_csv(valid_df_path)
    vdf = clean_df(vdf)
    valid_generator = data_generator(vdf, INPUT_SIZE, BATCH_SIZE, shuffle=False)

    if not os.path.exists(save_model_path):
        print('Creating model...')
        model = build_densenet(transfer_learning=transfer_learning)
    else:
        print('Loading model...')
        model = load_model(save_model_path)
        if train:
            if transfer_learning:
                n_layers = len(model.layers)
                for i, layer in enumerate(model.layers):
                    if i < n_layers - 1:
                        layer.trainable = False
            else:
                for layer in model.layers:
                    layer.trainable = True

    if train:
        train_df_path = file_path + 'train.csv'
        tdf = pd.read_csv(train_df_path)
        tdf = clean_df(tdf)
        train_generator = data_generator(tdf, INPUT_SIZE, BATCH_SIZE)

        history = train_nn(model, train_generator, valid_generator,
                           epochs=EPOCHS, compile=compile,
                           save_model_path='models/checkpoint.h5')

    # pickle history
    if save_history:
        save_hist_path = os.path.dirname(save_model_path) + '/chexpert121_history.p'
        with open(save_hist_path, mode='wb') as f:
            # history is a list of history objects return from individual model.fit()
            pickle.dump({'history': history}, f)

    if plot_loss:
        loss_plot(history, save_plot_path, model.name)

    auroc_scores = eval_model(vdf, model, valid_generator, valid_generator.n // valid_generator.batch_size + 1)
    print(auroc_scores)

    with open('data/chexpert121_scores_tf_to_full.p', mode='wb') as f:
        pickle.dump({'auroc_scores': auroc_scores}, f)


if __name__ == '__main__':
    run('CheXpert-v1.0-small/',
        save_model_path='models/chexpert_keras_tf_to_full.h5',
        train=False,
        transfer_learning=False,
        compile=False,
        plot_loss=False,
        save_history=False)
