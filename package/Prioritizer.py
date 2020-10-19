from collections import defaultdict

from sklearn.manifold import TSNE
import umap
from sklearn.model_selection import KFold

from Data import DataCI
from DataGenerator import DataGenerator
from Model import Model, Metrics
from Visualizer import Visualizer

from tensorflow import keras
import numpy as np
import pandas as pd


def reduce_dim(weights, components=3, method='TSNE'):
    """
    Reduce dimensions of embeddings
    :param weights:
    :param components:
    :param method:
    :return: TSNE or UMAP element
    """
    if method == 'TSNE':
        return TSNE(components, metric='cosine').fit_transform(weights)
    elif method == 'UMAP':
        # Might want to try different parameters for UMAP
        return umap.UMAP(n_components=components, metric='cosine',
                         init='random', n_neighbors=5).fit_transform(weights)


class NNEmbeddings(Model, Metrics, Visualizer):
    """
    Neural Networks Embeddings model which inherits from abstract class Model and class Metrics.
    Once it is created, all the data becomes available from DataCI class and there is the possibility
    of loading a previously trained model, or to train from scratch.
    """

    def __init__(self, D: DataCI, model_file: str = 'model.h5', embedding_size: int = 50, optimizer: str = 'Adam',
                 negative_ratio=1, nb_epochs: int = 10, batch_size: int = 10, load: bool = False,
                 save: bool = False, classification: bool = True):
        """
        NNEmbeddings Class initialization.
        :param D:
        :param model_file:
        :param embedding_size:
        :param optimizer:
        :param save:
        :param load:
        """
        Model.__init__(self)
        Metrics.__init__(self)
        Visualizer.__init__(self)
        self.Data = D

        self.model_file = model_file
        self.nr_revision = len(self.Data.pairs)

        if load:
            self.model = keras.models.load_model(self.model_file)
        else:
            self.model = self.build_model(embedding_size=embedding_size, optimizer=optimizer,
                                          classification=classification)
            self.train(nb_epochs=nb_epochs, batch_size=batch_size, negative_ratio=negative_ratio, save_model=save)

        # print(self.crossValidation(nb_epochs=epochs, k_folds=kfolds, save_model=save))
        # y_true, y_pred = self.test()
        # self.evaluate_classification(y_true, y_pred)

    def build_model(self, embedding_size=50, optimizer='Adam', classification=True):
        """
        Build model architecture/framework
        :return: model
        """

        from keras.layers import Input, Embedding, Dot, Reshape, Dense
        from keras.models import Model

        # Both inputs are 1-dimensional
        file = Input(name='file', shape=[1])
        test = Input(name='test', shape=[1])

        # Embedding the book (shape will be (None, 1, 50))
        file_embedding = Embedding(name='file_embedding',
                                   input_dim=len(self.Data.file_index),
                                   output_dim=embedding_size)(file)

        # Embedding the link (shape will be (None, 1, 50))
        test_embedding = Embedding(name='test_embedding',
                                   input_dim=len(self.Data.test_index),
                                   output_dim=embedding_size)(test)

        # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
        merged = Dot(name='dot_product', normalize=True, axes=2)([file_embedding, test_embedding])

        # Reshape to be a single number (shape will be (None, 1))
        merged = Reshape(target_shape=[1])(merged)

        # If classification, add extra layer and loss function is binary cross entropy
        if classification:
            merged = Dense(1, activation='sigmoid')(merged)
            model = Model(inputs=[file, test], outputs=merged)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Otherwise loss function is mean squared error
        else:
            model = Model(inputs=[file, test], outputs=merged)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        model.summary()
        return model

    def train(self, nb_epochs=10, batch_size: int = 10, negative_ratio=1, save_model=False, plot=False):
        """
        Train model.
        :param batch_size:
        :param plot: If true accuracy vs loss is plotted for training and validation set
        :param n_positive:
        :param negative_ratio: Ratio of positive vs. negative labels. Positive -> there is link between files.
        Negative -> no link
        :param save_model: If true model is saved as a .h5 file
        :param validation_set_size: percentage of whole dataset for validation
        :param training_set_size: percentage of whole dataset for training
        :param nb_epochs: Number of epochs
        :return:
        """
        # Generate training set
        training_set = self.Data.pairs

        train_gen = DataGenerator(pairs=training_set, batch_size=batch_size, nr_files=len(self.Data.all_files),
                                  nr_tests=len(self.Data.all_tests), negative_ratio=negative_ratio)

        # Train
        self.model.fit(train_gen,
                       epochs=nb_epochs,
                       verbose=2)
        if plot:
            self.plot_acc_loss(self.model)
        if save_model:
            self.model.save(self.model_file)

    def predict(self, pickle_file: str = None):
        """
        Makes model prediction for unseen data.
        :param pickle_file:
        :return:
        """
        apfd = []
        #fdr = []
        data = self.Data.df_unseen
        data = data.explode('name')
        data = data.explode('mod_files')

        grouped = data.groupby(['revision'])

        for name, group in grouped:  # for each revision
            preds_per_files = []
            tests = group['name'].to_list()
            labels = []
            duration = []
            for t in self.Data.all_tests:
                duration.append(self.Data.test_duration[t])
                if t in tests:
                    labels.append(1)
                else:
                    labels.append(0)
            for row in group.iterrows():  # for each file
                unseen_pairs = []
                for t in self.Data.all_tests:  # pair with every test
                    if row[1]['mod_files'] in self.Data.all_files:
                        unseen_pairs.append((self.Data.file_index[row[1]['mod_files']], self.Data.test_index[t]))

                def generate_predictions(pairs, batch_size):
                    batch = np.zeros((batch_size, 2))
                    while True:
                        for idx, (file_id, test_id) in enumerate(pairs):
                            batch[idx, :] = (file_id, test_id)

                        # Increment idx by 1
                        idx += 1

                        yield {'file': batch[:, 0], 'test': batch[:, 1]}

                if unseen_pairs:
                    x = next(generate_predictions(unseen_pairs, len(unseen_pairs)))
                    preds_per_files.append(self.model.predict(x))

            pred = [max(idx) for idx in zip(*preds_per_files)]  # return maximum score of test

            prioritization = [x for _, x in sorted(zip(pred, labels), reverse=True)] # Reorder test case list
            #duration = [x for _, x in sorted(zip(pred, duration), reverse=True)]

            apfd.append(self.apfd(prioritization)) # calculate apfd
            #fdr.append(self.fdr(prioritization, duration))

            print(f'APFD -> {np.round(self.apfd(prioritization), 2)}')
            print(f'FDR -> {np.round(self.fdr(prioritization, duration), 2)}')

        df = pd.DataFrame({'apfd': apfd},
                          columns=['apfd'])

        if pickle_file is not None:
            df.to_pickle(pickle_file)

        return df

    def test(self, negative_ratio=1.0):
        # Generate training set
        test_set = self.Data.unseen_pairs

        test_gen = DataGenerator(pairs=test_set, batch_size=10, nr_files=len(self.Data.all_files),
                                 nr_tests=len(self.Data.all_tests), negative_ratio=negative_ratio)

        X, y = next(test_gen.data_generation(test_set))
        pred = self.model.predict(X)

        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        return y, pred

    def evaluate_classification(self, y, pred):
        """
        Provide Classification report and metrics
        :param y:
        :param pred:
        :return:
        """
        print(' Evaluating Network...')
        print(f' Test set accuracy - {np.round(100 * self.accuracy(y, pred), 1)}')
        print(self.report(y, pred))
        print(self.cnf_mtx(y, pred))

    def extract_weights(self, name):
        """
        Extract weights from a neural network model
        :param name:
        :return:
        """
        # Extract weights
        weight_layer = self.model.get_layer(name)
        weights = weight_layer.get_weights()[0]

        # Normalize
        weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
        return weights

    def get_components(self, method='TSNE'):
        """
        Extract 2 components from multi-dimensional manifold
        :param method:
        :return:
        """
        file_weight_class = self.extract_weights('file_embedding')
        test_weight_class = self.extract_weights('test_embedding')

        file_r = reduce_dim(file_weight_class, components=2, method=method)
        test_r = reduce_dim(test_weight_class, components=2, method=method)
        return file_r, test_r

    def get_file_labels(self):
        """
        Creates pairs of (file, file label) for color plot
        :return: (files, file labels)
        """
        pjs = []
        for row in self.Data.df_link.iterrows():
            for item in row[1]['mod_files']:
                pjs.append((item, item.split('/')[0]))
        return list(set(pjs))

    def get_test_labels(self):
        """
        Creates pairs of (test, test label) for color plot
        :return: (tests, tests labels)
        """
        tst = []
        for row in self.Data.df_link.iterrows():
            for item in row[1]['name']:
                label = item.split('_')
                if len(label) > 2:
                    tst.append((item, label[2]))
                else:
                    tst.append((item, 'Other'))
        return list(set(tst))

    def plot_embeddings(self, method='TSNE'):
        """
        Plots file and tests embeddings side by side without labels, with the corresponding dim reduction method.
        :param method: TSNE or UMAP
        :return: NoneType
        """
        # Embeddings
        files, tests = self.get_components(method=method)
        self.plot_embed_both(files, tests, method=method)

    def plot_embeddings_labeled(self, layer='tests', method='TSNE'):
        """
        Plots file or test embedding with corresponding label, for the 10 most frequent items.
        :param layer: File or Test layer
        :param method: TSNE or UMAP
        :return:
        """
        if layer == 'tests':
            tst_labels = self.get_test_labels()
            _, test_r = self.get_components(method=method)
            self.plot_embed_tests(tst_label=tst_labels, test_r=test_r, method=method)
        elif layer == 'files':
            file_labels = self.get_file_labels()
            file_r, _ = self.get_components(method=method)
            self.plot_embed_files(file_r=file_r, pjs_labels=file_labels, method=method)

    def plot_model(self, show_shapes: bool = True):
        """
        Plots and saves Keras model schema
        :param show_shapes:
        :return:
        """
        keras.utils.plot_model(
            self.model,
            to_file="model.png",
            show_shapes=show_shapes
        )

    def crossValidation(self, k_folds=10, nb_epochs=10, n_positive=500, negative_ratio=3.0, save_model=False):
        # Training with K-fold cross validation
        kf = KFold(n_splits=k_folds, random_state=None, shuffle=True)
        kf.get_n_splits(self.Data.pairs)
        X = np.array(self.Data.pairs)

        cv_accuracy_train = []
        cv_accuracy_val = []
        cv_loss_train = []
        cv_loss_val = []

        i = 1
        for train_index, test_index in kf.split(X):
            training_set = X[train_index]
            validation_set = X[test_index]

            print(training_set)
            exit()

            print("=========================================")
            print("====== K Fold Validation step => %d/%d =======" % (i, k_folds))
            print("=========================================")

            train_gen = DataGenerator(pairs=training_set, negative_ratio=negative_ratio)

            val_gen = DataGenerator(pairs=validation_set, negative_ratio=negative_ratio)

            # Train
            h = self.model.fit(train_gen,
                               validation_data=val_gen,
                               epochs=nb_epochs,
                               verbose=2)

            cv_accuracy_train.append(np.array(h.history['accuracy'])[-1])
            cv_accuracy_val.append(np.array(h.history['val_accuracy'])[-1])
            cv_loss_train.append(np.array(h.history['loss'])[-1])
            cv_loss_val.append(np.array(h.history['val_loss'])[-1])

            i += 1

        if save_model:
            self.model.save(self.model_file)

        return np.mean(cv_accuracy_train), np.mean(cv_loss_train), \
               np.mean(cv_accuracy_val), np.mean(cv_loss_val)