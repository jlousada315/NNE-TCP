import random
import numpy as np
from tensorflow import keras

random.seed()


class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras Model. Receives pairs of (files, tests) and generates random combinations of
    (file, test, label). Where the label is either 1 if the pair exists in the data or 0 otherwise. The class balance
    is given by the negative_ratio parameter, if parameter is 1, class balance is 50% each.
    Parameter splits gives the percentage attributed to (training_set, validation_set, test_set)
    """

    def __init__(self, pairs, nr_files, nr_tests, negative_ratio=1.0, batch_size=10, classification=True, shuffle=True):
        """
        Data Generator constructor.
        :param pairs:
        :param negative_ratio:
        :param classification:
        :param shuffle:
        """
        self.pairs = pairs
        self.nr_files = nr_files
        self.nr_tests = nr_tests
        self.negative_ratio = negative_ratio

        self.batch_size = batch_size

        self.shuffle = shuffle
        self.classification = classification

        self.on_epoch_end()

    def __len__(self):
        """
        Steps needed to complete one epoch, i.e. go through the entire dataset
        :return:
        """
        return len(self.pairs) // self.batch_size

    def on_epoch_end(self):
        """
        When epoch is finished shuffle indexes
        :return:
        """
        self.indexes = np.array(list(self.pairs.keys()))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """
        Returns data generated in one batch
        :param index:
        :return:
        """
        # Generate indexes of the batch
        index = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        batch = [self.pairs[k] for k in index][0]
        # Generate data
        X, y = next(self.data_generation(batch))
        return X, y

    def data_generation(self, pairs):
        """Generate batches of samples for training"""
        batch_size = int(len(pairs) * (1 + self.negative_ratio))
        batch = np.zeros((batch_size, 3))
        pairs_set = list(set(pairs))

        # Adjust label based on task
        if self.classification:
            neg_label = 0
        else:
            neg_label = -1

        # This creates a generator
        while True:
            for idx, (file_id, test_id) in enumerate(pairs):
                batch[idx, :] = (file_id, test_id, 1)

            # Increment idx by 1
            idx += 1

            # Add negative examples until reach batch size
            while idx < batch_size:

                # random selection
                random_file = random.sample(pairs, 1)[0][0]
                # random_test = random.randrange(self.nr_tests) # to select random test
                random_test = random.randrange(self.nr_tests)

                # Check to make sure this is not a positive example
                if (random_file, random_test) not in pairs_set:
                    # Add to batch and increment index
                    batch[idx, :] = (random_file, random_test, neg_label)
                    idx += 1

            np.random.shuffle(batch)
            yield {'file': batch[:, 0], 'test': batch[:, 1]}, batch[:, 2]
