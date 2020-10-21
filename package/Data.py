import collections
from abc import abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd

from Visualizer import Visualizer


class Data:
    """
    Abstract class for Data to solve Test Case Prioritization Problem.
    """
    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def create_data_input(self):
        pass

    @abstractmethod
    def get_data_info(self):
        pass


class DataCI(Data, Visualizer):
    """
    Receives, transforms, analyzes and cleans raw data.
    into pairs of (files, tests), by iterating through every
    revision and combining every single file with every single test.
    """

    def __init__(self, commits, test_details, test_history, mod_files, start_date: str = '2019-03-11 00:00:00.000000',
                 threshold: int = 5, threshold_pairs: int = 0, predict_len: int = 100):
        """
        Data Class Constructor: Reads raw data, transforms it, cleans it and reshapes to match desired output.
        :param commits:
        :param test_details:
        :param test_history:
        :param mod_files:
        """
        # Read data
        self.commits = commits
        self.test_details = test_details
        self.test_history = test_history
        self.mod_files = mod_files
        self.predict_len = predict_len
        # self.test_duration = pd.Series(self.test_details.duration.values, index=self.test_details.name).to_dict()

        # Data Cleaning
        self.transform()
        self.clean_files(start_date=start_date)
        self.df_link = self.create_data_input()
        self.clean_tests(threshold=threshold)

        # Create input pairs
        self.df_unseen = self.df_link.tail(self.predict_len) # remove last df rows for predicting on unseen data
        self.df_link.drop(self.df_link.tail(self.predict_len).index, inplace=True)

        self.pairs = self.create_pairs(data=self.df_link)
        self.unseen_pairs = self.create_pairs(data=self.df_unseen)
        self.clean_pairs(threshold_pairs=threshold_pairs)

        self.update_pairs()

        P = {}
        for k, v in self.pairs.items():
            if v != []:
                P[k] = [(self.file_index[t[0]], self.test_index[t[1]]) for t in v]
        self.pairs = P

        # encode new predictions
        self.unseen_pairs = self.create_pairs(data=self.df_unseen)
        for k, v in self.unseen_pairs.items():
            if v != []:
                try: # if a file was never seen before
                    self.unseen_pairs[k] = [(self.file_index[t[0]], self.test_index[t[1]]) for t in v]
                except Exception:
                    pass

        self.get_data_info()

    def transform(self):
        """
        For commit list, removes missing values and converts timestamp to datetime
        For test details, replaces None value durations by 2 minutes. (average duration)
        For test history, drops NaN values.
        For modified files, splits string into list of strings and cleans file path to format dir/filename
        :return:
        """
        # commit list
        self.commits['changes'] = self.commits['changes'].map(lambda x: x.lstrip('#').rstrip('aAbBcC'))
        self.commits[['nbModifiedFiles', 'nbAddedFiles']] = self.commits['changes'].str.split('+', expand=True)
        self.commits[['nbModifiedFiles', 'nbRemovedFiles']] = self.commits['nbModifiedFiles'].str.split('-',
                                                                                                        expand=True)
        self.commits[['nbAddedFiles', 'nbRemovedFiles']] = self.commits['nbAddedFiles'].str.split('-', expand=True)
        self.commits[['nbModifiedFiles', 'nbMovedFiles']] = self.commits['nbModifiedFiles'].str.split('~', expand=True)
        self.commits[['nbRemovedFiles', 'nbMovedFiles']] = self.commits['nbRemovedFiles'].str.split('~', expand=True)

        # Missing and Empty Values
        self.commits.nbModifiedFiles.fillna(value='0', inplace=True)
        self.commits.nbModifiedFiles.replace('', '0', inplace=True)
        self.commits.nbAddedFiles.fillna(value=0, inplace=True)
        self.commits.nbAddedFiles.replace('', '0', inplace=True)
        self.commits.nbRemovedFiles.fillna(value=0, inplace=True)
        self.commits.nbRemovedFiles.replace('', '0', inplace=True)
        self.commits.comment.fillna(value='no_comment', inplace=True)
        self.commits.nbMovedFiles.fillna(value=0, inplace=True)
        self.commits.nbMovedFiles.replace('', '0', inplace=True)

        self.commits.timestamp = pd.to_datetime(self.commits.timestamp, format='%Y-%m-%d %H:%M:%S')
        self.commits = self.commits[['rev', 'user', 'timestamp', 'nbModifiedFiles', 'nbAddedFiles',
                                     'nbRemovedFiles', 'nbMovedFiles', 'comment']]

        # test details list - replace None test duration
        self.test_details.duration = self.test_details.duration.replace(to_replace='None', value=2)
        self.test_details['duration'] = pd.to_datetime(self.test_details['duration']).sub(
            pd.Timestamp('00:02:00')).dt.seconds
        self.test_details['duration'] = np.round(self.test_details['duration'] / 60, 3)
        self.test_details['name'] = self.test_details['name'].str.replace(',',
                                                                          ';')  # replace every comma in test name for char

        # Test history list - remove revisions without label
        i = self.test_history[(self.test_history.revision == 'None')].index
        self.test_history = self.test_history.drop(i)

        # transform modified files list
        self.mod_files['mod_files'] = self.mod_files['mod_files'].apply(lambda x:
                                                                        x.strip("[]").replace("'", "").split(", "))

        def get_filename(column):
            """
            Filename is of the form ABC/DEF/GHI/JKL and function trims down to GHI/JKL
            :param column:
            :return: list
            """
            import os
            li = []
            for i in column:
                file = os.path.basename(i)
                path = os.path.normpath(i)
                if len(path.split(os.sep)) > 1:
                    li.append(path.split(os.sep)[-2] + "/" + file)
                else:
                    li.append(str(path.split(os.sep[0])) + "/" + file)
            return li

        self.mod_files['mod_files'] = self.mod_files['mod_files'].apply(
            lambda x: get_filename(x))  # remove full path of file
        self.mod_files['mod_files'] = self.mod_files['mod_files'].apply(
            lambda x: list(pd.unique(x)))  # remove duplicate files on each commit

    def clean_files(self, start_date: str = '2019-03-11 00:00:00.000000'):
        """
        Some files present in the data are deprecated or unused. Thus we only want to keep relevant files.
        To do that, only files that have been modified in the past year are stored, otherwise it gets removed.
        :param start_date: Date threshold to remove modified files
        :return:
        """

        # print(f'Removing files that are not modified since {start_date}')
        all_files = list(self.mod_files['mod_files'].explode().unique())
        # print(f'There are {len(all_files)} files before cleaning')
        d = {k: v for v, k in enumerate(all_files)}
        index_file = {idx: file for file, idx in d.items()}

        def encode_files(m: list):
            idx = [d[k] for k in m]  # match indexes of tests in current revision
            return list(set(idx))

        self.mod_files['mod_files'] = self.mod_files['mod_files'].apply(lambda x:
                                                                        encode_files(x))

        self.mod_files['timestamp'] = pd.to_datetime(self.mod_files['timestamp'])
        dt = pd.to_datetime(start_date)
        start = self.mod_files['timestamp'].sub(dt).abs().idxmin()

        file_list = list(range(0, len(all_files)))

        # get list of files not modified since 2019
        for i in self.mod_files.iloc[start:]['mod_files']:
            for f in file_list:
                if f in i:
                    file_list.remove(f)

        def remove_old_files(t: list):
            l1 = [x for x in t if x not in file_list]
            if l1:
                return l1
            else:
                return None

        # remove non relevant files from mod_files column
        self.mod_files['mod_files'] = self.mod_files['mod_files'].apply(lambda t: remove_old_files(t))
        self.mod_files.dropna(inplace=True)  # Drop None values

        def recover_files(m: list):
            idx = [index_file[k] for k in m]  # match indexes of tests in current revision
            return idx

        self.mod_files['mod_files'] = self.mod_files['mod_files'].apply(lambda t: recover_files(t))

    def create_data_input(self):
        """
        Creates unified input data for ML algorithm, where columns are (revision, mod_files and test names)
        :return: df
        """
        self.test_history = self.test_history.merge(self.test_details[['name', 'id']], how='inner', left_on='test_id',
                                                    right_on='id')

        self.test_history = self.test_history.groupby(['revision'])['name'].apply(', '.join).reset_index()  # by name
        self.test_history['revision'] = self.test_history['revision'].astype(int)
        self.test_history = self.test_history.sort_values(by=['revision'])
        self.test_history = self.test_history.reset_index()
        self.test_history.drop(columns=['index'], inplace=True)

        self.test_history['name'] = self.test_history['name'].apply(lambda x: x.split(', '))
        self.test_history['name'] = self.test_history['name'].apply(lambda x: list(pd.unique(x)))

        df = self.test_history.merge(self.mod_files, how='inner', on='revision')
        return df[['revision', 'mod_files', 'name']]

    def clean_tests(self, threshold: int = 5):
        """
        Drops tests/files from data that cause less than threshold
        transitions/modifications, i.e. very stable tests/files.
        """
        col = ['mod_files', 'name']
        for c in col:
            # Count test frequency
            dist = self.df_link[c].explode().values
            dist = collections.Counter(dist)

            # Threshold
            good_items = [k for k, v in dist.items() if float(v) >= threshold]

            # Remove test below threshold from data
            def remove_stable_items(t: list):
                l1 = [x for x in t if x in good_items]
                if l1:
                    return l1
                else:
                    return None

            self.df_link[c] = self.df_link[c].apply(lambda t: remove_stable_items(t))

            # Drop None values
            self.df_link.dropna(inplace=True)

    def create_pairs(self, data):
        """
        Each row of the dataset corresponds to 1 revisions, composed of lists of modified files and tests.
        This function explodes the lists and forms pair-wise combinations between items in both lists.
        :param data:
        :return: pairs: dict of tuples of pairs (test, file). Each key is a revision.
        """
        data = data.explode('name')
        data = data.explode('mod_files')
        grouped = data.groupby(['revision'])

        pairs = defaultdict(list)
        for name, group in grouped:
            for row in group.iterrows():
                pairs[name].append((row[1]['mod_files'], row[1]['name']))
        return pairs

    def clean_pairs(self, threshold_pairs: int):
        """
        Remove pairs that are very rare, thus very unlikely representing a real connection in the data. They most likely
        occurred by chance.
        :param threshold_pairs:
        :return:
        """
        all_tuples = [t for k, v in self.pairs.items() for t in v]
        C = collections.Counter(all_tuples)

        for k, v in self.pairs.items():
            self.pairs[k] = [t for t in v if C[t] > threshold_pairs]

    def update_pairs(self):
        # count number of distinct directories and files
        self.all_pairs = [t for k, v in self.pairs.items() for t in v]
        self.files = [file for file, test in self.all_pairs]
        self.tests = [test for file, test in self.all_pairs]

        self.all_tests = list(np.unique(self.tests))
        self.all_files = list(np.unique(self.files))

        # encode files and test to ints
        self.file_index = {file: idx for idx, file in enumerate(self.all_files)}
        self.index_file = {idx: file for file, idx in self.file_index.items()}

        self.test_index = {test: idx for idx, test in enumerate(self.all_tests)}
        self.index_test = {idx: test for test, idx in self.test_index.items()}
        print(f'There are {len(self.all_files)} unique files and {len(self.all_tests)}')

    def get_data_info(self):
        """
        Prints stats about the transformation steps applied to the raw dataset
        :return:
        """
        # revisions
        print(f'Nr of Revisions {len(self.pairs)}')

        # Files
        print(f'\nNumber of Files - {len(self.all_files)}')

        dist = collections.Counter(self.files)
        mpf = sum(dist.values()) / len(dist.keys())
        print(f'   Total number of modifications - {sum(dist.values())}')
        print(f'   Average number of modification per file - {mpf}')

        # Tests
        print(f'\nNumber of Tests - {len(self.all_tests)}')

        # Count test frequency
        dist = collections.Counter(self.tests)
        tpt = sum(dist.values()) / len(dist.keys())
        print(f'   Total number of transitions - {sum(dist.values())}')
        print(f'   Average number of transitions per test - {tpt}')

        return mpf, tpt

    def to_csv(self):
        """
        Converts dataframe to csv.
        :return:
        """
        self.df_link.to_csv('../pub_data/df.csv')
