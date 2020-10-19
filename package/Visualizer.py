from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import numpy as np
import pandas as pd
import collections
plt.style.use('fivethirtyeight')


class Visualizer:

    def plot_function(self, x_min: float, x_max: float, f):
        x = np.arange(x_min, x_max)
        source = pd.DataFrame({
            'x': x,
            'f(x)': f
        })

        alt.Chart(source).mark_line().encode(
            x='x',
            y='f(x)'
        )

    def bar_plot(self, labels: list, values: list):
        source = pd.DataFrame({
            'a': labels,
            'b': values
        })

        alt.Chart(source).mark_bar().encode(
            x='a',
            y='b'
        )

    def layered_histogram(self, source):
        alt.Chart(source).transform_fold(
            source.keys(),
            as_=['Experiment', 'Measurement']
        ).mark_area(
            opacity=0.3,
            interpolate='step'
        ).encode(
            alt.X('Measurement:Q', bin=alt.Bin(maxbins=100)),
            alt.Y('count()', stack=None),
            alt.Color('Experiment:N')
        )

    def histogram(self, distribution: list, bins: int = 100, xlabel: str = None, ylabel: str = None,
                  title: str = None):

        n, bins, patches = plt.hist(x=distribution, bins=bins, color='#0504aa',
                                    alpha=1, rwidth=.9)
        plt.grid(axis='y', alpha=1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        plt.axis([0, 10, 0, np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10])
        plt.show()

    def plot_acc_loss(self, model):
        l = np.array(model.history['loss'])
        lt = np.array(model.history['val_loss'])
        a = np.array(model.history['accuracy'])
        at = np.array(model.history['val_accuracy'])
        e = range(len(l))

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Epochs', fontsize=15)
        ax1.set_ylabel('Loss', color=color, fontsize=15)
        ax1.plot(e, l, color=color, lw=2, label='Train')
        ax1.plot(e, lt, color=color, lw=2, linestyle='--', label='Validation')
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.legend(loc='center right')
        ax1.grid(True)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color, fontsize=15)  # we already handled the x-label with ax1
        ax2.plot(e, a, color=color, lw=2)
        ax2.plot(e, at, color=color, lw=2, linestyle='--')

        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(None)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    def count_items(self, l):
        # Create a counter object
        counts = Counter(l)

        # Sort by highest count first and place in ordered dictionary
        counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        counts = collections.OrderedDict(counts)

        return counts

    def plot_embed_both(self, file, test, method='TSNE'):
        fig = plt.figure(figsize=(10, 8))
        fig.tight_layout()


        f_comp_1 = file[:, 0]
        f_comp_2 = file[:, 1]
        t_comp_1 = test[:, 0]
        t_comp_2 = test[:, 1]

        plt.subplot(1, 2, 1)
        plt.plot(f_comp_1, f_comp_2, 'r.', label='Files ' + method)
        plt.xlabel(method + ' 1')
        plt.ylabel(method + ' 2')
        plt.title('File Embeddings Visualized with ' + method)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(t_comp_1, t_comp_2, 'b.', label='Tests ' + method)
        plt.xlabel(method + ' 1')
        plt.ylabel(method + ' 2')
        plt.title('Test Embeddings Visualized with ' + method)
        plt.legend()

        plt.show()

    def plot_embed_files(self, file_r, pjs_labels, method='TSNE'):
        projects = [file[1] for file in pjs_labels]

        # Remove genres not found
        projects_counts = self.count_items(projects)
        projects_to_include = list(projects_counts.keys())[:10]

        # Remove genres not found
        idx_include = []
        projects = []
        for i, file in enumerate(pjs_labels):
            if file[1] in projects_to_include:
                idx_include.append(i)
                projects.append(file[1])

        ints, gen = pd.factorize(projects)
        gen = np.asarray(gen, dtype=object)

        comp_1 = file_r[idx_include, 0]
        comp_2 = file_r[idx_include, 1]

        # Plot embedding
        plt.figure(figsize=(12, 10))
        plt.scatter(comp_1, comp_2,
                    c=ints, cmap=plt.cm.tab10)

        # Add colorbar and appropriate labels
        cbar = plt.colorbar()
        cbar.set_ticks([])
        for j, lab in enumerate(gen):
            cbar.ax.text(10, (2 * j + 1) / 2.25, lab)
        cbar.ax.set_title('Project', loc='left')

        plt.xlabel(method + ' 1')
        plt.ylabel(method + ' 2')
        plt.title(method + ' Visualization of File Embeddings')

        plt.show()

    def plot_embed_tests(self, tst_label, test_r, method='TSNE'):
        folders = [test[1] for test in tst_label]
        # Remove genres not found
        folders_counts = self.count_items(folders)
        folders_to_include = list(folders_counts.keys())[:11]

        # Remove genres not found
        idx_include = []
        folders = []
        for i, test in enumerate(tst_label):
            if test[1] in folders_to_include and test[1] != 'Other':
                idx_include.append(i)
                folders.append(test[1])

        ints, gen = pd.factorize(folders)

        comp_1 = test_r[idx_include, 0]
        comp_2 = test_r[idx_include, 1]

        # Plot embedding
        plt.figure(figsize=(12, 10))
        plt.scatter(comp_1, comp_2,
                    c=ints, cmap=plt.cm.tab10)

        # Add colorbar and appropriate labels
        cbar = plt.colorbar()
        cbar.set_ticks([])
        for j, lab in enumerate(gen):
            cbar.ax.text(10, (2 * j + 1) / 2.25, lab)
        cbar.ax.set_title('Project', loc='left')

        plt.xlabel(method + ' 1')
        plt.ylabel(method + ' 2')
        plt.title(method + ' Visualization of Test Embeddings')

        plt.show()



