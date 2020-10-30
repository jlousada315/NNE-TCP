import glob
import os

from sklearn.model_selection import ParameterGrid

from Data import DataCI
from Prioritizer import NNEmbeddings
import pandas as pd
from matplotlib.pylab import plt
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid")


def data_clean_analysis(dates, thresholds, thresholds_pairs):
    """
    Calculate file/test density for several combinations of data cleaning steps
    :param dates:
    :param thresholds:
    :param thresholds_pairs:
    :return:
    """
    mpf = []
    tpt = []
    date = []
    thresh = []
    thresh_pairs = []

    for k, v in dates.items():
        for t in thresholds:
            for tp in thresholds_pairs:
                print(k)
                print(t)
                print(tp)
                print('-----')

                commits = pd.read_csv('../pub_data/test_commits_pub.csv', encoding='latin-1', sep='\t')
                test_details = pd.read_csv('../pub_data/test_details_pub.csv', sep='\t')
                test_status = pd.read_csv('../pub_data/test_histo_pub.csv', sep='\t')
                mod_files = pd.read_csv("../pub_data/test_commits_mod_files_pub.csv", sep='\t')

                D = DataCI(commits, test_details, test_status, mod_files, start_date=v, threshold=t, threshold_pairs=tp)
                modification, transition = D.get_data_info()

                mpf.append(modification)
                tpt.append(transition)
                date.append(k)
                thresh.append(t)
                thresh_pairs.append(tp)

    print(len(date))
    print(len(thresh))
    print(len(thresh_pairs))
    print(len(mpf))
    print(len(tpt))

    df = pd.DataFrame(list(zip(date, thresh, thresh_pairs, mpf, tpt)),
                      columns=['date', 'threshold', 'threshold_pairs', 'mpf', 'tpt']
                      )

    df.to_pickle('start_date_analysis1.pkl')


def plot_df(name: str = 'start_date_analysis1.pkl'):
    """
    Plot 2D for file/test density
    :param name:
    :return:
    """
    df = pd.read_pickle(name)
    print(df)
    fig, axarr = plt.subplots(2, 2, sharey=True, sharex=True)
    df = df.iloc[::-1]

    plt.suptitle('Threshold Start Date Analysis', fontsize=14)

    for idx, row in enumerate(sorted(df['threshold_pairs'].unique())):
        data = df[df['threshold_pairs'] == row]

        if idx == 0 or idx == 1:
            column = 0
        else:
            column = 1

        sns.lineplot(x="date", y="mpf", hue="threshold", data=data,
                     palette='tab10', ax=axarr[idx % 2, column])
        sns.lineplot(x="date", y="tpt", hue="threshold", data=data,
                     palette='tab10', ax=axarr[idx % 2, column], linestyle='--', legend=False)

        axarr[idx % 2, column].set_xlabel('Start Date')
        axarr[idx % 2, column].set_ylabel('Frequency')
        axarr[idx % 2, column].set_title(f'Pairs Threshold - {row}')
        axarr[idx % 2, column].legend(loc='center left')

    # plot vertical line
    # plt.axvline(x=3, linestyle='-.', label='Optimal Value')

    # plt.tight_layout()
    plt.show()


def surface_plot(name: str = 'start_date_analysis1.pkl'):
    """
    3D plot of data cleaning steps
    :param name:
    :return:
    """
    df = pd.read_pickle(name)

    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # ===============
    #  First subplot
    # ===============
    # set up the axes for the first plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title('Modifications per File')
    ax.set_xlabel('Date (Months)')
    ax.set_ylabel('Threshold Individual')
    for idx, row in enumerate(sorted(df['threshold_pairs'].unique())):
        data = df[df['threshold_pairs'] == row]
        label = 'Threshold pairs ' + str(row)
        # Plot the surface.
        surf = ax.plot_trisurf(data['date'], data['threshold'], data['mpf'], alpha=0.7,
                               linewidth=0, antialiased=False, label=label)
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d
    # ===============
    # Second subplot
    # ===============
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title('Transitions per Test')
    ax.set_xlabel('Date (Months)')
    ax.set_ylabel('Threshold Individual')
    for idx, row in enumerate(sorted(df['threshold_pairs'].unique())):
        data = df[df['threshold_pairs'] == row]
        label = 'Threshold pairs ' + str(row)
        # Plot the surface.

        surf = ax.plot_trisurf(data['date'], data['threshold'], data['tpt'], alpha=0.7,
                               linewidth=0, antialiased=False, label=label)

        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d

    # cbar = fig.colorbar(surf)
    # cbar.locator = LinearLocator(numticks=10)
    # cbar.update_ticks()

    plt.suptitle('Threshold Start Date Analysis 3D', fontsize=14)
    plt.legend()
    plt.show()


def plot_single(df_metrics):
    """
    APFD plot for single Embedding Neural Network model
    :param df_metrics:
    :return:
    """
    apfd = df_metrics['apfd']

    miu = np.round(np.mean(apfd), 2)
    sigma = np.round(np.std(apfd), 2)
    label = 'regression' + '\n $\mu$ - ' + str(miu) + ' $\sigma$ - ' + str(sigma)

    sns.distplot(apfd, kde=True,
                 bins=int(180 / 5), color=sns.color_palette()[0],
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4, 'clip': (0.0, 1.0)}, label=label)

    plt.legend(frameon=True, loc='upper left', prop={'size': 20})
    plt.xlabel('APFD')

    #plt.title('APFD Distribution - 100 revisions ')
    plt.show()


def plot_metric(df_metrics, name, batch_size=10, epochs=10):
    """
    Parameter tuning plots with several subplots
    :param df_metrics:
    :param name:
    :param batch_size:
    :param epochs:
    :return:
    """

    # One groupplot
    fig, axarr = plt.subplots(3, 4, sharey=True, sharex=True)
    plotname = 'apfd'
    subplot_labels = ['(a)', '(b)', '(c)']

    for column, nr in enumerate(sorted(df_metrics['negative_ratio'].unique())):
        for row, emb_size in enumerate(df_metrics['emb_size'].unique()):
            for agidx, (labeltext, task, linestyle) in enumerate(
                    [('Classification', 'True', '-'), ('Regression', 'False', '-.')]):
                rel_df = df_metrics[
                    (df_metrics['emb_size'] == str(emb_size)) & (df_metrics['negative_ratio'] == str(nr)) &
                    (df_metrics['batch_size'] == str(batch_size)) & (df_metrics['epochs'] == str(epochs))]

                # rel_df[rel_df['agent'] == agent].plot(x='step', y='napfd', label=labeltext, ylim=[0, 1], linewidth=0.8,
                #                                      style=linestyle, color=sns.color_palette()[agidx], ax=axarr[row,column])

                apfd = rel_df.loc[rel_df['classification'] == task, 'apfd']
                miu = np.round(np.mean(apfd), 2)
                sigma = np.round(np.std(apfd), 2)
                label = labeltext + '\n $\mu$ - ' + str(miu) + ' $\sigma$ - ' + str(sigma)

                # sns.displot(data=rel_df, x="apfd", hue='classification', kde=True, ax=axarr[row, column])

                sns.distplot(apfd, kde=True,
                             bins=int(180 / 5), color=sns.color_palette()[agidx],
                             hist_kws={'edgecolor': 'black'},
                             kde_kws={'linewidth': 4, 'clip': (0.0, 1.0)}, label=label, ax=axarr[row, column])

                axarr[row, column].xaxis.grid(True, which='major')

                axarr[row, column].set_title('Emb_size - %s - Neg_Ratio - %s' % (emb_size, nr), fontsize=10)

                if row == 2:
                    axarr[row, column].set_xlabel('APFD')
                if column == 0:
                    axarr[row, column].set_ylabel('Density')

                axarr[row, column].legend(frameon=True, prop={'size': 6})

    # Tweak spacing to prevent clipping of ylabel
    fig.suptitle('APFD Parameter Tuning - %d Epochs and batch-size - %d' % (epochs, batch_size))
    fig.tight_layout()
    plt.savefig(name, bbox_inches='tight')
    plt.show()


def load_stats_dataframe(files, aggregated_results=None):
    """
    Load pickle files and transform to dataframe.
    :param files:
    :param aggregated_results:
    :return:
    """
    if os.path.exists(aggregated_results) and all(
            [os.path.getmtime(f) < os.path.getmtime(aggregated_results) for f in files]):
        return pd.read_pickle(aggregated_results)

    df = pd.DataFrame()
    for f in files:
        tmp_dict = pd.read_pickle(f)
        tmp_dict['emb_size'] = f.split('_')[2]
        tmp_dict['negative_ratio'] = f.split('_')[4]
        tmp_dict['batch_size'] = f.split('_')[6]
        tmp_dict['epochs'] = f.split('_')[8]
        tmp_dict['classification'] = f.split('_')[-1].split('.')[0]

        tmp_df = pd.DataFrame.from_dict(tmp_dict)
        df = pd.concat([df, tmp_df])

    if aggregated_results:
        df.to_pickle(aggregated_results)

    return df


def parameter_tuning(D, param_grid):
    """
    Train model with different combinations of parameters
    :param D:
    :param param_grid:
    :return:
    """
    grid = ParameterGrid(param_grid)

    for params in grid:
        model_file = 'Theshpairs1_Ind_5' + '_emb_' + str(params['embedding_size']) + '_nr_' + str(
            params['negative_ratio']) + \
                     '_batch_' + str(params['batch_size']) + '_epochs_' \
                     + str(params['nb_epochs']) + '_classification_' + str(params['classification'])

        print(model_file)

        # Train Model
        Prio = NNEmbeddings(D, embedding_size=params['embedding_size'], negative_ratio=params['negative_ratio'],
                            nb_epochs=params['nb_epochs'], batch_size=params['batch_size'],
                            classification=params['classification'], save=True,
                            model_file='Models/' + model_file + '.h5')

        # New Predicitons
        df_metrics = Prio.predict(pickle_file=None)
        plot_single(df_metrics)
        plot_metric(df_metrics, name='Plot_Metrics/' + model_file + '.png')


def get_df_metrics():
    """
    Collect all pickle files containing metrics and transform them into dataframe
    :return: df: pd.Dataframe
    """
    DATA_DIR = 'metrics'
    search_pattern = '*.pkl'
    filename = 'stats'

    iteration_results = glob.glob(os.path.join(DATA_DIR, search_pattern))
    aggregated_results = os.path.join(DATA_DIR, filename)

    df = load_stats_dataframe(iteration_results, aggregated_results)
    print(f'Dataframe {df}')
    return df


def new_model(D: DataCI, params: dict, model_file: str, save: bool = False, load: bool = False):
    """
    Train or load existing model.
    :param D:
    :param params:
    :param model_file:
    :param save:
    :param load:
    :return:
    """
    if load:
        # Load existing trained model
        return NNEmbeddings(D=D, load=load, model_file=model_file)
    else:
        # Train New Model
        return NNEmbeddings(D, embedding_size=params['embedding_size'], negative_ratio=params['negative_ratio'],
                            nb_epochs=params['nb_epochs'], batch_size=params['batch_size'],
                            optimizer=params['optimizer'],
                            classification=params['classification'], save=save, model_file=model_file)


def model(Prio: NNEmbeddings, plot_emb: bool = False, pickle_file: str = None):
    """
    Make predictions and plots on unseen data.
    :param Prio:
    :param plot_emb:
    :return:
    """
    # New Predicitons
    df_metrics = Prio.predict(pickle_file=pickle_file)
    plot_single(df_metrics)

    if plot_emb:
        # TSNE Plots
        Prio.plot_embeddings()
        Prio.plot_embeddings_labeled(layer='tests')
        Prio.plot_embeddings_labeled(layer='files')

        # UMAP Plots
        Prio.plot_embeddings(method='UMAP')
        Prio.plot_embeddings_labeled(layer='tests', method='UMAP')
        Prio.plot_embeddings_labeled(layer='files', method='UMAP')


def main():
    commits = pd.read_csv('../pub_data/test_commits_pub.csv', encoding='latin-1', sep='\t')
    test_details = pd.read_csv('../pub_data/test_details_pub.csv', sep='\t')
    test_status = pd.read_csv('../pub_data/test_histo_pub.csv', sep='\t')
    mod_files = pd.read_csv("../pub_data/test_commits_mod_files_pub.csv", sep='\t')

    # Simple start
    nr_revs = 100 # nr of revs to test algorithm
    D = DataCI(commits, test_details, test_status, mod_files, predict_len=nr_revs, threshold_pairs=1, threshold=5) # create training set

    model_file = 'Models/CVTheshpairs1_Ind_5_sgd_emb_200_nr_1_batch_1_epochs_10_classification_False.h5'
    N = NNEmbeddings(D=D, load=True, model_file=model_file)

    # Plot Results
    model(N, plot_emb=True)


    """
    tests
   
    dates = {'3': '2019-09-11 00:00:00.000000',
             '6': '2019-06-11 00:00:00.000000',
             '12': '2019-03-11 00:00:00.000000',
             '18': '2018-09-11 00:00:00.000000',
             '24': '2018-03-11 00:00:00.000000',
             '36': '2017-09-11 00:00:00.000000',
             }

    thresholds = [0, 2, 5, 10]
    thresh_pairs = [0, 1, 2, 5]

    # data clean analysis
    # data_clean_analysis(dates=dates, thresholds=thresholds, thresholds_pairs=thresh_pairs)
    # surface_plot()
    # plot_df()

    # Parameter Grid
    param_grid = {'embedding_size': 200,
                  'negative_ratio': 1,
                  'batch_size': 1,
                  'nb_epochs': 10,
                  'classification': True,
                  'optimizer': 'sgd'
                  }

    # Create New NNEmbedding instance
    #model_file = 'Models/CVTheshpairs1_Ind_5_sgd_emb_200_nr_1_batch_1_epochs_10_classification_False.h5'
    #N = NNEmbeddings(D=D, model_file=model_file, optimizer='sgd')

    #df = pd.read_pickle('cv_scores.pkl')
    #N.plot_acc_loss(df)

    #model(Prio=N,
    #     pickle_file='metrics/CVTheshpairs1_Ind_5_sgd_emb_200_nr_1_batch_1_epochs_10_classification_False.pkl',
    #     plot_emb=True)

    # plot calculated metrics
    # df = get_df_metrics()
    # plot_metric(df, epochs=100, batch_size=1, name='apfd_emb_size_200_epochs_100_batch_size_1_regression')

    # parameter_tuning(D)

    plot_single(pd.read_pickle('metrics/_emb_100_nr_2_batch_5_epochs_10_classification_True.pkl'))
    """


if __name__ == '__main__':
    main()
