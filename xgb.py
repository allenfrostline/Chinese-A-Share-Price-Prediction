# import necessary libraries
import sys
import numpy as np
import seaborn as sns
import tushare as ts
import pandas as pd
from xgboost import XGBClassifier
from time import sleep
from datetime import datetime as dt
from tech import technical
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score


sns.set(context='paper')


class stock:
    def __init__(self, id):
        self.id = id
        self.train = None
        self.test = None

    def load(self, start='', end='', ratio='', seq_length='', test_size=0.2, verbose=False):

        print('-' * 63)
        # process input params
        if start == '':
            start = None
        if end == '':
            end = dt.today().strftime('%Y%m%d')
        if ratio == '':
            ratio = 0
        else:
            ratio = float(ratio)
        if seq_length == '':
            seq_length = 3
        else:
            seq_length = int(seq_length)
        # set pandas and numpy linewidth for better printing, and numpy random seed for reproduction
        pd.set_option('display.width', 200)
        pd.set_option('display.max_columns', 20)
        np.set_printoptions(linewidth=200, edgeitems=5)

        # load dataset through tushare
        try:  # in case the stock symbol is invalid
            df = ts.get_hist_data(self.id, start=start, end=end)[::-1]
            if start is None:
                start = df.index[0].replace('-', '')
            print('LOADING DATASET FROM TUSHARE: {} FROM {} TO {}.'.format(self.id, start, end))
            tot_len = len(df)
            print('DATASET LOADED, {} DAYS IN TOTAL.'.format(tot_len))
            if verbose is True:
                print(df.head())
        except:
            print('ERROR LOADING DATASET. PERHAPS THE STOCK {} DOES NOT EXIST.'.format(self.id))
            sys.exit(0)
            return None

        # use the technical function to load all the technical data
        tech = technical(df)

        # now we have all the features
        self.features = ['ma10', 'ma20', 'ma5', 'v_ma10', 'v_ma20', 'v_ma5'] + \
            sum([[c + '[T-{}]'.format(i) for c in ['open', 'close', 'high', 'low', 'volume', 'turnover']] for i in range(seq_length)], []) + \
            ['tech{}'.format(i + 1) for i in range(tech.shape[1])]

        # define the label function, ratio required
        def label(df, seq_length):
            return (df['p_change'].values[seq_length:] > ratio).astype(int)

        # split data into X and y
        X = df[['ma10', 'ma20', 'ma5', 'v_ma10', 'v_ma20', 'v_ma5']]
        y = label(df, seq_length)
        X_shift = [X]
        for i in range(seq_length):
            X_shift.append(df[['open', 'close', 'high', 'low', 'volume', 'turnover']].shift(i))
        X = np.concatenate([tech, np.log(pd.concat(X_shift, axis=1).values)], axis=1)[seq_length - 1: -1]
        print('SPLITTING X {} AND Y {}.'.format(X.shape, y.shape))

        # split data into train and test sets
        print('SPLIT THE LAST {:.0f}% OF THE DATA FOR TESTING.'.format(test_size * 100))
        test_len = int(tot_len * test_size)
        train_len = tot_len - test_len
        X_train, X_test = X[:train_len], X[-test_len:]
        y_train, y_test = y[:train_len], y[-test_len:]

        # print a brief table about the train and test dataset
        print('┌────────┬────────┬────────┐')
        print('│   %%   │   ++   │   --   │')
        print('├────────┼────────┼────────┤')
        print('│  TRAIN │ {:>5.2f}% │ {:>5.2f}% │'.format(y_train.sum() / len(y_train) * 100, 100 - y_train.sum() / len(y_train) * 100))
        print('├────────┼────────┼────────┤')
        print('│  TEST  │ {:>5.2f}% │ {:>5.2f}% │'.format(y_test.sum() / len(y_test) * 100, 100 - y_test.sum() / len(y_test) * 100))
        print('└────────┴────────┴────────┘')
        # update the train and test dataset
        self.train = [X_train, y_train]
        self.test = [X_test, y_test]


class classifier:
    def __init__(self):
        self.model = XGBClassifier()
        self.progress = 0

    def para_tuning(self, X, y, para, grid, seed=0, verbose=False):  # verbose = 1 for tuning log, verbose = 2 for plotting, verbose = 3 for both

        # determine which to parameter to tune this time
        if para == '':
            return None
        elif para == 'learning_rate':
            param_grid = dict(learning_rate=grid)  # [0,0.1]
        elif para == 'max_depth':
            param_grid = dict(max_depth=grid)  # int
        elif para == 'min_child_weight':
            param_grid = dict(min_child_weight=grid)  # [0,1]
        elif para == 'gamma':
            param_grid = dict(gamma=grid)  # [0,1]
        elif para == 'max_delta_step':
            param_grid = dict(max_delta_step=grid)  # int
        elif para == 'colsample_bytree':
            param_grid = dict(colsample_bytree=grid)  # [0,1]
        elif para == 'reg_alpha':
            param_grid = dict(reg_alpha=grid)  # [0,1]
        elif para == 'reg_lambda':
            param_grid = dict(reg_lambda=grid)  # [0,1]
        else:
            print('WRONG PARAMETER.')
            return None
        kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)
        grid_search = GridSearchCV(self.model, param_grid, scoring='accuracy', n_jobs=-1, cv=kfold)
        grid_result = grid_search.fit(X, y)
        # summarize results
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        if verbose == 1 or verbose == 3:
            for mean, stdev, param in zip(means, stds, params):
                print('{:.4f} ({:.4f}) WITH: {} = {}'.format(mean, stdev, para, list(param.values())[0]))
            print('-' * 63)
        self.progress += 1
        progress = int(self.progress / 7 * 100)
        progress_bar = int(self.progress / 7 * 58)
        print('\r' + '█' * progress_bar + ' ' * (58 - progress_bar) + ' {:>3}%'.format(progress), end='')
        if verbose == 2 or verbose == 3:
            # plot
            plt.close()
            plt.figure(figsize=(20, 10))
            plt.errorbar(grid, means, yerr=stds)
            plt.title('XGBoost {} Tuning'.format(para))
            plt.xlabel(para)
            plt.ylabel('accuracy')
            plt.show()
        return list(grid_result.best_params_.values())[0]

    def tune(self, X, y, verbose=False, seed=0):
        self.model.seed = seed
        # fit model no training data
        print('-' * 63)
        print('AUTO TUNING ON TRAINING DATASET.')
        self.model.n_estimators = 1024
        self.model.subsample = 0.6
        self.model.learning_rate = 0.01

        self.model.max_depth = self.para_tuning(X, y, 'max_depth', [2, 4, 6, 8], seed, verbose)
        self.model.min_child_weight = self.para_tuning(X, y, 'min_child_weight', [4, 8, 12, 16], seed, verbose)
        self.model.gamma = self.para_tuning(X, y, 'gamma', [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8], seed, verbose)
        self.model.max_delta_step = self.para_tuning(X, y, 'max_delta_step', [0, 1, 2, 4], seed, verbose)
        self.model.colsample_bytree = self.para_tuning(X, y, 'colsample_bytree', [0.5, 0.6, 0.7], seed, verbose)
        self.model.reg_alpha = self.para_tuning(X, y, 'reg_alpha', [0, 0.001, 0.01, 0.1, 10, 100], seed, verbose)
        self.model.reg_lambda = self.para_tuning(X, y, 'reg_lambda', [0, 0.001, 0.01, 0.1, 10, 100], seed, verbose)
        self.model.learning_rate /= 2

        sleep(3)
        print('\rAUTO TUNING FINISHED.' + ' ' * 42)
        print('-' * 63)
        if input('MODEL REVIEWING? (Y/N) ') == 'Y':
            print(self.model)

    def train(self, data, early_stopping_rounds=None, verbose=True, seed=0):
        X_train, y_train = data.train[0], data.train[1]
        X_test, y_test = data.test[0], data.test[1]

        # tune paramters using trainging dataset
        self.tune(X_train, y_train, seed=seed)
        print('-' * 63)
        # train the model with optimized parameters
        print('MODEL TRAINING.')
        metric = ['error', 'logloss', 'auc']
#         self.model.min_child_weight = 4
        self.model.fit(X_train, y_train,
                       eval_metric=metric,
                       eval_set=[(X_train, y_train), (X_test, y_test)],
                       early_stopping_rounds=early_stopping_rounds,
                       verbose=False)

        # make predictions for train data
        y_pred = self.model.predict(X_train)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(y_train, predictions)
        print('TRAINING FINISHED.')
        print('ACCURACY TRAINING: {:.2f}%'.format(accuracy * 100))

        # make predictions for test data
        y_pred = self.model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print('ACCURACY TESTING: {:.2f}%'.format(accuracy * 100))

        if verbose is True:
            try:
                # plot boosting results
                results = self.model.evals_result()
                epochs = len(results['validation_0'][metric[0]])
                x_axis = range(0, epochs)
                plt.style.use('ggplot')
                plt.rcParams['font.size'] = 8
                plt.figure(figsize=(20, 10))
                i = 0
                for m in metric:
                    ax = plt.subplot2grid((len(metric), 2), (i, 0))
                    i += 1
                    ax.plot(x_axis, results['validation_0'][m], label='Train')
                    ax.plot(x_axis, results['validation_1'][m], label='Test')
                    ax.legend()
                    ax.set_ylabel(m)
                # plot feature importances
                features = data.features
                mapFeat = dict(zip(['f' + str(i) for i in range(len(features))], features))
                imp = pd.Series(self.model.booster().get_fscore())
                imp.index = imp.reset_index()['index'].map(mapFeat)
                ax = plt.subplot2grid((len(metric), 2), (0, 1), rowspan=len(metric))
                imp.sort_values().plot(kind='barh')
                ax.set_ylabel('importance')
                plt.show()
            except:
                print('PLOTTING ERROR.')


if __name__ == '__main__':
    print('=' * 63)
    data = stock(input('STOCK ID: '))
    print('-' * 63)
    print('INPUT MODEL PARAMETERS, SKIP FOR DEFAULT:')
    data.load(ratio=input('THRESHOLD RATIO (%): '), start=input('START DATE: '), end=input('END DATE: '), seq_length=input('SEQUENCE LENGTH AS INPUT: '), verbose=False)
    clf = classifier()
    clf.train(data, seed=123, early_stopping_rounds=20)
    print('=' * 63)
