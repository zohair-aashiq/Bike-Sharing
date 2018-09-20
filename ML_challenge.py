#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Any, Union

import pandas as pd
import numpy as np
from time import time
from scipy.stats import randint as sp_randint
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVR
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.model_selection import RandomizedSearchCV
import tkinter
import seaborn as sns


class ML_challenge:

    def __init__(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        ):
        self.x_train = X_train
        self.y_train = y_train
        self.x_test = X_test
        self.y_test = y_test


    def ElasticNet_regressor(self, dataset_predict):
        """
        To train the Ridge regressor with features and target data
        :param dataset_predict: Dataset to fit in the model for predictions
        :return: trained ElasticNet regressor
        """

        # run randomized search

        ElasticNet_trained_model = ElasticNet()
        ElasticNet_trained_model.fit(self.x_train, self.y_train)
        ElasticNet_y = ElasticNet_trained_model.predict(dataset_predict)
        return ElasticNet_y

    @staticmethod
    def multivariant_analysis(df_x):
        '''
        plot correlation matrix on given dataset
        :param: dataset to plot correlation matrix

        '''
        (f, ax) = plt.subplots(figsize=(10, 6))
        corr = df_x.corr()
        hm = sns.heatmap(
            round(corr, 2),
            annot=True,
            ax=ax,
            cmap='coolwarm',
            fmt='.2f',
            linewidths=.05,
            )
        f.subplots_adjust(top=0.93)
        t = f.suptitle('Bike sharing Attributes Correlation Heatmap',
                       fontsize=14)
        plt.show()

    @staticmethod
    def plot_x(split_num,train,test):
        '''
         plot train and test error
        :param split_num: Number of folds
        :param train: Error on training dataset
        :param test : Error on test dataset

        '''
        plt.plot(split_num,train,label='train')
        plt.plot(split_num,test,label='test')
        plt.legend()
        plt.show()
    @staticmethod
    def test_train(
        train_x_pred,
        train_y,
        test_y_pred,
        test_y,
        ):
        """
        print the accuracies of training and test predictions
        :param train_x_pred: predication of training set
        :param test_y_pred: predication of test dataset
        :param test_y_pred : predication of of test dataset
        :param test_y : Actual output of test dataset
        :retrun the mean absolute error and mean squrared error
        """

        mean_squared_error_train = mean_squared_error(train_y,
                train_x_pred)
        mean_squared_error_test = mean_squared_error(test_y,
                test_y_pred)
        mean_absolute_error_train = mean_squared_error(test_y,
                test_y_pred)
        mean_absolute_error_test = mean_absolute_error(test_y,
                test_y_pred)
        error_list = [mean_squared_error_train,
                      mean_squared_error_test,
                      mean_absolute_error_train,
                      mean_absolute_error_test]
        return error_list

    def ML_models(self):
        x_train_ml = self.x_train
        y_train_ml = self.y_train
        x_test_ml = self.x_test
        y_test_ml = self.y_test

        # Apply ElasticNet regression model for prediction on test and train data

        train_y_ElasticNet = self.ElasticNet_regressor(x_train_ml)
        test_y_ElasticNet = self.ElasticNet_regressor(x_test_ml)

        # Calculate the error of appplied models
        Elastic_error_list = \
            ML_challenge.test_train(train_y_ElasticNet, y_train_ml,
                                    test_y_ElasticNet, y_test_ml)
        return  Elastic_error_list


class ReduceVIF(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        thresh=9.5,
        impute=True,
        impute_strategy='median',
        ):

        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.

        self.thresh = thresh

        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.

        if impute:
            self.imputer = Imputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=9.5):
        dropped = True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values,
                   X.columns.get_loc(var)) for var in X.columns]

            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped = True
        return X


def main():

# Main method

    elastic_train_list_squared =  []
    elastic_test_list_squared =  []
    elastic_train_list_absolute =  []
    elastic_test_list_absolute =  []

    elastic_meanError_train = []
    elastic_meanError_test = []
    elastic_meanAbsolute_train = []
    elastic_meanAbsolute_test = []

    SGDR_meanError_train = []
    SGDR_meanError_test = []
    SGDR_meanAbsolute_train = []
    SGDR_meanAbsolute_test = []

    Decision_meanError_train = []
    Decision_meanError_test = []
    Decision_meanAbsolute_train = []
    Decision_meanAbsolute_test = []

    Ridge_meanError_train = []
    Ridge_meanError_test = []
    Ridge_meanAbsolute_train = []
    Ridge_meanAbsolute_test = []
    day_path = 'day.csv'  # File Path of day/train file
    hour_path = 'hour.csv'  # File Path of hours/test file
    day_dataset = pd.read_csv(day_path)  # Open file of train dataset
    hour_dataset = pd.read_csv(hour_path)
    df = pd.concat([day_dataset, hour_dataset])
    df = df.fillna(-1)
    y_dataset = df['cnt']
    dataset = df.drop('cnt', axis=1)
    X_dataset = dataset.drop('dteday', axis=1)

    # Visualization for multivariant analysis

    ML_challenge.multivariant_analysis(X_dataset)

    # Find colinearity and drop the columns which have more than 90% colinearity

    transformer = ReduceVIF()
    X_train_features = \
        transformer.fit_transform(X_dataset[X_dataset.columns[-X_dataset.shape[1]:]],
                                  y_dataset)
    X_train_features.head()

    # split the dataset of trainingset into training and test datasets for training and validation
    split_num = []
    for x in range(2,8):
        rskf = RepeatedStratifiedKFold(n_splits=x, n_repeats=10,
                                       random_state=36851234)
        for (train_index, test_index) in rskf.split(X_train_features,
                y_dataset):
            (X_train, X_test) = (X_train_features.iloc[train_index],
                                 X_train_features.iloc[test_index])
            (y_train, y_test) = (y_dataset.iloc[train_index],
                                 y_dataset.iloc[test_index])
            ml = ML_challenge(X_train, X_test, y_train, y_test)
            (elastic_error, SGDR_error, Ridge_error, Decision_error) = \
                ml.ML_models()

            # Mean error and mean absolute error of train and test datasets of Elastic model

            elastic_meanError_train.append(elastic_error[0])
            elastic_meanError_test.append(elastic_error[1])
            elastic_meanAbsolute_train.append(elastic_error[2])
            elastic_meanAbsolute_test.append(elastic_error[3])

             # Mean error and mean absolute error of train and test datasets of Ridge model

            Ridge_meanError_train.append(SGDR_error[0])
            Ridge_meanError_test.append(SGDR_error[1])
            Ridge_meanAbsolute_train.append(SGDR_error[2])
            Ridge_meanAbsolute_test.append(SGDR_error[3])

             # Mean error and mean absolute error of train and test datasets of Decision model

            Decision_meanError_train.append(Decision_error[0])
            Decision_meanError_test.append(Decision_error[1])
            Decision_meanAbsolute_train.append(Decision_error[2])
            Decision_meanAbsolute_test.append(Decision_error[3])

             # Mean error and mean absolute error of train and test datasets of SGDR model

            SGDR_meanError_train.append(SGDR_error[0])
            SGDR_meanError_test.append(SGDR_error[1])
            SGDR_meanAbsolute_train.append(SGDR_error[2])
            SGDR_meanAbsolute_test.append(SGDR_error[3])

        elastic_train_list_squared.append(np.mean(elastic_meanError_train))
        elastic_test_list_squared.append(np.mean(elastic_meanError_test))
        elastic_train_list_absolute.append(np.mean(elastic_meanAbsolute_train))
        elastic_train_list_absolute.append(np.mean(elastic_meanAbsolute_test))
        split_num.append(x)
    #plot the absolute error and suqared error of train and test dateset
    ML_challenge.plot_x(split_num,elastic_train_list_squared,elastic_test_list_squared)
    print ('#################### ElasticNet Model Errors #################')
    print ('ElasticNet mean error train :',
           np.mean(elastic_meanError_train),
           'ElasticNet mean error test :',
           np.mean(elastic_meanError_test))
    print ('ElasticNet Absolute error train :',
           np.mean(elastic_meanAbsolute_train),
           'ElasticNet Absolute error test :',
           np.mean(elastic_meanAbsolute_test))

    print ('#################### Ridge Model Errors #################')
    print ('Ridge mean error train :', np.mean(Ridge_meanError_train),
           'Ridge mean error test :', np.mean(Ridge_meanError_test))
    print ('Ridge Absolute error train :',
           np.mean(Ridge_meanAbsolute_train),
           'Ridge Absolute error test :',
           np.mean(Ridge_meanAbsolute_test))

    print ('#################### Decision Model Errors #################')
    print ('Decision mean error train :',
           np.mean(Decision_meanError_train),
           'Decision mean error test :',
           np.mean(Decision_meanError_test))
    print ('Decision Absolute error train :',
           np.mean(Decision_meanAbsolute_train),
           'Decision Absolute error test :',
           np.mean(Decision_meanAbsolute_test))

    print('#################### SGDR Model Errors #################')
    print ('SGDR mean error train :', np.mean(SGDR_meanError_train),
           'ElasticNet mean error test :', np.mean(SGDR_meanError_test))
    print ('SGDR Absolute error train:',
           np.mean(SGDR_meanAbsolute_train),
           'ElasticNet Absolute error test :',
           np.mean(SGDR_meanAbsolute_test))


if __name__ == '__main__':
    main()
