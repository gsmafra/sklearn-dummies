import pytest

import numpy as np
import pandas as pd

from sklearn_dummies import DataFrameDummies


@pytest.fixture
def df():
    return pd.read_csv('tests/test_data/data.csv')


class TestDF:

    def _test(self, df, dfd, col, cats, sk_dummies):

        # checking number of columns and samples
        assert len(dfd.columns) == len(cats)
        assert len(df) == len(dfd)

        # check name of the categorical columns
        assert sk_dummies.cat_cols == [col]

        # number of null values in the origin dataframe
        n_orig_null = pd.isnull(df).sum()[0]

        # number of null values in each of the columns of the dummy dataframe
        n_dumm_nulls = [pd.isnull(dfd[[cat]]).sum()[0] for cat in cats]

        # check if number of null values are the same for every column
        for ni in n_dumm_nulls:
            assert n_orig_null == ni

        # position of null values in the origin dataframe
        orig_null = pd.isnull(df[col])

        # position of null values in each of the columns of the dummy dataframe
        cat_nulls = [pd.isnull(dfd[cat]) for cat in cats]

        assert all((orig_null == cn).all() for cn in cat_nulls)

    @pytest.mark.parametrize('data_type', ['df', 'srs'])
    @pytest.mark.parametrize('col', ['gender', 'marital_status'])
    def test_basic(self, df, col, data_type):

        """ Test dataframe with one column """

        df = df[[col]]
        cats = pd.get_dummies(df).columns

        sk_dummies = DataFrameDummies()

        if data_type == 'df':
            dfd = sk_dummies.fit_transform(df)
        else:
            dfd = sk_dummies.fit_transform(df[col])

        self._test(df, dfd, col, cats, sk_dummies)

    @pytest.mark.parametrize('data_type', ['df', 'srs'])
    @pytest.mark.parametrize('col', ['gender', 'marital_status'])
    @pytest.mark.parametrize('n_samples', range(10))
    def test_subsample(self, df, col, n_samples, data_type):

        """ Transform subsample of the training data """

        df = df[[col]]
        cats = pd.get_dummies(df).columns

        sk_dummies = DataFrameDummies()

        # fit on original data
        if data_type == 'df':
            dfd = sk_dummies.fit_transform(df)
        else:
            dfd = sk_dummies.fit_transform(df[col])

        # get subsample and transform it
        dfs = df[[col]].sample(n_samples)
        dfd = sk_dummies.transform(dfs)

        self._test(dfs, dfd, col, cats, sk_dummies)
