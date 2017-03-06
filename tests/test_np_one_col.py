import pytest

import numpy as np
import pandas as pd

from sklearn_dummies import NPArrayDummies


@pytest.fixture
def df():
    return pd.read_csv('tests/test_data/data.csv')


class TestNP:

    def _test(self, Xa, Xb, cats):

        sk_dummies = NPArrayDummies()
        sk_dummies.fit(Xa)
        Xt = sk_dummies.transform(Xb)

        # check labels
        assert len(cats) == len(sk_dummies.labels)
        assert all(cats == sk_dummies.labels)

        # check shape of transformed data
        assert Xt.shape[0] == len(Xb)
        assert Xt.shape[1] == len(cats)

        # check range of data
        assert np.all(Xt[pd.notnull(Xt)] <= 1)
        assert np.all(Xt[pd.notnull(Xt)] >= 0)

        orig_null = pd.isnull(Xb)
        tran_null = pd.isnull(Xt[:,0])

        assert len(orig_null) == len(Xb)
        assert len(tran_null) == len(Xb)

        assert np.sum(orig_null) == np.sum(tran_null)

    @pytest.mark.parametrize('col', ['gender', 'marital_status'])
    def test_np_1c(self, df, col):

        """ test numpy array with one column """

        X = np.asarray(df[col])
        cats = np.unique(X[pd.notnull(X)])

        self._test(X, X, cats)

    @pytest.mark.parametrize('n_samples', range(1,10))
    @pytest.mark.parametrize('col', ['gender', 'marital_status'])
    def test_np_1c_subsample(self, df, col, n_samples):

        Xa = np.asarray(df[col])
        cats = np.unique(Xa[pd.notnull(Xa)])

        dfb = df[[col]].sample(n_samples)
        Xb = np.asarray(dfb)

        self._test(Xa, Xb, cats)
