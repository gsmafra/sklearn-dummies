import numpy as np
import pandas as pd

from sklearn_dummies import DataFrameDummies, NPArrayDummies


class TestDF:

    def _test_col(self, df, col, cats):

        df = df[[col]]

        assert len(df[col].value_counts()) == len(cats)

        sk_dummies = DataFrameDummies()
        dfd = sk_dummies.fit_transform(df)

        assert len(dfd.columns) == len(cats)
        assert sk_dummies.cat_cols == [col]
        assert len(df) == len(dfd)

        n_orig_null = pd.isnull(df).sum()[0]
        n_dumm_nulls = [pd.isnull(dfd[[cat]]).sum()[0] for cat in cats]

        for n_dumm_null in n_dumm_nulls:
            assert n_orig_null == n_dumm_null

        orig_null = list(pd.isnull(df[col]))
        cat_nulls = [list(pd.isnull(dfd[cat])) for cat in cats]

        for cat_null in cat_nulls:
            assert orig_null == cat_null

        for i in range(10):

            dfs = df[[col]].sample(1)

            assert type(dfs) is pd.DataFrame
            assert len(dfs) == 1
            assert list(dfs.columns) == [col]

            dfd = sk_dummies.transform(dfs)

            assert len(dfd.columns) == len(cats)
            assert len(dfd) == len(dfs)

    def test_gender(self):

        np.random.seed(0)

        df = pd.read_csv('tests/test_data/data.csv')
        col = 'gender'
        cats = ['gender_M', 'gender_F']

        self._test_col(df, col, cats)

    def test_marital_status(self):

        np.random.seed(0)

        df = pd.read_csv('tests/test_data/data.csv')
        col = 'marital_status'
        cats = ['marital_status_C', 'marital_status_D', 'marital_status_S']

        self._test_col(df, col, cats)


class TestNP:

    def _test_np(self, df, col, cats):

        X = np.asarray(df[col])
        sk_dummies = NPArrayDummies()
        sk_dummies.fit(X)

        assert len(sk_dummies.labels) == len(cats)
        for cat in cats:
            assert cat in sk_dummies.labels

        Xt = sk_dummies.transform(X)

        assert Xt.shape[0] == len(X)
        assert Xt.shape[1] == len(cats)

        assert np.all(Xt[pd.notnull(Xt)] <= 1)
        assert np.all(Xt[pd.notnull(Xt)] >= 0)

        assert len(Xt[0,:]) == len(cats)
        assert len(Xt[:,0]) == len(X)

        orig_null = list(pd.isnull(X))
        tran_null = list(pd.isnull(Xt[:,0]))

        assert len(orig_null) == len(X)
        assert len(tran_null) == len(X)

        assert np.sum(orig_null) == np.sum(tran_null)

        for i in [1, 2]:
            for j in range(10):

                dfs = df[[col]].sample(i)
                X = np.asarray(dfs)

                Xt = sk_dummies.transform(X)

                assert Xt.shape[0] == i
                assert Xt.shape[1] == len(cats)

                assert np.all(Xt[pd.notnull(Xt)] <= 1)
                assert np.all(Xt[pd.notnull(Xt)] >= 0)

                assert len(Xt[:,0]) == i
                assert len(Xt[0,:]) == len(cats)

                orig_null = list(pd.isnull(X))
                tran_null = list(pd.isnull(Xt[:,0]))

                assert len(orig_null) == len(X)
                assert len(tran_null) == len(X)

                assert np.sum(orig_null) == np.sum(tran_null)

    def test_gender(self):

        np.random.seed(0)

        df = pd.read_csv('tests/test_data/data.csv')
        col = 'gender'
        cats = ['M', 'F']

        self._test_np(df, col, cats)


    def test_marital_status(self):

        np.random.seed(0)

        df = pd.read_csv('tests/test_data/data.csv')
        col = 'marital_status'
        cats = ['C', 'D', 'S']

        self._test_np(df, col, cats)
