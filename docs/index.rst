sklearn-dummies
===============

Scikit-learn label binarizer with support for missing values.

Usage example
-------------

.. code-block:: python

    import pandas as pd
    import sklearn_dummies as skdm

    df = pd.DataFrame(['A', 'B', None, 'A'], columns=['val'])

    df_dummy = skdm.DataFrameDummies().fit_transform(df)

Result:

=====  =====  =====
idx    val_A  val_B
=====  =====  =====
0      1.0    0.0
1      0.0    1.0
2      NaN    NaN
3      1.0    0.0
=====  =====  =====

Installing
----------

Sklearn-dummies is available in PyPI_. Install via pip:

.. _PyPI: https://pypi.python.org/pypi/sklearn_dummies

.. code-block:: shell

    pip install sklearn_dummies

.. toctree::

    sklearn_dummies.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
