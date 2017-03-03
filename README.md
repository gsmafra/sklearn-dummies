# sklearn-dummies
Scikit-learn label binarizer with support for missing values

## Usage example

```
import pandas as pd
import sklearn_dummies as skdm

df = pd.DataFrame(['A', 'B', None, 'A'], columns=['val'])

df_dummy = skdm.DataFrameDummies().fit_transform(df)

```

   | val_A | val_B |
---|:-----:|:-----:|
0  |   1.0 |   0.0 |
1  |   0.0 |   1.0 |
2  |   NaN |   NaN |
3  |   1.0 |   0.0 |

## Installing

Sklearn-dummies is available in [PyPI](https://pypi.python.org/pypi/sklearn_dummies). Installation via pip:

```
pip install sklearn_dummies
```