from typing import Optional, List, Literal
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from scipy.stats import chi2_contingency, kstest


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import (
    SimpleImputer,
    KNNImputer,
)
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
)


#====================================================================#
#                               Classes                              #
#====================================================================#


class DropColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop specified columns
        X_transformed = X.drop(columns=self.columns, axis=1)
        return X_transformed


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy: Literal['mean', 'median', 'most_frequent', 'fill_value'] = 'mean', columns: Optional[List[str]] = None):
        self.strategy = strategy
        self.columns = columns if columns is not None else []
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        self.imputer.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.imputer.transform(X[self.columns])
        return X_transformed


class NaNIndicator(BaseEstimator, TransformerMixin):

    """
    NaNIndicator
    ------------
    This transformer identifies missing values in specified columns and creates new binary columns indicating the presence of missing values.

    Parameters
    ----------
    columns : list of strings
        The list of columns to check for missing values.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, None],
    ...     'B': [4, None, 6]
    ... })
    >>> nan_indicator = NaNIndicator(columns=['A', 'B'])
    >>> transformed_data = nan_indicator.fit_transform(data)
    >>> print(transformed_data)
         A    B  A_missing  B_missing
    0  1.0  4.0          0          0
    1  2.0  NaN          0          1
    2  NaN  6.0          1          0
    """

    def __init__(self, columns, default_suffix: str = '_missing'):
        self.columns = columns
        self.suffix = default_suffix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            new_column_name = column + self.suffix
            X_transformed[new_column_name] = X_transformed[column].isna().astype(int)
        return X_transformed


class CustomWhitespaceRemover(BaseEstimator, TransformerMixin):

    """
    CustomWhitespaceRemover
    -----------------------
    This transformer removes all whitespace characters from the specified columns in a DataFrame.

    Parameters
    ----------
    columns : list of strings
        The list of columns from which to remove whitespace.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'A': ['hello world', '   leading', 'trailing   '],
    ...     'B': ['no spaces', ' some  spaces ', '   whitespace   ']
    ... })
    >>> whitespace_remover = CustomWhitespaceRemover(columns=['A', 'B'])
    >>> transformed_data = whitespace_remover.fit_transform(data)
    >>> print(transformed_data)
                 A              B
    0   helloworld       nospaces
    1      leading     somespaces
    2     trailing     whitespace
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            X_transformed[column] = X_transformed[column].astype(str).apply(lambda x: re.sub(r"\s+", "", x) if pd.notnull(x) else x)
        return X_transformed


class CustomLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for column in self.columns:
            self.encoders[column] = LabelEncoder()
            self.encoders[column].fit(X[column])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            X_transformed[column] = self.encoders[column].transform(X[column])
        return X_transformed


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for column in self.columns:
            self.encoders[column] = OneHotEncoder(sparse_output=False)
            self.encoders[column].fit(X[[column]])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            encoded = pd.DataFrame(
                self.encoders[column].transform(X[[column]]),
                columns=self.encoders[column].get_feature_names_out([column]),
                index=X.index,
            )
            X_transformed = pd.concat(
                [X_transformed.drop(columns=column), encoded], axis=1
            )
        return X_transformed

class CustomCategoryDivider(BaseEstimator, TransformerMixin):
    
    """
    CustomCategoryDivider
    ---------------------
    This transformer splits a categorical column into multiple columns based on a delimiter and assigns each part 
    before the delimiter as a category and the part after as an entry to that category.

    Parameters
    ----------
    column : str
        The name of the column to split.

    Attributes
    ----------
    columns_ : list
        The unique categories extracted from the column after splitting.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'Category_Description': ['Fruit - Apple', 'Vegetable - Carrot', 'Fruit - Banana']
    ... })
    >>> divider = CustomCategoryDivider(column='Category_Description')
    >>> transformed_data = divider.fit_transform(data)
    >>> print(transformed_data)
       Fruit   Vegetable
    0  Apple   NaN
    1  NaN     Carrot
    2  Banana  NaN
    """

    def __init__(self, column: str) -> None:
        self.column = column
        self.columns_ = None

    def fit(self, X, y=None):
        # Use regex to extract the category part
        categories = X[self.column].str.extract(r'^(.+?)\s*-\s*(.*)$', expand=True)[0].unique()
        self.columns_ = categories
        return self

    def transform(self, X):
        # Use regex to extract both parts
        split_columns = X[self.column].str.extract(r'^(.+?)\s*-\s*(.*)$', expand=True)

        unique_categories = self.columns_
        new_df = pd.DataFrame(index=X.index, columns=unique_categories)

        for i, row in split_columns.iterrows():
            category = row[0]
            description = row[1]
            if category in unique_categories:
                new_df.at[i, category] = description

        X_transformed = pd.concat([X, new_df], axis=1)
        
        X_transformed = X_transformed.drop(columns=[self.column])
        
        return X_transformed


class CustomStandardScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.scaler.transform(X[self.columns])
        return X_transformed


class CustomOutlierDetector(BaseEstimator, TransformerMixin):
    def __init__(self, data: pd.DataFrame, IQR_multiplier: float = 1.5):
        self.data = data
        self.IQR_multiplier = IQR_multiplier
    
    def detect_outliers_iqr(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.IQR_multiplier * IQR
        upper_bound = Q3 + self.IQR_multiplier * IQR
        outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
        return outliers


# Custom transformer class to detect and remove outliers
class CustomOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.numeric_cols = None
        self._outliers = None

    # This function identifies the numerical columns
    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(include=np.number).columns
        return self

    def transform(self, X):
        if self.numeric_cols is None:
            raise ValueError("Call 'fit' before 'transform'.")

        # Make a copy of numerical columns
        X_transformed = X.copy()

        z_scores = stats.zscore(X_transformed[self.numeric_cols])

        # Concat with non-numerical columns
        self._outliers = (abs(z_scores) > self.threshold).any(axis=1)
        return X_transformed[~self._outliers]

    @property
    def outliers(self):
        return self._outliers


# Custom transformer for Normalization
class CustomMinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.scaler.transform(X[self.columns])
        return X_transformed

