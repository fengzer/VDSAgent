import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import warnings
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from itertools import combinations
import json
from sklearn.impute import KNNImputer

def fill_missing_values_tools(
    data: pd.DataFrame,
    target_columns: Union[str, List[str]],
    method: str = 'auto',
    group_columns: Optional[Union[str, List[str]]] = None,
    time_column: Optional[str] = None,
    fill_value: Optional[Any] = None,
    max_group_null_ratio: float = 0.8,
    **params
) -> pd.DataFrame:
    """
    General missing value filling function
    
    Args:
        data: Input DataFrame
        target_columns: Target columns to fill
        method: Filling method
            - 'auto': Automatic selection (mean for numeric, mode for categorical)
            - 'mean','median','mode': Statistical value filling
            - 'ffill','bfill': Forward/backward filling
            - 'interpolate': Interpolation filling
            - 'constant': Constant value filling
            - 'knn': KNN-based filling
        group_columns: Group columns for grouped filling
        time_column: Time column for time-series related filling
        fill_value: Fill value for constant method
        max_group_null_ratio: Maximum allowed null ratio in groups
        **params: Additional parameters
        
    Returns:
        DataFrame with filled values
    """
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            raise ValueError("Invalid params string format. Must be valid JSON.")

    df = data.copy()
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    if isinstance(group_columns, str):
        group_columns = [group_columns]
        
    for target_col in target_columns:
        # Check if column exists
        if target_col not in df.columns:
            raise ValueError(f"Column {target_col} not found in data")
            
        # Skip if no missing values
        if not df[target_col].isna().any():
            continue
            
        # Perform initial type conversion
        if pd.api.types.is_numeric_dtype(df[target_col]):
            df[target_col] = df[target_col].astype(float)
            
        # Determine filling method
        if method == 'auto':
            if pd.api.types.is_numeric_dtype(df[target_col]):
                # For numeric columns, calculate fill value first
                fill_val = df[target_col].mean()
                # Ensure column is float type
                df[target_col] = df[target_col].astype(float)
                df[target_col].fillna(fill_val, inplace=True)
            else:
                # For non-numeric columns, use mode to fill
                fill_val = df[target_col].mode()[0] if not df[target_col].mode().empty else None
                if fill_val is not None:
                    df[target_col] = df[target_col].fillna(fill_val)
            continue
            
        # If constant filling but no fill value provided, set default
        if method == 'constant' and fill_value is None:
            if pd.api.types.is_numeric_dtype(df[target_col]):
                fill_value = 0
            else:
                fill_value = 'Unknown'
            warnings.warn(f"No fill_value provided for constant method in column {target_col}. "
                        f"Using default value: {fill_value}")
        
        # If using grouped filling, optimize filling logic
        if group_columns:
            # Calculate grouped statistics
            if method in ['mean', 'median', 'mode']:
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    if method == 'mean':
                        fill_values = df.groupby(group_columns)[target_col].transform('mean')
                    elif method == 'median':
                        fill_values = df.groupby(group_columns)[target_col].transform('median')
                    else:
                        fill_values = df.groupby(group_columns)[target_col].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
                else:
                    fill_values = df.groupby(group_columns)[target_col].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
                
                # Fill all valid values at once
                mask = df[target_col].isna()
                group_null_ratios = df[mask].groupby(group_columns).size() / df.groupby(group_columns).size()
                valid_groups = group_null_ratios[group_null_ratios <= max_group_null_ratio].index
                
                if len(valid_groups) > 0:
                    valid_mask = mask & df[group_columns].isin(valid_groups).all(axis=1)
                    df.loc[valid_mask, target_col] = fill_values[valid_mask]
            
            elif method in ['ffill', 'bfill']:
                # Use transform for filling
                if time_column:
                    df[target_col] = df.sort_values([time_column]).groupby(group_columns)[target_col].transform(lambda x: x.fillna(method=method))
                else:
                    df[target_col] = df.groupby(group_columns)[target_col].transform(lambda x: x.fillna(method=method))
                    
            elif method == 'interpolate':
                # Use transform for interpolation
                df[target_col] = df.groupby(group_columns)[target_col].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )
            
        # Global filling (when no groups are specified)
        else:
            if method == 'constant':
                if pd.api.types.is_numeric_dtype(df[target_col]) and isinstance(fill_value, (int, float)):
                    df[target_col] = df[target_col].astype(float)
                df[target_col].fillna(fill_value, inplace=True)
            elif method in ['mean', 'median', 'mode']:
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    # Ensure numeric columns are float type
                    df[target_col] = df[target_col].astype(float)
                    if method == 'mean':
                        fill_val = df[target_col].mean()
                    elif method == 'median':
                        fill_val = df[target_col].median()
                    else:
                        fill_val = df[target_col].mode()[0]
                else:
                    fill_val = df[target_col].mode()[0] if not df[target_col].mode().empty else None
                
                if fill_val is not None:
                    df[target_col].fillna(fill_val, inplace=True)
            elif method in ['ffill', 'bfill']:
                df[target_col].fillna(method=method, inplace=True)
            elif method == 'interpolate':
                df[target_col].interpolate(method='linear', limit_direction='both', inplace=True)       

        # Add knn method handling in the method decision part
        if method == 'knn':
            # Get KNN parameters
            n_neighbors = params.get('n_neighbors', 5)
            weights = params.get('weights', 'uniform')  # 'uniform' or 'distance'
            
            # Prepare features for KNN filling
            if group_columns:
                # If group columns exist, use them as additional features
                features = group_columns.copy() if isinstance(group_columns, list) else [group_columns]
            else:
                # If no group columns, use all numeric columns as features
                features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target column (if it's in features)
            if target_col in features:
                features.remove(target_col)
            
            if not features:
                raise ValueError("No features available for KNN imputation")
            
            # Prepare feature matrix
            X = df[features].copy()
            
            # Encode categorical features
            categorical_features = X.select_dtypes(include=['object', 'category']).columns
            for cat_col in categorical_features:
                X[cat_col] = pd.Categorical(X[cat_col]).codes
            
            # Standardize numeric features
            numeric_features = X.select_dtypes(include=[np.number]).columns
            if len(numeric_features) > 0:
                scaler = StandardScaler()
                X[numeric_features] = scaler.fit_transform(X[numeric_features])
            
            # Initialize KNN imputer
            imputer = KNNImputer(
                n_neighbors=n_neighbors,
                weights=weights,
                copy=True
            )
            
            # Prepare data to impute
            data_to_impute = df[[target_col]].copy()
            if pd.api.types.is_numeric_dtype(data_to_impute[target_col]):
                # For numeric columns, perform KNN imputation directly
                data_to_impute = pd.DataFrame(
                    imputer.fit_transform(pd.concat([data_to_impute, X], axis=1))[:, 0],
                    index=df.index,
                    columns=[target_col]
                )
            else:
                # For categorical columns, convert to numeric first
                le = LabelEncoder()
                non_missing_mask = ~data_to_impute[target_col].isna()
                temp_values = data_to_impute[target_col].copy()
                temp_values[non_missing_mask] = le.fit_transform(temp_values[non_missing_mask])
                temp_values = temp_values.astype(float)
                
                # Perform KNN imputation
                temp_values = pd.DataFrame(
                    imputer.fit_transform(pd.concat([pd.DataFrame(temp_values), X], axis=1))[:, 0],
                    index=df.index,
                    columns=[target_col]
                )
                
                # Convert filled numeric values back to categorical
                temp_values[target_col] = le.inverse_transform(temp_values[target_col].round().astype(int))
                data_to_impute = temp_values
            
            # Update values in the original DataFrame
            df.loc[df[target_col].isna(), target_col] = data_to_impute.loc[df[target_col].isna(), target_col]
        
    return df

def remove_columns_tools(
    data: pd.DataFrame,
    strategy: Union[str, List[str], None] = None,
    columns: Optional[List[str]] = None,
    threshold: Union[float, Dict[str, float]] = 0.5,
    exclude_columns: Optional[List[str]] = None,
    min_unique_ratio: float = 0.01,
    correlation_threshold: float = 0.95
) -> pd.DataFrame:
    """
    General column deletion function, supporting multiple deletion strategies
    
    Args:
        data: Input DataFrame
        strategy: Deletion strategy, can be a single strategy or a list of strategies
            - 'missing': Delete based on missing value ratio
            - 'constant': Delete based on unique value ratio
            - 'correlation': Delete based on correlation
            - 'variance': Delete based on variance
            - None: Only use columns specified by the columns parameter to delete
        columns: List of column names to delete directly
        threshold: Thresholds for each strategy
            - missing: Maximum allowed missing ratio
            - constant: Maximum allowed constant value ratio
            - correlation: Maximum allowed correlation coefficient
            - variance: Minimum allowed variance
        exclude_columns: Columns not to check
        min_unique_ratio: Minimum unique value ratio
        correlation_threshold: Correlation threshold
        
    Returns:
        Processed DataFrame
    """
    df = data.copy()
    if isinstance(strategy, str):
        strategy = [strategy]
    if exclude_columns is None:
        exclude_columns = []
        
    columns_to_drop = set()
    
    # Add columns directly specified to delete
    if columns:
        columns_to_drop.update([col for col in columns if col in df.columns])
    
    # Only execute strategy-based deletion if strategy is not None
    if strategy is not None:
        for strat in strategy:
            if strat == 'missing':
                # Delete columns with too many missing values
                missing_ratio = df.isnull().mean()
                cols = missing_ratio[
                    (missing_ratio > threshold) & 
                    (~missing_ratio.index.isin(exclude_columns))
                ].index
                columns_to_drop.update(cols)
                
            elif strat == 'constant':
                # Delete constant columns
                for col in df.columns:
                    if col in exclude_columns:
                        continue
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < min_unique_ratio:
                        columns_to_drop.add(col)
                        
            elif strat == 'correlation':
                # Delete highly correlated columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                corr_matrix = df[numeric_cols].corr().abs()
                
                # Get columns to delete with high correlation
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
                
                # Exclude specified columns
                to_drop = [col for col in to_drop if col not in exclude_columns]
                columns_to_drop.update(to_drop)
                
            elif strat == 'variance':
                # Delete low variance columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col in exclude_columns:
                        continue
                    if df[col].var() < threshold:
                        columns_to_drop.add(col)
                        
    if columns_to_drop:
        #warnings.warn(f"Removing columns: {list(columns_to_drop)}")
        df = df.drop(columns=list(columns_to_drop))
        
    return df

def handle_outliers_tools(
    data: pd.DataFrame,
    target_columns: Union[str, List[str]],
    method: str = 'iqr',
    strategy: str = 'clip',
    sensitivity: str = 'medium',
    group_columns: Optional[Union[str, List[str]]] = None,
    params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    General outlier handling function
    
    Args:
        data: Input DataFrame
        target_columns: Target columns
        method: Outlier detection method
            - 'iqr': IQR method
            - 'zscore': Z-score method
            - 'isolation_forest': Isolation Forest
            - 'dbscan': DBSCAN clustering
            - 'mad': MAD method
        strategy: Processing strategy 'clip' or 'remove'
        sensitivity: Sensitivity of outlier detection
            - 'low': Loose threshold, only detect extreme outliers
            - 'medium': Medium threshold (default)
            - 'high': Strict threshold, detect more outliers
        group_columns: Group columns
        params: Dictionary of parameters for each method
    """
    # Define parameter configurations for different sensitivities
    sensitivity_params = {
        'low': {
            'iqr': {'threshold': 3.0},
            'zscore': {'threshold': 4.0},
            'isolation_forest': {'contamination': 0.05},
            'dbscan': {'eps': 0.8, 'min_samples': 3},
            'mad': {'threshold': 5.0}
        },
        'medium': {
            'iqr': {'threshold': 1.5},
            'zscore': {'threshold': 3.0},
            'isolation_forest': {'contamination': 0.1},
            'dbscan': {'eps': 0.5, 'min_samples': 5},
            'mad': {'threshold': 3.5}
        },
        'high': {
            'iqr': {'threshold': 1.0},
            'zscore': {'threshold': 2.0},
            'isolation_forest': {'contamination': 0.15},
            'dbscan': {'eps': 0.3, 'min_samples': 7},
            'mad': {'threshold': 2.5}
        }
    }

    # Validate sensitivity parameter
    if sensitivity not in sensitivity_params:
        raise ValueError(
            f"Invalid sensitivity: '{sensitivity}'. "
            f"Please use one of: {list(sensitivity_params.keys())}"
        )

    # Get default parameters for corresponding sensitivity
    default_params = sensitivity_params[sensitivity][method]
    
    # Merge user-defined parameters and default parameters
    params = params or {}
    params = {**default_params, **params}  # User parameters have higher priority

    # If method is 'clip', automatically correct to 'iqr'
    if method == 'clip':
        warnings.warn("'clip' was passed as method but it's a strategy. Using default method 'iqr' instead.")
        method = 'iqr'

    # Validate method parameter
    valid_methods = {
        'iqr': 'IQR method - Identifies outliers using interquartile range',
        'zscore': 'Z-score method - Identifies outliers based on standard deviation',
        'isolation_forest': 'Isolation Forest - Identifies outliers using an isolation tree algorithm',
        'dbscan': 'DBSCAN clustering - Identifies outliers based on density clustering',
        'mad': 'MAD method - Identifies outliers based on MAD'
    }
    
    valid_strategies = {
        'clip': 'Limits outliers to a reasonable range',
        'remove': 'Deletes rows containing outliers'
    }
    
    if method not in valid_methods:
        methods_description = "\n".join([f"- '{k}': {v}" for k, v in valid_methods.items()])
        error_msg = f"Invalid method: '{method}'\n"
        
        # Check if strategy was mistakenly written as method
        if method in valid_strategies:
            error_msg += (
                f"\nNOTE: It seems you might have confused 'method' with 'strategy'.\n"
                f"'{method}' is a valid strategy, not a method.\n"
                f"Did you mean to write:\n"
                f"    handle_outliers_tools(data, target_columns, method='iqr', strategy='{method}')\n\n"
            )
        
        error_msg += (
            f"Please use one of the following methods:\n{methods_description}\n"
            f"Example usage: handle_outliers_tools(data, 'column1', method='iqr', strategy='clip')"
        )
        raise ValueError(error_msg)
    
    # Validate strategy parameter
    if strategy not in valid_strategies:
        strategies_description = "\n".join([f"- '{k}': {v}" for k, v in valid_strategies.items()])
        error_msg = f"Invalid strategy: '{strategy}'\n"
        
        # Check if method was mistakenly written as strategy
        if strategy in valid_methods:
            error_msg += (
                f"\nNOTE: It seems you might have confused 'strategy' with 'method'.\n"
                f"'{strategy}' is a valid method, not a strategy.\n"
                f"Did you mean to write:\n"
                f"    handle_outliers_tools(data, target_columns, method='{strategy}', strategy='clip')\n\n"
            )
            
        error_msg += (
            f"Please use one of the following strategies:\n{strategies_description}\n"
            f"Example usage: handle_outliers_tools(data, 'column1', method='iqr', strategy='clip')"
        )
        raise ValueError(error_msg)

    df = data.copy()
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    if isinstance(group_columns, str):
        group_columns = [group_columns]
        
    for target_col in target_columns:
        if pd.api.types.is_numeric_dtype(df[target_col]):
            df[target_col] = df[target_col].astype('float64')
        else:
            warnings.warn(f"Column {target_col} is not numeric, skipping")
            continue
            
        # Group processing
        if group_columns:
            if method == 'iqr':
                # Use transform to calculate statistics for all groups at once
                Q1 = df.groupby(group_columns)[target_col].transform('quantile', 0.25)
                Q3 = df.groupby(group_columns)[target_col].transform('quantile', 0.75)
                IQR = Q3 - Q1
                threshold = params.get('threshold', 1.5)
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                
                if strategy == 'clip':
                    df[target_col] = df[target_col].clip(lower=lower, upper=upper)
                else:  # remove
                    outlier_mask = (df[target_col] < lower) | (df[target_col] > upper)
                    df = df.loc[~outlier_mask]
                    
            else:
                # For other methods, use transform to calculate outlier masks for each group
                def detect_group_outliers(x):
                    return detect_outliers_tools(x, method=method, params=params)
                
                outlier_mask = df.groupby(group_columns)[target_col].transform(detect_group_outliers)
                
                if strategy == 'clip':
                    # Calculate mean of normal values for each group
                    normal_values = df[~outlier_mask].groupby(group_columns)[target_col].transform('mean')
                    # Replace all outliers at once
                    df.loc[outlier_mask, target_col] = normal_values[outlier_mask]
                else:  # remove
                    df = df.loc[~outlier_mask]
        
        # Global processing
        else:
            if method == 'iqr':
                Q1 = df[target_col].quantile(0.25)
                Q3 = df[target_col].quantile(0.75)
                IQR = Q3 - Q1
                threshold = params.get('threshold', 1.5)
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                
                if strategy == 'clip':
                    # Use pandas' clip method here as well
                    df[target_col] = df[target_col].clip(lower=lower, upper=upper)
                else:  # remove
                    outlier_mask = (df[target_col] < lower) | (df[target_col] > upper)
                    df = df.loc[~outlier_mask]
            else:
                outlier_mask = detect_outliers_tools(df[target_col], method=method, params=params)
                
                if strategy == 'clip':
                    normal_values = df[target_col][~outlier_mask].mean()
                    df.loc[outlier_mask, target_col] = normal_values
                else:  # remove
                    df = df.drop(index=df[outlier_mask].index)
                
    return df

def detect_outliers_tools(
    series: pd.Series,
    method: str = 'iqr',
    params: Optional[Dict[str, Any]] = None
) -> pd.Series:
    """
    Helper function to detect outliers
    
    Args:
        series: Input data column
        method: Detection method
        params: Method parameters
        
    Returns:
        Boolean series, True for outliers
    """
    params = params or {}
    
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        threshold = params.get('threshold', 1.5)
        return (series < (Q1 - threshold * IQR)) | (series > (Q3 + threshold * IQR))
        
    elif method == 'zscore':
        threshold = params.get('threshold', 3)
        z_scores = np.abs(stats.zscore(series))
        return z_scores > threshold
        
    elif method == 'mad':
        threshold = params.get('threshold', 3.5)
        median = series.median()
        mad = stats.median_abs_deviation(series)
        modified_zscore = 0.6745 * (series - median) / mad
        return np.abs(modified_zscore) > threshold
        
    elif method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(
            contamination=params.get('contamination', 0.1),
            random_state=params.get('random_state', 42)
        )
        return iso.fit_predict(series.values.reshape(-1, 1)) == -1
        
    elif method == 'dbscan':
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(
            eps=params.get('eps', 0.5),
            min_samples=params.get('min_samples', 5)
        )
        return dbscan.fit_predict(series.values.reshape(-1, 1)) == -1
        
    else:
        raise ValueError(f"Unknown method: {method}")

def encode_categorical_tools(data: pd.DataFrame, 
                      target_columns: Union[str, List[str]], 
                      method: str = 'auto',
                      group_columns: Optional[Union[str, List[str]]] = None,
                      handle_unknown: str = 'ignore',
                      keep_original: bool = True) -> pd.DataFrame:
    """
    General categorical feature encoding function
    
    Args:
        data: Input DataFrame
        target_columns: Target columns to encode
        method: Encoding method
            - 'auto': Automatic selection (default uses one-hot encoding)
            - 'label': Label encoding
            - 'onehot': One-hot encoding
            - 'frequency': Frequency encoding
            - 'count': Count encoding
        group_columns: Group columns (for grouped encoding)
        handle_unknown: How to handle unknown categories
        keep_original: Whether to keep original categorical columns, default is True
        
    Returns:
        DataFrame with encoded features
    """
    df = data.copy()
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    
    # Filter out columns that truly need encoding
    columns_to_encode = []
    for col in target_columns:
        # Skip non-existent columns
        if col not in df.columns:
            warnings.warn(f"Column {col} not found in dataframe")
            continue
            
        # Check column type
        if df[col].dtype == 'object':
            # Object type directly added to encoding list
            columns_to_encode.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Numeric type needs to check if it's a categorical variable
            unique_count = df[col].nunique()
            if unique_count < len(df[col]) * 0.05:  # If the number of different values is less than 5%, it's considered a categorical variable
                columns_to_encode.append(col)
                df[col] = df[col].astype(str)  # Convert to string
    
    for target_col in columns_to_encode:
        try:
            if method == 'auto' or method == 'onehot':
                dummies = pd.get_dummies(df[target_col], prefix=target_col, drop_first=False)
                df = pd.concat([df, dummies], axis=1)
                if not keep_original:
                    df.drop(columns=[target_col], inplace=True)
                    
            elif method == 'label':
                encoder = LabelEncoder()
                df[f"{target_col}_encoded"] = encoder.fit_transform(df[target_col].astype(str))
                if not keep_original:
                    df.drop(columns=[target_col], inplace=True)
                    
            elif method == 'frequency':
                freq = df[target_col].value_counts(normalize=True)
                df[f"{target_col}_freq"] = df[target_col].map(freq)
                if not keep_original:
                    df.drop(columns=[target_col], inplace=True)
                    
            elif method == 'count':
                if group_columns:
                    df[f"{target_col}_count"] = df.groupby(group_columns)[target_col].transform('count')
                else:
                    df[f"{target_col}_count"] = df[target_col].map(df[target_col].value_counts())
                if not keep_original:
                    df.drop(columns=[target_col], inplace=True)
                    
        except Exception as e:
            warnings.warn(f"Error encoding column {target_col}: {str(e)}")
            continue
                
    return df


###########################################
#FEATURE ENGINEERING
###########################################

def transform_features(data: pd.DataFrame,
                      columns: Union[str, List[str]],
                      method: str = 'standard',
                      params: Optional[dict] = None,
                      keep_original: bool = True,
                      scaler: Optional[object] = None) -> Tuple[pd.DataFrame, object]:
    """
    Transforms and scales features using various methods.
    
    Parameters:
        data (pd.DataFrame): Input DataFrame
        columns (Union[str, List[str]]): Column names to transform or a list of column names
        method (str): Transformation method, options:
            - 'standard': Standardization (mean=0, variance=1)
            - 'minmax': Min-max normalization (scaled to [0,1])
            - 'robust': Robust scaling (uses quartiles, less sensitive to outliers)
            - 'log': Natural logarithm transformation
            - 'sqrt': Square root transformation
            - 'power': Power transformation (Yeo-Johnson transformation)
        params (dict, optional): Additional parameters for the transformer
        keep_original (bool): Whether to keep original columns, default is True
        scaler: Pre-trained scaler object. If None, a new scaler is created.
    
    Returns:
        Tuple[pd.DataFrame, object]: (Transformed DataFrame, used scaler)
    """
    # Unified input format and validation
    columns = [columns] if isinstance(columns, str) else columns
    if missing := set(columns) - set(data.columns):
        raise ValueError(f"Columns not found: {missing}")
    if non_numeric := [col for col in columns if not pd.api.types.is_numeric_dtype(data[col])]:
        raise ValueError(f"Non-numeric columns: {non_numeric}")

    # Define transformation methods dictionary
    transform_methods = {
        'log': lambda: {
            'transform': lambda x: np.sign(x) * np.log1p(np.abs(x)),
            'method': 'log'
        },
        'sqrt': lambda: {
            'transform': lambda x: np.sign(x) * np.sqrt(np.abs(x)),
            'method': 'sqrt'
        }
    }

    # Handle parameter-less transformation methods
    if method in transform_methods:
        scaler = transform_methods[method]() if scaler is None else scaler
        transformed_data = scaler['transform'](data[columns].values)
        if method == 'log':  # Handle log-specific infinite values
            transformed_data = np.where(np.isinf(transformed_data),
                                      np.sign(transformed_data) * np.log1p(np.finfo(float).max),
                                      transformed_data)
    
    # Handle standardization methods
    elif method == 'standard':
        if scaler is None:
            standard_scaler = StandardScaler()
            robust_scaler = RobustScaler()
            transformed_data = standard_scaler.fit_transform(data[columns])
            
            # Check RobustScaler results
            if np.any(~np.isfinite(transformed_data)):
                warnings.warn("StandardScaler produced infinite or NaN values, switching to robust scaling")
                transformed_data = robust_scaler.fit_transform(data[columns])
                # 检查RobustScaler的结果
                if np.any(~np.isfinite(transformed_data)):
                    warnings.warn("RobustScaler also produced outliers, using 0 to fill")
                    transformed_data = np.where(~np.isfinite(transformed_data), 0, transformed_data)
                scaler = robust_scaler
            elif np.any(extreme_columns := np.any(np.abs(transformed_data) > 10, axis=0)):
                extreme_cols = [col for col, is_extreme in zip(columns, extreme_columns) if is_extreme]
                warnings.warn(f"The following columns contain extreme values, robust scaling is applied to these columns: {extreme_cols}")
                robust_transformed = robust_scaler.fit_transform(data[extreme_cols])
                # 检查RobustScaler的结果
                if np.any(~np.isfinite(robust_transformed)):
                    warnings.warn("RobustScaler produced outliers, using 0 to fill")
                    robust_transformed = np.where(~np.isfinite(robust_transformed), 0, robust_transformed)
                transformed_data[:, extreme_columns] = robust_transformed
                scaler = {'standard': standard_scaler, 'robust': robust_scaler,
                         'extreme_columns': extreme_columns, 'extreme_cols': extreme_cols}
            else:
                scaler = standard_scaler
        else:
            transformed_data = (scaler['standard'].transform(data[columns]) 
                              if isinstance(scaler, dict) 
                              else scaler.transform(data[columns]))
            if isinstance(scaler, dict) and np.any(scaler['extreme_columns']):
                transformed_data[:, scaler['extreme_columns']] = scaler['robust'].transform(
                    data[scaler['extreme_cols']]
                )

    # Handle other scaling methods
    elif method in ['minmax', 'robust', 'power']:
        if scaler is None:
            scaler = (PowerTransformer(method='yeo-johnson') if method == 'power' 
                     else MinMaxScaler(**(params or {})) if method == 'minmax'
                     else RobustScaler(**(params or {})))
            transformed_data = scaler.fit_transform(data[columns])
            
            # Handle special cases
            if np.any(~np.isfinite(transformed_data)):
                if method == 'power':
                    warnings.warn("PowerTransformer produced outliers, attempting robust scaling")
                    robust_scaler = RobustScaler()
                    mask = ~np.isfinite(transformed_data)
                    robust_transformed = robust_scaler.fit_transform(data[columns].values)
                    
                    # 检查RobustScaler的结果
                    if np.any(~np.isfinite(robust_transformed)):
                        warnings.warn("RobustScaler also produced outliers, using 0 to fill")
                        robust_transformed = np.where(~np.isfinite(robust_transformed), 0, robust_transformed)
                    
                    transformed_data[mask] = robust_transformed[mask]
                    scaler = {'power': scaler, 'robust': robust_scaler, 'mask': mask}
                else:
                    warnings.warn(f"{method.capitalize()}Scaler produced outliers, using 0 to fill")
                    transformed_data = np.where(~np.isfinite(transformed_data), 0, transformed_data)
        else:
            if isinstance(scaler, dict) and method == 'power':
                transformed_data = scaler['power'].transform(data[columns])
                if np.any(scaler['mask']):
                    transformed_data[scaler['mask']] = scaler['robust'].transform(
                        data[columns].values
                    )[scaler['mask']]
            else:
                transformed_data = scaler.transform(data[columns])
    else:
        raise ValueError(f"Unknown transform method: {method}")

    # Create result DataFrame
    transformed_df = pd.DataFrame(transformed_data, 
                                columns=[f"{col}_{method}" for col in columns],
                                index=data.index)
    
    # Merge results
    result = pd.concat(
        [data.copy(), transformed_df] if keep_original 
        else [data.drop(columns=columns), transformed_df], 
        axis=1
    )
    
    return result, scaler

def reduce_dimensions(data: pd.DataFrame,
                     method: str = 'pca',
                     n_components: Union[int, float] = 0.95,
                     target: Optional[pd.Series] = None,
                     keep_original: bool = True) -> pd.DataFrame:
    """
    Reduces dimensions using PCA or LDA.
    
    Parameters:
        data (pd.DataFrame): Input feature matrix
        method (str): Dimensionality reduction method ('pca' or 'lda')
        n_components (Union[int, float]): Number of components to retain
            - If integer: Use the specified number of components
            - If float: Retain components explaining this proportion of variance
            Note: For LDA, the number of components cannot exceed min(number of features, number of classes-1)
        target (pd.Series, optional): Target variable, required for LDA method
        keep_original (bool): Whether to keep original features, default is True
    
    Returns:
        pd.DataFrame: Reduced DataFrame, if keep_original=True, original features are included
        
    Exceptions:
        ValueError: When an invalid method or parameter is specified
    """
    if method not in ['pca', 'lda']:
        raise ValueError("Method must be 'pca' or 'lda'")
        
    if method == 'lda':
        if target is None:
            raise ValueError("Target required for LDA")
        # Calculate maximum possible number of components for LDA
        n_classes = len(np.unique(target))
        max_components = min(data.shape[1], n_classes - 1)
        if isinstance(n_components, int) and n_components > max_components:
            n_components = max_components
            warnings.warn(f"n_components reduced to {max_components} for LDA")
    
    # Validate numeric features
    non_numeric = data.select_dtypes(exclude=['number']).columns
    if len(non_numeric):
        raise ValueError(f"Non-numeric columns found: {non_numeric}")
        
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        transformed = reducer.fit_transform(data)
        
        # Create column names
        cols = [f'PC{i+1}' for i in range(transformed.shape[1])]
        
    else:  # LDA
        reducer = LinearDiscriminantAnalysis(n_components=n_components)
        transformed = reducer.fit_transform(data, target)
        
        # Create column names  
        cols = [f'LD{i+1}' for i in range(transformed.shape[1])]
    
    # Create reduced dimension DataFrame
    transformed_df = pd.DataFrame(transformed, columns=cols, index=data.index)
    
    # If original features need to be kept, merge original features and reduced features
    if keep_original:
        return pd.concat([data, transformed_df], axis=1)
    else:
        return transformed_df

from multiprocessing import cpu_count
from functools import partial
from sklearn.linear_model import LassoCV

def select_features(data: pd.DataFrame,
                   target: Optional[pd.Series] = None,
                   method: str = 'variance',
                   n_features: Optional[int] = None,
                   params: Optional[dict] = None) -> List[str]:
    """
    Performs feature selection using various methods.
    
    Parameters:
        data (pd.DataFrame): Input feature matrix
        target (pd.Series, optional): Target variable, required for some methods
        method (str): Feature selection method:
            - 'variance': Remove low variance features
            - 'correlation': Select based on correlation with target variable
            - 'mutual_info': Select based on mutual information
            - 'rfe': Recursive feature elimination
            - 'lasso': Feature selection based on L1 regularization
        n_features (int, optional): Number of features to select
            - If None, then:
                variance: Use threshold selection
                correlation: Use threshold selection
                mutual_info: Default to 10
                rfe: Default to half of features
                lasso: Use alpha parameter to control
        params (dict, optional): Additional parameters for the selector:
            - variance: {'threshold': float}
            - correlation: {'threshold': float}
            - mutual_info: {'k': int}
            - rfe: {'step': int}
            - lasso: {'alpha': float}
    
    Returns:
        List[str]: List of selected feature names
        
    Exceptions:
        ValueError: When an invalid method or missing required parameters is specified
    """
    params = params or {}
    n_jobs = params.get('n_jobs', -1)
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    df = data.copy()
    
    # Handle target variable
    if target is not None:
        # Save original target variable
        original_target = target.copy()
        
        # If target is discrete, encode it
        if not pd.api.types.is_numeric_dtype(target):
            encoder = LabelEncoder()
            target = pd.Series(
                encoder.fit_transform(target),
                index=target.index,
                name=target.name
            )
        
        # If target is in features, remove it
        if target.name in df.columns:
            df = df.drop(columns=[target.name])
    
    # Numeric type conversion
    numeric_cols = df.select_dtypes(include=np.number).columns
    #df[numeric_cols] = df[numeric_cols].astype('float32')
    
    if target is not None:
        target = target.astype('float32')
    
    if method in ['correlation', 'mutual_info', 'rfe', 'lasso'] and target is None:
        raise ValueError(f"Target required for method '{method}'")
    
    if n_features is not None and n_features <= 0:
        raise ValueError("n_features must be positive")
    
    if n_features is not None:
        if n_features > df.shape[1]:
            warnings.warn(f'n_features ({n_features}) adjusted to match maximum available features ({df.shape[1]})')
            n_features = df.shape[1]
        
    if method == 'variance':
        if n_features is None:
            threshold = params.get('threshold', 0.0)
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(df)
            mask = selector.get_support()
        else:
            # Use numpy operations to avoid pandas type inference
            variances = np.var(df.values, axis=0)
            idx = np.argsort(variances)[-n_features:]
            mask = df.columns[idx]
            
    elif method == 'correlation':
        if not pd.api.types.is_numeric_dtype(target):
            raise ValueError("Target must be numeric for correlation method")
            
        correlations = np.zeros(df.shape[1])
        for i in range(df.shape[1]):
            try:
                mask = ~np.isnan(df.iloc[:, i]) & ~np.isnan(target)
                if np.sum(mask) > 1:
                    correlations[i] = np.abs(stats.spearmanr(
                        df.iloc[mask, i],
                        target[mask]
                    )[0])
            except:
                correlations[i] = 0
                
        # Feature selection
        if n_features is None:
            threshold = params.get('threshold', 0.1)
            mask = df.columns[correlations > threshold]
        else:
            idx = np.argsort(-np.abs(correlations))[:n_features]
            mask = df.columns[idx]
            
    elif method == 'mutual_info':
        k = n_features if n_features is not None else params.get('k', 10)
        
        if pd.api.types.is_numeric_dtype(target):
            mi_func = mutual_info_regression
        else:
            mi_func = mutual_info_classif
            
        selector = SelectKBest(
            partial(mi_func, n_jobs=n_jobs),
            k=k
        )
        selector.fit(df, target)
        mask = selector.get_support()
        
    elif method == 'rfe':
        n_features_to_select = n_features if n_features is not None else df.shape[1] // 2
        step = params.get('step', 1)
        
        if pd.api.types.is_numeric_dtype(target):
            estimator = RandomForestRegressor(
                n_estimators=100,
                n_jobs=n_jobs
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=100,
                n_jobs=n_jobs
            )
            
        selector = RFE(
            estimator, 
            n_features_to_select=n_features_to_select, 
            step=step,
            n_jobs=n_jobs
        )
        selector.fit(df, target)
        mask = selector.support_
        
    elif method == 'lasso':
        if n_features is None:
            alpha = params.get('alpha', 1.0)
            if pd.api.types.is_numeric_dtype(target):
                selector = LassoCV(
                    cv=5,
                    n_jobs=n_jobs,
                    random_state=42
                )
            else:
                selector = LogisticRegression(
                    penalty='l1',
                    solver='saga',
                    C=1/alpha,
                    n_jobs=n_jobs
                )
            
            selector.fit(df, target)
            mask = np.abs(selector.coef_) > 1e-10
        else:
            if pd.api.types.is_numeric_dtype(target):
                selector = Lasso(alpha=0.01)
            else:
                selector = LogisticRegression(penalty='l1', solver='liblinear', C=100)
            
            selector.fit(df, target)
            coef_abs = np.abs(selector.coef_)
            idx = np.argsort(coef_abs)[-n_features:]
            mask = df.columns[idx]
    else:
        raise ValueError(f"Unknown selection method: {method}")
        
    if isinstance(mask, pd.Index):
        selected_columns = mask.tolist()
    else:
        selected_columns = df.columns[mask].tolist()

    return selected_columns

def create_polynomial_features(data: pd.DataFrame,
                             columns: Union[str, List[str]],
                             degree: int = 2,
                             interaction_only: bool = False,
                             keep_original: bool = True) -> pd.DataFrame:
    """
    Creates polynomial and interaction features.
    
    Parameters:
        data (pd.DataFrame): Input DataFrame
        columns (Union[str, List[str]]): Columns to create polynomials from
        degree (int): Highest polynomial degree
        interaction_only (bool): If True, only interaction features are created
        keep_original (bool): Whether to keep original columns, default is True
        
    Returns:
        pd.DataFrame: DataFrame
            - If keep_original=True, includes original features and new polynomial features
            - If keep_original=False, includes only non-specified original columns and new polynomial features
        
    Exceptions:
        ValueError: When parameters are invalid or columns are not numeric type
    """
    if isinstance(columns, str):
        columns = [columns]
        
    # Validate columns
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"Columns not found: {missing}")
        
    # Check numeric
    non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(data[col])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns: {non_numeric}")
        
    if degree < 1:
        raise ValueError("Degree must be >= 1")
        
    # Create result DataFrame
    result = data.copy()
    
    # Create polynomial features
    poly_features = pd.DataFrame(index=data.index)
    
    # Create single column polynomials
    if not interaction_only:
        for col in columns:
            # Always start from 2nd degree, as 1st degree is just the original column
            start_degree = 2
            for d in range(start_degree, degree + 1):
                poly_features[f"{col}^{d}"] = data[col] ** d
            
            # If original columns are kept, add 1st degree (i.e., a copy of the original column)
            if keep_original:
                poly_features[col] = data[col]
                
    # Create interactions
    if len(columns) > 1:
        for d in range(2, degree + 1):
            for combo in combinations(columns, min(d, len(columns))):
                name = ' * '.join(combo)
                poly_features[name] = data[list(combo)].prod(axis=1)
    
    # If original columns are not kept, remove these columns from the result
    if not keep_original:
        result = result.drop(columns=columns)
    
    # Add polynomial features to the result
    result = pd.concat([result, poly_features], axis=1)
    
    # Ensure original columns are not duplicated
    if keep_original:
        # Remove duplicate columns (keep the first occurrence)
        result = result.loc[:, ~result.columns.duplicated()]
                
    return result

def discretize_features(data: pd.DataFrame,
                       columns: Union[str, List[str]],
                       method: str = 'equal_width',
                       n_bins: int = 10,
                       labels: Optional[List[str]] = None,
                       keep_original: bool = True,
                       return_numeric: bool = True) -> pd.DataFrame:
    """
    Discretizes continuous features into categorical data.
    
    Parameters:
        data (pd.DataFrame): Input DataFrame
        columns (Union[str, List[str]]): Columns to discretize
        method (str): Discretization method:
            - 'equal_width': Equal-width binning
            - 'equal_freq': Equal-frequency binning
            - 'kmeans': K-means clustering binning
        n_bins (int): Number of bins
        labels (List[str], optional): Labels for bins
        keep_original (bool): Whether to keep original columns, default is True
        return_numeric (bool): Whether to return numeric results, default is True
            - True: Returns integers 0 to n_bins-1
            - False: Returns categorical variables or interval objects
        
    Returns:
        pd.DataFrame: DataFrame
            - If keep_original=True, includes original columns and discretized columns (with _bin suffix)
            - If keep_original=False, includes only discretized columns
        
    Exceptions:
        ValueError: When invalid parameters are specified
    """
    if isinstance(columns, str):
        columns = [columns]
        
    result = data.copy()
    
    for col in columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"Column {col} must be numeric")
            
        # Create new column name with suffix
        new_col = f"{col}_bin" if keep_original else col
        
        # If numeric results are needed and no labels provided, use integers as labels
        numeric_labels = None
        if return_numeric and labels is None:
            numeric_labels = list(range(n_bins))
            
        if method == 'equal_width':
            # Use pd.cut for equal-width binning
            result[new_col] = pd.cut(data[col], bins=n_bins, labels=labels if not return_numeric else numeric_labels)
        elif method == 'equal_freq':
            try:
                # Try using pd.qcut for equal-frequency binning, automatically handle duplicates
                result[new_col] = pd.qcut(data[col], q=n_bins, labels=labels if not return_numeric else numeric_labels, duplicates='drop')
            except ValueError as e:
                # If error occurs, print warning and use equal-width binning as fallback
                import warnings
                warnings.warn(f"Equal-frequency binning failed: {str(e)}. Using equal-width binning as a fallback.")
                result[new_col] = pd.cut(data[col], bins=n_bins, labels=labels if not return_numeric else numeric_labels)
        elif method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_bins)
            result[new_col] = kmeans.fit_predict(data[col].values.reshape(-1, 1))
            if labels and not return_numeric:
                result[new_col] = result[new_col].map(dict(enumerate(labels)))
        else:
            raise ValueError(f"Unknown discretization method: {method}")
            
        # If numeric results needed but no numeric labels used, convert categorical to numeric
        if return_numeric and labels is not None and not pd.api.types.is_numeric_dtype(result[new_col]):
            # Create category to number mapping
            cat_to_num = {cat: i for i, cat in enumerate(result[new_col].unique())}
            result[new_col] = result[new_col].map(cat_to_num)
            
        # If result is Interval type, convert to integer
        if return_numeric and pd.api.types.is_interval_dtype(result[new_col]):
            # Use interval index as numeric value
            result[new_col] = result[new_col].cat.codes
        
        # If original columns are not kept, delete original column
        if not keep_original:
            result = result.drop(columns=[col])
            
    return result