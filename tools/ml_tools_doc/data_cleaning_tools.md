## fill_missing_values_tools

**Name:** fill_missing_values_tools  
**Description:** Fill missing values in specified columns of a DataFrame. This tool can handle missing value imputation for both numerical and categorical features, supporting multiple filling methods.  
**Use Cases:** Handle missing values for various features, supports group-based filling and time series filling

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame
- `target_columns`:
  - **Type:** `string | List[str]`
  - **Description:** Target columns to fill
- `method`:
  - **Type:** `string`
  - **Description:** Filling method
  - **Options:** `auto` | `mean` | `median` | `mode` | `ffill` | `bfill` | `interpolate` | `constant` | `knn`
  - **Default:** `auto`
- `group_columns`:
  - **Type:** `string | List[str] | None`
  - **Description:** Grouping columns for group-based filling
  - **Default:** `None`
- `time_column`:
  - **Type:** `string | None`
  - **Description:** Time column for time series related filling
  - **Default:** `None`
- `fill_value`:
  - **Type:** `Any | None`
  - **Description:** Fill value when using constant method
  - **Default:** `None`
- `max_group_null_ratio`:
  - **Type:** `float`
  - **Description:** Maximum allowed missing ratio within groups
  - **Default:** `0.8`


**Required Parameters:** `data`, `target_columns`  
**Returns:** DataFrame after filling  
**Notes:**
- `auto` method automatically selects filling method based on data type (mean for numeric, mode for categorical)
- Supports group-based filling to better preserve data distribution characteristics
- Supports forward/backward filling and interpolation for time series data
- Recommend choosing appropriate filling method based on actual business scenario
- When using batch filling operations, each filling operation needs to specify four parameters: target columns, filling method, group columns, and fill value. For methods that don't need fill value (like mean, median, mode), fill value parameter should be set to None

---

## remove_columns_tools

**Name:** remove_columns_tools  
**Description:** Remove columns from DataFrame based on multiple strategies. Supports various deletion strategies based on missing value ratio, constant value ratio, correlation, and variance, also supports directly specifying column names to delete.  
**Use Cases:** Feature selection during data preprocessing, remove useless or redundant feature columns

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame
- `strategy`:
  - **Type:** `string | List[str] | None`
  - **Description:** Removal strategy
  - **Options:** `missing` | `constant` | `correlation` | `variance` | `None`
  - **Default:** `missing`
- `columns`:
  - **Type:** `List[str] | None`
  - **Description:** List of column names to directly remove
  - **Default:** `None`
- `threshold`:
  - **Type:** `float | Dict[str, float]`
  - **Description:** Thresholds for various strategies
  - **Default:** `0.5`
- `exclude_columns`:
  - **Type:** `List[str] | None`
  - **Description:** Columns to exclude from checking
  - **Default:** `None`
- `min_unique_ratio`:
  - **Type:** `float`
  - **Description:** Minimum unique value ratio
  - **Default:** `0.01`
- `correlation_threshold`:
  - **Type:** `float`
  - **Description:** Correlation threshold
  - **Default:** `0.95`

**Required Parameters:** `data`  
**Returns:** Processed DataFrame  
**Notes:**
- Supports combination of multiple removal strategies
- Can directly specify columns to remove through columns parameter
- Can protect important features from removal through exclude_columns
- correlation strategy only applies to numeric features
- Will warn about columns to be removed through warnings
- Recommend carefully setting thresholds based on business requirements
- When strategy is None, only removes columns specified in columns parameter

---

## handle_outliers_tools

**Name:** handle_outliers_tools  
**Description:** General outlier handling function. Supports multiple outlier detection methods, can handle by groups, and provides multiple handling strategies.  
**Use Cases:** Outlier handling during data cleaning, especially suitable for scenarios requiring group-based outlier handling

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame
- `target_columns`:
  - **Type:** `string | List[str]`
  - **Description:** Target columns to process
- `method`:
  - **Type:** `string`
  - **Description:** Outlier detection method
  - **Options:** `iqr` | `zscore` | `isolation_forest` | `dbscan` | `mad`
  - **Default:** `'iqr'`
- `strategy`:
  - **Type:** `string`
  - **Description:** Handling strategy
  - **Options:** `clip` | `remove`
  - **Default:** `'clip'`
- `sensitivity`:
  - **Type:** `string`
  - **Description:** Sensitivity of outlier detection
  - **Options:** `low` | `medium` | `high`
  - **Default:** `'medium'`
  - **Parameter Description:**
    - `low`: Loose threshold, only detects extreme outliers
    - `medium`: Medium threshold, balanced detection
    - `high`: Strict threshold, detects more outliers
- `group_columns`:
  - **Type:** `string | List[str] | None`
  - **Description:** Grouping columns
  - **Default:** `None`
- `params`:
  - **Type:** `Dict[str, Any] | None`
  - **Description:** Parameter dictionary for each method
  - **Default:** `None`
  - **Parameter Configuration Examples:**
    - iqr: `{'threshold': 1.5}`
    - zscore: `{'threshold': 3}`
    - isolation_forest: `{'contamination': 0.1, 'random_state': 42}`
    - dbscan: `{'eps': 0.5, 'min_samples': 5}`
    - mad: `{'threshold': 3.5}`

**Required Parameters:** `data`, `target_columns`  
**Returns:** Processed DataFrame  
**Notes:**
- Different detection methods are suitable for different data distribution characteristics
- `clip` strategy limits outliers to reasonable range, `remove` strategy directly removes outliers
- Supports group-based outlier handling to better preserve data distribution characteristics
- Can quickly adjust outlier detection strictness through `sensitivity` parameter
- For finer control, can override default settings through `params` parameter
- Recommend choosing appropriate detection method and sensitivity based on data characteristics
- Need to consider whether outlier handling will affect subsequent analysis

---

## encode_categorical_tools

**Name:** encode_categorical_tools  
**Description:** General categorical feature encoding function. Supports multiple encoding methods, can handle unknown categories, supports group-based encoding.  
**Use Cases:** Categorical feature preprocessing, convert categorical variables into numerical form that machine learning algorithms can process

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame
- `target_columns`:
  - **Type:** `string | List[str]`
  - **Description:** Target columns to encode
- `method`:
  - **Type:** `string`
  - **Description:** Encoding method
  - **Options:** `auto` | `label` | `onehot` | `frequency` | `count`
  - **Default:** `'auto'`
- `group_columns`:
  - **Type:** `string | List[str] | None`
  - **Description:** Grouping columns (for group-based encoding)
  - **Default:** `None`
- `handle_unknown`:
  - **Type:** `string`
  - **Description:** How to handle unknown categories
  - **Default:** `'ignore'`
- `keep_original`:
  - **Type:** `bool`
  - **Description:** Whether to keep original categorical columns
  - **Default:** `True`

**Required Parameters:** `data`, `target_columns`  
**Returns:** Encoded DataFrame  
**Notes:**
- `auto` method defaults to one-hot encoding
- One-hot encoding significantly increases feature dimensions
- Frequency encoding and count encoding can preserve category frequency information
- Label encoding is suitable for ordinal categorical variables
- Recommend choosing appropriate encoding method based on feature characteristics and model requirements
- Can control whether to keep original categorical columns through `keep_original` parameter

--- 