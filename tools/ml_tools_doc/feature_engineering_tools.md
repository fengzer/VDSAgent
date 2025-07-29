## transform_features

**Name:** transform_features  
**Description:** Transform and scale features using multiple methods. This tool provides various feature transformation and scaling options to improve model performance. Supports both training and transformation modes.  
**Use Cases:** Feature scaling, standardization, data transformation, particularly suitable for data preparation in machine learning models sensitive to feature scales

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** Pandas DataFrame object representing the dataset
- `columns`:
  - **Type:** `string | array`
  - **Description:** Column names to transform
- `method`:
  - **Type:** `string`
  - **Description:** Transformation method to use
  - **Options:** `standard` | `minmax` | `robust` | `log` | `sqrt` | `power`
  - **Default:** `standard`
- `params`:
  - **Type:** `dict | null`
  - **Description:** Additional parameters for transformer
  - **Default:** `None`
- `keep_original`:
  - **Type:** `boolean`
  - **Description:** Whether to keep original columns
  - **Default:** `True`
- `scaler`:
  - **Type:** `object | dict | null`
  - **Description:** Trained transformer object. Used to ensure test data uses same transformation parameters as training data
  - **Default:** `None`

**Required Parameters:** `data`, `columns`  

**Returns:** 
- **Type:** `Tuple[pd.DataFrame, object]`
- **Description:** Returns a tuple containing:
  1. Transformed DataFrame
  2. Transformer object used (for subsequent test data transformation)

**Usage Modes:**
1. **Training Mode** (scaler=None):
   - Learn transformation parameters from data
   - Return transformed data and trained transformer
2. **Transform Mode** (scaler provided):
   - Transform using parameters from existing transformer
   - Ensure same transformation parameters as training data

**Transformation Results:** 
- New columns are named as: `original_name_method` (e.g., 'age_standard', 'salary_minmax')
- If keep_original=True, both original and transformed columns are kept
- If keep_original=False, only transformed columns are kept

**Method Details:**
- 'standard':
  - Standardize features to zero mean and unit variance
  - Automatically detect extreme values, use robust standardization for them
- 'minmax': Scale features to fixed range [0,1]
- 'robust': Scale using quartiles, insensitive to outliers
- 'log': Apply natural logarithm transformation, suitable for skewed distributions
- 'sqrt': Apply square root transformation, suitable for moderately skewed distributions
- 'power': Apply Yeo-Johnson power transformation, automatically handles negative values

**Exception Handling:**
- For infinite values or NaN, automatically switch to more robust method or fill with 0
- For extreme values, use robust standardization method
- All exceptional cases will trigger warning messages

**Notes:**
- Can only transform numeric columns
- Specified columns must exist in DataFrame
- Training and test data must use same transformer for consistency
- Some methods (like log) automatically handle negative values and zeros

---
## reduce_dimensions

**Name:** reduce_dimensions  
**Description:** Reduce dataset dimensions using PCA or LDA. This tool is used for feature extraction and dimensionality reduction.  
**Use Cases:** Dimensionality reduction, feature extraction, high-dimensional data visualization

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** Pandas DataFrame object representing the dataset
- `method`:
  - **Type:** `string`
  - **Description:** Dimensionality reduction method
  - **Options:** `pca` | `lda`
  - **Default:** `pca`
- `n_components`:
  - **Type:** `int | float`
  - **Description:** Number of components to keep. If float, represents variance ratio for PCA
  - **Default:** `0.95`
- `target`:
  - **Type:** `pd.Series | null`
  - **Description:** Target variable required for LDA method. Required when method='lda'
  - **Default:** `None`
- `keep_original`:
  - **Type:** `boolean`
  - **Description:** Whether to keep original features
  - **Default:** `True`

**Required Parameters:** `data`  
**Results:** 
- Reduced DataFrame will contain new dimension reduction feature columns
- PCA method columns named as: 'PC1', 'PC2', 'PC3'... 
- LDA method columns named as: 'LD1', 'LD2', 'LD3'...
- If keep_original=True, both original and reduced columns are kept
- If keep_original=False, only reduced columns are kept

**Notes:**
- PCA finds directions of maximum variance
- LDA finds directions that maximize class separation
- LDA components cannot exceed min(n_features, n_classes-1)
- Need to balance dimension reduction and information loss
- PCA is unsupervised, while LDA is supervised
- Can only process numeric columns, non-numeric columns will raise error
- Target parameter is required for LDA method
- When n_components is float (e.g., 0.95), PCA keeps components explaining that proportion of variance

---
## select_features

**Name:** select_features  
**Description:** Perform feature selection using various statistical and machine learning methods. This tool provides multiple feature selection methods.  
**Use Cases:** Feature selection, dimensionality reduction, identifying important features

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** Pandas DataFrame object representing the dataset
- `target`:
  - **Type:** `pd.Series | null`
  - **Description:** Target variable required for supervised methods
  - **Default:** `None`
- `method`:
  - **Type:** `string`
  - **Description:** Feature selection method
  - **Options:** `variance` | `correlation` | `mutual_info` | `rfe` | `lasso`
  - **Default:** `variance`
- `n_features`:
  - **Type:** `int | null`
  - **Description:** Number of features to select
  - **Default:** `None`
  - **Default behavior for each method:**
    - variance method: Uses variance threshold when None
    - correlation method: Uses correlation coefficient threshold when None
    - mutual_info method: Defaults to 10 features when None
    - rfe method: Defaults to half features when None
    - lasso method: Number of features controlled by alpha parameter when None
- `params`:
  - **Type:** `dict | null`
  - **Description:** Additional parameters for selector
  - **Default:** `None`
  - **Parameters for each method:**
    - variance: {'threshold': float}  # Variance threshold, features below this will be removed
    - correlation: {'threshold': float}  # Correlation coefficient threshold, features below this will be removed
    - mutual_info: {'k': int}  # Number of features to select per round
    - rfe: {'step': int}  # Number of features to remove per round
    - lasso: {'alpha': float}  # L1 regularization strength, higher values select fewer features

**Required Parameters:** `data`  
**Returns:** `List[str]`: Returns list of selected feature names

**Notes:**
- 'variance': Remove low variance features
- 'correlation': Select based on correlation with target
- 'mutual_info': Select using mutual information
- 'rfe': Perform recursive feature elimination
- 'lasso': Select using L1 regularization
- Some methods require target variable
- Consider interpretability of selected features

---
## discretize_features

**Name:** discretize_features  
**Description:** Convert continuous features to discrete categories. This tool provides multiple methods for binning continuous variables.  
**Use Cases:** Feature discretization, continuous variable binning, creating categorical features

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** Pandas DataFrame object representing the dataset
- `columns`:
  - **Type:** `string | array`
  - **Description:** Column names to discretize
- `method`:
  - **Type:** `string`
  - **Description:** Discretization method
  - **Options:** `equal_width` | `equal_freq` | `kmeans`
  - **Default:** `equal_width`
- `n_bins`:
  - **Type:** `int`
  - **Description:** Number of bins to create
  - **Default:** `10`
- `labels`:
  - **Type:** `array | null`
  - **Description:** Labels for bins
  - **Default:** `None`
- `keep_original`:
  - **Type:** `boolean`
  - **Description:** Whether to keep original continuous features
  - **Default:** `True`

**Required Parameters:** `data`, `columns` 
**Returns:** `pd.DataFrame`: DataFrame containing new discretized feature columns

**Notes:**
- 'equal_width': Create equal-width bins
- 'equal_freq': Create equal-frequency bins
- 'kmeans': Use k-means clustering for binning
- New columns named as: `original_name_bin` (e.g., 'age_bin', 'salary_bin')
- If keep_original=True, both original and discretized columns are kept, if keep_original=False, only discretized columns are kept
- Returns only one parameter

---
## create_polynomial_features

**Name:** create_polynomial_features  
**Description:** Create polynomial and interaction features. This tool can generate higher-order terms of features and interaction terms between features.  
**Use Cases:** Non-linear feature engineering, feature interaction modeling, capturing complex relationships

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** Pandas DataFrame object representing the dataset
- `columns`:
  - **Type:** `string | array`
  - **Description:** Column names to create polynomial features
- `degree`:
  - **Type:** `int`
  - **Description:** Maximum polynomial degree
  - **Default:** `2`
- `interaction_only`:
  - **Type:** `boolean`
  - **Description:** Whether to create only interaction features
  - **Default:** `False`
- `keep_original`:
  - **Type:** `boolean`
  - **Description:** Whether to keep original columns
  - **Default:** `True`

**Required Parameters:** `data`, `columns`  

**Results:** 
- Generated DataFrame will contain new polynomial feature columns
- Single variable polynomial features named as: `variable^degree` (e.g., 'age^2', 'salary^3')
- Interaction features named as: `variable1 * variable2` (e.g., 'age * salary')
- If keep_original=True, both original and new feature columns are kept
- If keep_original=False, only new feature columns are kept

**Notes:**
- Can only process numeric columns
- degree must be greater than or equal to 1
- When interaction_only=True, only generates interaction features, not single variable higher-order terms
- Number of interaction features grows rapidly with degree
- Recommend standardizing features before use to avoid numerical overflow
- Too high degree may lead to overfitting