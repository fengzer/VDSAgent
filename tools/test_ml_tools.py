import pandas as pd
import numpy as np
import pytest
from ml_tools import fill_missing_values_tools, handle_outliers_tools, encode_categorical_tools, transform_features, reduce_dimensions, select_features, create_polynomial_features, discretize_features

def test_fill_missing_values():
    """Test missing value filling function"""
    # Prepare test data
    data = pd.DataFrame({
        'num': [1, 2, None, 4, 5, None],
        'cat': ['A', 'B', None, 'B', 'C', None],
        'group': ['g1', 'g1', 'g1', 'g2', 'g2', 'g2']
    })
    
    # Test automatic filling method
    result = fill_missing_values_tools(data.copy(), ['num', 'cat'], method='auto')
    assert result['num'].isna().sum() == 0, "Numeric columns should be filled"
    assert result['cat'].isna().sum() == 0, "Categorical columns should be filled"
    assert result['num'].mean() == 3.0, "Numeric columns should be filled with mean"
    assert result['cat'].value_counts()['B'] >= 2, "Categorical columns should be filled with mode"
    
    # Test grouped filling
    result = fill_missing_values_tools(
        data.copy(), 
        ['num', 'cat'], 
        method='mean',
        group_columns='group'
    )
    assert result['num'].isna().sum() == 0, "Numeric columns should be filled after grouping"
    
    # Test constant filling
    result = fill_missing_values_tools(
        data.copy(), 
        ['num', 'cat'], 
        method='constant',
        fill_value=999
    )
    assert (result.loc[result['num'].isna(), 'num'] == 999).all(), "Should use specified constant to fill"

def test_handle_outliers():
    """Test outlier handling function"""
    # Prepare test data
    data = pd.DataFrame({
        'value': [1, 2, 3, 100, 4, 5, -50, 6],
        'group': ['g1', 'g1', 'g1', 'g1', 'g2', 'g2', 'g2', 'g2']
    })
    
    # Test IQR method
    result = handle_outliers_tools(
        data.copy(), 
        'value',
        method='iqr',
        strategy='clip'
    )
    assert result['value'].max() < 100, "Max value should be clipped"
    assert result['value'].min() > -50, "Min value should be clipped"
    
    # Test grouped outlier handling
    result = handle_outliers_tools(
        data.copy(),
        'value',
        method='zscore',
        strategy='clip',
        group_columns='group'
    )
    assert len(result) == len(data), "No rows should be deleted"
    
    # Test remove strategy
    result = handle_outliers_tools(
        data.copy(),
        'value',
        method='iqr',
        strategy='remove'
    )
    assert len(result) < len(data), "Rows with outliers should be deleted"

def test_encode_categorical():
    """Test categorical feature encoding function"""
    # Prepare test data
    data = pd.DataFrame({
        'cat_high': ['A', 'B', 'C', 'A', 'B', 'A'],
        'cat_low': ['X', 'X', 'Y', 'Y', 'Z', 'Z'],
        'group': ['g1', 'g1', 'g1', 'g2', 'g2', 'g2']
    })
    
    # Test automatic encoding method (now unified to one-hot encoding)
    result = encode_categorical_tools(data.copy(), ['cat_high', 'cat_low'])
    # Check one-hot encoded columns for cat_high
    assert 'cat_high_A' in result.columns, "Should create one-hot encoded column cat_high_A"
    assert 'cat_high_B' in result.columns, "Should create one-hot encoded column cat_high_B"
    assert 'cat_high_C' in result.columns, "Should create one-hot encoded column cat_high_C"
    # Check one-hot encoded columns for cat_low
    assert 'cat_low_X' in result.columns, "Should create one-hot encoded column cat_low_X"
    assert 'cat_low_Y' in result.columns, "Should create one-hot encoded column cat_low_Y"
    assert 'cat_low_Z' in result.columns, "Should create one-hot encoded column cat_low_Z"
    
    # Test label encoding
    result = encode_categorical_tools(data.copy(), ['cat_high'], method='label')
    assert 'cat_high_encoded' in result.columns, "Should create label encoded column"
    
    # Test frequency encoding
    result = encode_categorical_tools(
        data.copy(), 
        ['cat_high'],
        method='frequency'
    )
    assert 'cat_high_freq' in result.columns, "Should create frequency encoded column"
    assert result['cat_high_freq'].max() <= 1, "Frequency values should be between 0 and 1"
    
    # Test grouped encoding
    result = encode_categorical_tools(
        data.copy(),
        ['cat_high'],
        method='count',
        group_columns='group'
    )
    assert 'cat_high_count' in result.columns, "Should create count encoded column"

def test_edge_cases():
    """Test edge cases"""
    # Empty DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        fill_missing_values_tools(empty_df, ['col'])
    
    # Non-existent column
    data = pd.DataFrame({'A': [1, 2, 3]})
    with pytest.raises(ValueError):
        fill_missing_values_tools(data, ['B'])
    
    # Column with all missing values
    data = pd.DataFrame({'A': [None, None, None]})
    result = fill_missing_values_tools(data, ['A'], method='constant', fill_value=0)
    assert result['A'].isna().sum() == 0, "All missing values should be filled"

def test_transform_features():
    """Test feature transformation function"""
    # Prepare test data
    data = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'num2': [10, 20, 30, 40, 50],
        'group': ['A', 'A', 'B', 'B', 'B']
    })
    
    # Test standardization transformation (keep original columns)
    result = transform_features(data.copy(), ['num1', 'num2'], method='standard')
    assert 'num1' in result[0].columns, "Original columns should be kept"
    assert 'num1_standard' in result[0].columns, "Should create standardized column"
    assert result[0]['num1_standard'].mean() < 1e-10, "Standardized mean should be close to 0"
    
    # Test standardization transformation (do not keep original columns)
    result = transform_features(data.copy(), ['num1'], method='standard', keep_original=False)
    assert 'num1' not in result[0].columns, "Original columns should not be kept"
    assert 'num1_standard' in result[0].columns, "Should only contain standardized columns"
    
    # Test log transformation
    result = transform_features(data.copy(), 'num1', method='log')
    assert 'num1_log' in result[0].columns, "Should create log transformed column"
    
    # Test error cases
    with pytest.raises(ValueError):
        transform_features(data, ['group'], method='standard')  # Non-numeric column
    with pytest.raises(ValueError):
        transform_features(data, ['not_exist'])  # Non-existent column

def test_reduce_dimensions():
    """Test dimensionality reduction function"""
    # Prepare test data
    data = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6, 7, 8],
        'f2': [2, 4, 6, 8, 10, 12, 14, 16],
        'f3': [1, 3, 5, 7, 2, 4, 6, 8]
    })
    # Create target variable with three categories
    target = pd.Series([0, 0, 0, 1, 1, 1, 2, 2])
    
    # Test PCA (keep original features)
    result = reduce_dimensions(data.copy(), method='pca', n_components=2)
    assert 'PC1' in result.columns, "Should include first principal component"
    assert 'PC2' in result.columns, "Should include second principal component"
    assert 'f1' in result.columns, "Original features should be kept"
    
    # Test PCA (do not keep original features)
    result = reduce_dimensions(data.copy(), method='pca', n_components=2, keep_original=False)
    assert len(result.columns) == 2, "Should only contain two principal components"
    
    # Test LDA (with three categories, can get two discriminant axes)
    result = reduce_dimensions(
        data.copy(), 
        method='lda',
        n_components=2,
        target=target
    )
    assert 'LD1' in result.columns, "Should include first discriminant axis"
    assert 'LD2' in result.columns, "Should include second discriminant axis"
    
    # Test error cases
    with pytest.raises(ValueError):
        reduce_dimensions(data, method='lda')  # Missing target variable
    with pytest.raises(ValueError):
        reduce_dimensions(data, method='invalid')  # Invalid method

def test_select_features():
    """Test feature selection function"""
    # Prepare test data
    data = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5],
        'f2': [2, 4, 6, 8, 10],
        'f3': [0.1, 0.1, 0.1, 0.1, 0.2]
    })
    target = pd.Series([0, 0, 1, 1, 1])
    
    # Test variance selection (keep original features)
    result = select_features(data.copy(), method='variance', n_features=2)
    assert 'f1' in result, "Original features should be kept"
    
    # Test correlation selection (do not keep original features)
    result = select_features(
        data.copy(),
        target=target,
        method='correlation',
        n_features=2,
    )
    assert len(result) == 2, "Should only keep two selected features"
    
    # Test mutual information selection
    result = select_features(
        data.copy(),
        target=target,
        method='mutual_info',
        n_features=2
    )
    assert len(result) == 2, "Should only keep two selected features"
    
    # Test error cases
    with pytest.raises(ValueError):
        select_features(data, method='correlation')  # Missing target variable

def test_create_polynomial_features():
    """Test polynomial feature creation function"""
    # Prepare test data
    data = pd.DataFrame({
        'x1': [1, 2, 3],
        'x2': [0, 1, 2]
    })
    
    # Test basic polynomial features (keep original features)
    result = create_polynomial_features(data.copy(), ['x1'], degree=2)
    assert 'x1' in result.columns, "Original features should be kept"
    assert 'x1^2' in result.columns, "Should create squared term"
    
    # Test interaction features (do not keep original features)
    result = create_polynomial_features(
        data.copy(),
        ['x1', 'x2'],
        degree=2,
        keep_original=False
    )
    assert 'x1 * x2' in result.columns, "Should create interaction term"
    assert 'x1' not in result.columns, "Original features should not be included"
    
    # Test only create interaction terms
    result = create_polynomial_features(
        data.copy(),
        ['x1', 'x2'],
        degree=2,
        interaction_only=True
    )
    assert 'x1^2' not in result.columns, "Squared term should not be created"
    assert 'x1 * x2' in result.columns, "Should only contain interaction terms"
    
    # Test error cases
    with pytest.raises(ValueError):
        create_polynomial_features(data, ['not_exist'])  # Non-existent column
    with pytest.raises(ValueError):
        create_polynomial_features(data, ['x1'], degree=0)  # Invalid degree

def test_discretize_features():
    """Test feature discretization function"""
    # Prepare test data
    data = pd.DataFrame({
        'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C']
    })

    # Test equal-width binning (keep original features)
    result = discretize_features(data.copy(), 'value', method='equal_width', n_bins=3)
    assert 'value' in result.columns, "Should keep original features"
    assert 'value_bin' in result.columns, "Should create binned features"
    assert result['value_bin'].nunique() == 3, "Should create 3 bins"

    # Test equal-frequency binning (do not keep original features)
    result = discretize_features(
        data.copy(),
        'value',
        method='equal_freq',
        n_bins=4,
        keep_original=False
    )
    assert 'value' not in result.columns, "Should not keep original features"
    assert len(result.columns) == 1, "Should only contain binned features"

    # Test custom labels
    labels = ['low', 'medium', 'high']
    result = discretize_features(
        data.copy(),
        'value',
        method='equal_width',
        n_bins=3,
        labels=labels,
        return_numeric=False
    )
    assert set(result['value_bin'].unique()) == set(labels), "Should use custom labels"

    # Test error cases
    try:
        result = discretize_features(data.copy(), 'group', method='equal_width', n_bins=3)
        assert False, "Should raise error for non-numeric column"
    except ValueError:
        pass

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__]) 