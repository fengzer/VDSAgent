# 1. Base System Message Template
PREDICTION_INFERENCE_TEMPLATE = """You are a data science expert, focusing on feature engineering, model training, and predictive inference. You excel at using various feature engineering techniques to improve model performance and making predictions using trained models based on processed datasets.
1. Definition of Data Science Terms:
   1. Prediction Problem:
      The goal of a prediction problem is to predict unobserved values of the response variable in future data. Most prediction algorithms work by approximating relationships between observed (current/historical) response values and predictive features (other variables). If these fundamental observed relationships continue to hold in the future, they can be used to generate predictions for unobserved responses in future data.
   2. Prediction Algorithms:
      Prediction algorithms aim to predict the values of a response variable based on the values of predictor variables (also known as covariates or predictive features). Prediction algorithms typically work by finding some specific combination of predictor variables whose combined value approximates the actual response value as closely as possible. If an algorithm can be shown to work well on data where the response is known, it can also be used to predict cases where the response is unknown in similar future data (assuming the relationships between predictor variables and response captured by the algorithm hold in future data)
   3. Quantitative Metrics for Predictive Performance:
      1. Mean Squared Error (MSE), Mean Absolute Error (MAE), and Median Absolute Deviation (MAD) are three performance metrics for evaluating continuous response predictions. MSE and MAE are calculated by applying squared loss and absolute loss functions to observed and predicted responses. MAD is calculated using the median (rather than the mean) of absolute value losses between observed and predicted responses. Lower values of each metric indicate better predictive performance.
      2. Correlation Coefficient and R-squared: A correlation coefficient close to 0 between observed and predicted responses indicates no apparent linear relationship (very poor performance), while a correlation coefficient close to 1 indicates a very strong linear relationship between predicted and observed responses.
   4. Response Variable:
      The response variable is the variable to be predicted in a prediction problem.
   5. Predictor Variables:
      Predictor variables are the variables used to predict the response variable in a prediction problem.

3. Note: If numerical conversion of discrete variables is necessary, please use Label encoder

4. In data cleaning and EDA exploration, you have obtained the following information:
Data Science Project Description: {problem_description}
Data Background Description: {context_description}
EDA Analysis Summary: {eda_summary}
"""

# 2. Feature Engineering Suggestion Generation Template
FEATURE_ENGINEERING_TEMPLATE = """Task: Feature Engineering Suggestion Generation
Task Description: Generate feature engineering suggestions based on the problem description and background information.

Input Data:
1. Data Science Project Description: {problem_description}
2. Data Background Description: {context_description}
3. Column Names: {column_names}

Note:
1. At least two methods must be generated, and preferably three:
   - The first is a feature creation/transformation method (required), which can be selected from the following list (e.g., PCA dimensionality reduction, LDA dimensionality reduction, discretization, standardization, normalization, logarithmic transformation, square root transformation, etc.)
   - The second is a feature combination based on business understanding: (best to complete)
     * Generate new features with actual business and physical meaning based on existing column names
   - The third is a feature selection method (required), which can be selected from the following list (e.g., variance selection, correlation coefficient selection, mutual information selection, RFE selection, etc.)
2. For feature creation/transformation methods, the description should indicate that all possible predictor variables should be transformed. No conversion is needed for id type columns.

Output Data:
A JSON list containing feature engineering suggestions, formatted as follows:
```json
[
  {{"Feature Engineering Method": "Feature Creation/Transformation Method", "Description": "Detailed description of why this feature engineering method is suitable for the data science project"}},
  {{"Feature Engineering Method": "Business Feature Creation", "Description": "Detailed description of why this new feature has physical meaning and why it is suitable for the data science project"}},
  {{"Feature Engineering Method": "Feature Selection Method", "Description": "Detailed description of why this feature engineering method is suitable for the data science project"}}
]
```
"""

# 3. Modeling Method Suggestion Generation Template
MODELING_METHODS_TEMPLATE = """Task: Modeling Method Suggestion Generation
Task Description: Generate up to three modeling methods from best to worst based on the problem description and background information.

Input Data:
1. Data Science Project Description: {problem_description}
2. Data Background Description: {context_description}
3. EDA Analysis Summary: {eda_summary}

Output Data:
A JSON list containing up to three modeling methods, formatted as follows:
```json
[
  {{"Method": "Modeling Method", "Description": "Detailed description of how this method is implemented"}},
  {{"Method": "Modeling Method", "Description": "Detailed description of how this method is implemented"}}
]
```
"""
COMBINED_MODEL_CODE_TEMPLATE = """Task: Model Training and Evaluation Code Generation
Task Description: Generate Python code for training and evaluating based on the specified modeling method and feature engineering method.

Available tools function:
{tools}
These functions do not need to be reimplemented and can be called directly by name.

tools function descriptions:
{tool_descriptions}

Input Data:
1. Models: {models}
2. Data Science Project: {problem_description}
3. Data Background: {context_description}
4. csv data path: {csv_path}
5. csv data variable preview: {csv_columns}
6. Feature Engineering Methods: {feature_engineering_methods}
7. Response variable and column information: {response_variable}

Output Requirements:
Please generate Python code based on the following requirements:
0. After reading the dataset, first split it into training and test sets to ensure that the three model methods use the same training and test sets for training and evaluation.
1. tools function has been imported by default, and you do not need to re-import the function. The function should correctly call tools function according to the description document to perform data processing, and expand implementation of other processing functions if necessary.
2. Before starting feature engineering, first separate the response variable and predictor variables. Do not perform feature transformation on the response variable! Transform all predictor variables. In this step, keep_original is set to True.
3. When calling transform_features on the training set, do not fill in the scale parameter, and when calling transform_features on the test set, fill in the scaler parameter trained during training set conversion to obtain the feature pool. If a binning strategy is required, perform numerical binning, and after binning, do not use string labels.
4. Initialize 3 different feature numbers (n1, n2, n3), perform feature selection on all variables in the feature pool, and receive all selected feature lists. Obtain three feature sets, and each model must use these three different feature sets for training and evaluation.
5. Train the model using the training set. Check if the response variable is leaked into the predictor variable before training.
6. Use k-fold cross-validation to evaluate model performance using the relevant evaluation metric based on the data science project, and if the response variable is discrete, the evaluation result should also be discrete.
7. The results should be stored in the variable result, which should include:
  - The selected feature list for each feature number
  - The evaluation metrics and performance results of each model at each feature number (a total of 9 results)
8. Write the feature engineering and model training evaluation part as a complete function, with csv_path as the input parameter and result as the return value.
9. For the logistic regression model, at least 2000 iterations are required.

Output Data:
Generated Python code, formatted as follows:
```python
def train_and_evaluate_models(csv_path: str) -> list: 
    data = pd.read_csv(csv_path)

    # Feature count list
    feature_counts = [n1, n2, n3]  # Three different feature numbers

    [Feature Engineering and Feature Selection]
    [Model Training and Evaluation, a total of 9 models are trained]
    
    # Store all combination results
    all_combinations = []
    
    # Iterate through all model and feature number combinations
    for model_name, model in models.items():
        for feature_count in feature_counts:
            # Get current feature set
            features = selected_features_sets[feature_count]
            
            # Train and evaluate model
            [Training Model Code and Evaluation Code]
            
            # Add result to list
            combination = {{
                'dataset': os.path.basename(csv_path),
                'model': model_name,
                'feature_count': feature_count,
                'features': features,
                'metrics': {{
                    'Metric1': ...,
                    'Metric2': ...,
                    'Metric3': ...,
                    '...': ...
                }}
            }}
            all_combinations.append(combination)
    
    return all_combinations

result = train_and_evaluate_models(csv data path)
print(result)
```
"""

# 5. Response Variable Identification Template
RESPONSE_VARIABLE_TEMPLATE = """Task: Response Variable Identification
Task Description: Analyze and identify the response variable to be predicted based on the problem description, background information, and column names of the dataset.

Input Data:
1. Data Science Project Description: {problem_description}
2. Data Background Description: {context_description}
3. Column Names: {column_names}

Note:
1. The response variable is the target variable to be predicted in a prediction problem, which may have one or more (e.g., TRUE/FALSE, YES/NO).
2. It must be clearly specified for each variable whether it is continuous or discrete.
3. The response variable must be an existing column name in the dataset.

Output Data:
A JSON array containing response variable information, formatted as follows:
```json
[
    {{
        "Response Variable": "Variable Name 1",
        "Variable Type": "Continuous/Discrete",
        "Explanation": "Explanation of why this variable is the response variable"
    }},
    {{
        "Response Variable": "Variable Name 2",
        "Variable Type": "Continuous/Discrete",
        "Explanation": "Explanation of why this variable is the response variable"
    }}
]
```

Note: If there is only one response variable, only one element is included in the array.
"""

# 6. Batch Dataset Evaluation Template
BATCH_EVALUATION_TEMPLATE = """Task: Batch Dataset Evaluation Code Generation
Task Description: Generate code for evaluating multiple datasets based on existing model training evaluation code.

Input Data:
1. Dataset Directory Path: {datasets_dir}
2. Existing Feature Engineering Model Training Code Content: {model_code}
3. Data Science Project Description: {problem_description}
4. Data Background Description: {context_description}

First, select the most appropriate evaluation metric based on the data science project description:
- For regression problems:
  - If particularly concerned about prediction accuracy, prioritize MSE (Mean Squared Error)
  - If concerned about model interpretability, prioritize R2
- For classification problems:
  - Accuracy (Accuracy): Suitable for balanced classification problems
  - F1 Score: Suitable for unbalanced classification problems

Iterate through all CSV datasets in the specified directory, call the training function for each dataset, collect evaluation metrics for all dataset-algorithm-feature number combinations, sort the results according to the selected evaluation metric, and output the top 5 dataset-algorithm-feature number combinations with the best performance, storing these 5 combinations in the result variable and outputting.

Notes:
- Your code needs to import all packages used in the existing model training evaluation code.
1. The existing model training code contains the function train_and_evaluate_models(csv_path), which:
   - Takes a single dataset path as input
   - Returns a dictionary containing multiple model evaluation results, formatted as:
     {{
       'top_5_combinations': [
         {{
           'dataset': 'Dataset Name',
           'model': 'Model Name',
           'feature_count': 'Feature Number',
           'features': ['Feature 1', 'Feature 2', ...],  # Feature list used for this combination
           'metrics': {{
             'metric1': value1,
             'metric2': value2,
             ...
           }}
         }},
         // ... other 4 best combinations ...
       ],
       'evaluation_metric': 'Name of the evaluation metric used for sorting'
     }}
     2. Do not reimplement the model training and evaluation logic, fully reproduce the complete logic of the uploaded function, do not omit any code
3. The main task is to iterate through all datasets in the directory and collect evaluation results to find the best combination.
4. Please ensure that the evaluation metrics of the five found combinations are different, if they are the same, replace one of the combinations with a combination whose evaluation metrics are the same.
5. For the final five dataset-algorithm-feature number combinations, perform visualization of the relevant evaluation metrics:
   - Create a bar chart showing the evaluation metric scores of these 5 best combinations
   - X-axis displays combination information (Dataset Name, Model Name, Feature Number)
   - Y-axis displays the selected evaluation metric scores
   - Display specific evaluation metric values above each bar
   - The chart title should include the name of the evaluation metric used
   - Store the image in a plot folder in the same directory as the dataset, named 'top_5_combinations.png'

Output Data:
Generated Python code, formatted as follows:
```python
def train_and_evaluate_models(csv_path):
    [Complete reproduction of uploaded code]

def evaluate_all_datasets(datasets_dir: str) -> dict:
    all_results = []  # Store all combination results
    csv_files = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]
    
    # Evaluate each dataset
    for csv_file in csv_files:
        csv_path = os.path.join(datasets_dir, csv_file)
        result = train_and_evaluate_models(csv_path)
        [Store result in result_all]

    
    [Find the best 5 combinations, these 5 combinations have different evaluation metric values, if the same, replace the combination. And store in result_best_5 variable]
    return result_best_5

[Call the function, store the evaluation result dictionary in the result variable, conform to the above structure, with top_5_combinations and evaluation_metric as required fields]
```
"""

# Add Result Summary Template
RESULT_SUMMARY_TEMPLATE = """Task: Model Evaluation Result Summary
Task Description: Sort and clearly display detailed information for each combination based on performance.

Input Data:
Evaluation Result Dictionary: {results}

Output Format:
```markdown
# Model Evaluation Ranking

## Top 1
- Dataset:
- Model:
- Feature Number:
- Used Features:
- Evaluation Metrics and Corresponding Scores:

## Top 2
[Similar format]

## Top 3
[Similar format]

## Top 4
[Similar format]

## Top 5
[Similar format]
```
"""