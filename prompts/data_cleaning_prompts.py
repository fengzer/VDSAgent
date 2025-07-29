DATA_CLEANING_TEMPLATE = """You are a data science expert, focusing on dataset cleaning, creating data cleaning operation functions, and EDA analysis. You will list one or more data cleaning operations and conduct EDA analysis based on my data and tasks.

1. Definition of Data Science Terms:
   1. Data cleaning:
      Data cleaning is the process of modifying data so it is as unambiguous as possible and will be correctly interpreted by your computer. There are many different possible clean versions of every dataset, so we suggest writing a function that allows you to create multiple alternative versions of your cleaned dataset, enabling you to investigate whether your downstream results are stable to any judgment calls that you made. 
   2. Preprocessing:
      Preprocessing is the process of modifying your clean data so that it is suited for the application of a specific algorithm or analysis. Like data cleaning, preprocessing involves making judgment calls, and we recommend writing a preprocessing function with the various judgment call options as the arguments, enabling you to investigate the stability of your downstream results to the preprocessing judgment calls.
   3. Action Items:
      Data cleaning and preprocessing action items are unique modifications that you will make to your data to clean and preprocess it. These action items are determined based on the issues and ambiguities identified in the data from learning about the background domain information, the data collection process, as well from an initial exploration of the data. Every dataset will have its own unique set of action items, and each action item may include several judgment call options.
   4. Exploratory and Explanatory Data Analysis:
      Exploratory data analysis (EDA) involves exploring the data in order to identify the patterns and relationships it contains. We recommend using a question-and-answer-based workflow (guided by domain knowledge and the project goal) to create a broad range of relevant "rough" data visualizations and numeric summaries. Having identified the most interesting and relevant patterns and trends, explanatory data analysis is the act of producing polished figures for communicating the particularly insightful exploratory findings to an external audience.

2. Technical Analysis Description and Possible Operations:
   1. Techniques for Identifying Invalid or Inconsistent Values
      1. Exploration Methods:
         • Look at randomly selected rows in the data, as well as sequential rows (e.g., the first and last 10 rows). Do you see anything surprising?
         • Print out the smallest (minimum), largest (maximum), and average values of each numeric column. Do these make sense, given your understanding of what the variable is measuring?
         • Look at histograms (see Section 5.2.1.1 in Chapter 5 for a description of histograms) of the distributions of each numeric variable. This can help you to visually identify any strange or inconsistent values.
         • Print out the unique values of categorical variables. Do the categories make sense?
   2. Techniques for Identifying and Handling Missing Values:
      1. Techniques for Identifying and Summarizing Missing Values
         • Print the smallest (minimum) and largest (maximum) values of each numeric column. Do these seem unusual in any way?
         • Print the unique values of categorical variables. Might any of the categorical values correspond to missing values (e.g., are there levels such as "unknown" or "missing")?
         • Make histograms of each continuous variable to reveal unusual values (e.g., if there is a surprising abundance of "99" values, then these might correspond to missing values).
         • Print the number (and the proportion) of missing values in each column.
         • Visualize the patterns of missing data (e.g., using a heatmap).
      2. Possible Action Items for Preprocessing Missing Values
         • Impute the missing values based on domain knowledge and the data itself.
         • Remove variables/columns where the missing value ratio exceeds a specific threshold. This threshold should be determined based on the specific data science context.
   3. Techniques for Data Integrity Checks:
      1. Techniques for Identifying Whether the Data is Complete
         • Check that every observational unit appears in the data exactly once.
         • Check that the number of rows in the data matches what you expect, such as from the data documentation. If there are fewer rows than you expect, this might indicate that your data is not complete.
      2. Action Items for Preprocessing Data Integrity:
         • If any observational units are missing from the data, add them to the data and populate unknown entries with missing values.
   4. Common Exploratory Data Analysis (EDA) Methods:
      • Typical summary values for single variables: mean and median
      • Histograms and box plots
      • Measures of variable "spread": variance and standard deviation
      • Linear relationships between two variables: covariance and correlation coefficient
      • Scatter plots
      • Logarithmic scales
      • Correlation coefficient heatmap

3. Based on our review of the data background and domain information, we have analyzed the following information:
   Data Science Project Description: {problem_description}
   Data Context Description: {context_description}
   Observation Unit: {check_unit}
   Variable Descriptions: {var_json}
   Designed Hypotheses: {hyp_json}

Note:
1. During the data cleaning process, do not remove the ID column or generate cleaning operations and code that would remove the ID column.
"""

# 1. Data Loading and Validation
DATA_LOADING_TEMPLATE = """Task: Data Loading and Validation
Task Description: Check if the data is loaded correctly by comparing the first 10 rows with 10 random rows to confirm data consistency.

Input Data:
1. First 10 rows content: {first_10_rows}
2. Random 10 rows content: {random_10_rows}

Output Data:
Confirm whether the data is loaded correctly in text format.
"""

# 2. Data Dimension Check
DIMENSION_CHECK_TEMPLATE = """Task: Data Dimension Check Code Generation
Task Description: Generate Python code to check data dimensions based on the existing scientific project description, data background description, observation unit, and data variable description. Check if the data dimensions meet expectations, if there are any logical issues that violate the data background, and if the actual number of years exceeds the range described in the data background. Please store all analysis results in the variable 'result', which should contain analytical information rather than just text conclusions.

Input Data:
1. CSV file path: {csv_path}
2. Data background information: {context_description}

Output Data:
Python code in the following format:
```python
import pandas as pd
data = pd.read_csv('path')
result = ""
# Code for dimension check
print(result)
```
"""

# 3. Data Dimension Analysis
DIMENSION_ANALYSIS_TEMPLATE = """Task: Data Dimension Analysis
Task Description: Analyze whether the data meets expectations based on Python execution results.

Input Data:
1. Python code execution result: {result}

Output Data:
If problems exist, output a list of problems in JSON format:
```json
[
  {{"problem": "Description of the specific problem."}}
]
```
"""

# 4. Invalid Value Analysis
INVALID_VALUE_TEMPLATE = """Task: Invalid Value Analysis
Task Description: Use LLM to check for invalid or unreasonable values in the data description.

Input Data:
1. Data description (numerical data description): {data_describe}

Note:
1. Available methods for outlier detection: `iqr` | `zscore` | `isolation_forest` | `dbscan` | `mad`
2. If multiple variables are suggested to use the same method for detection, these variables can be described in a single JSON

Output Data:
If invalid or unreasonable values exist, return problems in JSON format. If no problems exist, do not output JSON.
```json
[
  {{"problem": "Describe the specific variables with invalid or unreasonable values, and provide available methods. To maintain data integrity, it's best to adopt a clip strategy"}}
]
```
"""

# 5. Data Cleaning Operations Generation
CLEANING_OPERATIONS_TEMPLATE = """Task: Data Cleaning Operations Generation
Task Description: Generate reasonable data cleaning operations for each problem based on the discovered problem list.

Input Data:
1. Problem list (JSON format): {problem_list}

Output Data:
Cleaning operations in JSON format, including specific cleaning methods for each problem; if some problems don't require cleaning operations, don't generate related JSON.
```json
[
  {{
    "operation_type": "Type of cleaning operation, such as remove, fill, replace, etc.",
    "operation_method": "Specific operation method, such as which values to delete, what values to fill with, etc."
  }}
]
```
"""

# 6. Missing Value Analysis Code Generation
MISSING_VALUE_CODE_TEMPLATE = """Task: Missing Value Analysis Code Generation
Task Description: Generate Python code for in-depth exploratory analysis of missing values. Images will be saved in the 'data_cleaning_plots' folder under the CSV file directory; create the folder if it doesn't exist. Use English for legends and titles. Store analysis results in the result variable. Ensure all data with missing values have analysis results in the result.

Input Data:
1. Missing value ratio: {missing_ratio}
2. CSV file path: {csv_path}

Output Data:
Python code in the following format:
```python
import pandas as pd
import os

# Create directory for saving plots
csv_dir = os.path.dirname('csv_path')
plot_dir = os.path.join(csv_dir, 'data_cleaning_plots')
os.makedirs(plot_dir, exist_ok=True)

data = pd.read_csv('csv_path')
result = ''
missing_data = data.isna().sum() / len(data.index)
result += 'Missing value analysis: ...'
print(result)
```
"""

# 7. Missing Value Cleaning Operations Generation
MISSING_VALUE_CLEANING_TEMPLATE = """Task: Missing Value Cleaning Operations Generation
Task Description: Generate reasonable cleaning operations based on missing value analysis results.

Note:
1. For all variables, specify processing methods (including "no cleaning needed" notes). For variables with missing value ratio below 20%, specify methods for data cleaning.
2. The specific operation methods should comply with the data units.
3. When processing methods are the same, variables can be merged in the variable name field, with a maximum of four different filling methods, i.e., merged into at most 4 groups.
4. Available filling methods: | `mean` | `median` | `mode` | `ffill` | `bfill` | `interpolate` | `constant` | `knn`

Input Data:
1. Missing value ratio: {missing_ratio}
2. Missing value analysis results: {missing_analysis}
3. Data unit: {data_unit}

Output Data:
Cleaning operations in JSON format:
```json
[
  {{
    "variable_name": "variable_name1, variable_name2",
    "operation_type": "delete/impute etc.",
    "operation_method": "Delete variables with high missing ratio/Fill this variable according to data unit with specific method"
  }}
]
```
"""

# 8. Data Integrity Check Code Generation
DATA_INTEGRITY_CHECK_TEMPLATE = """Task: Data Integrity Check Code Generation
Task Description: Generate Python code to check data integrity and store analysis results in the result variable.

Input Data:
1. CSV file path: {csv_path}

Output Data:
Python code in the following format:
```python
import pandas as pd
data = pd.read_csv('csv_path')
result = ''
# Logic for checking data integrity
result += 'Analysis results: ...'
print(result)
```
"""

# 9. Data Integrity Cleaning Operations Generation
DATA_INTEGRITY_CLEANING_TEMPLATE = """Task: Data Integrity Cleaning Operations Generation
Task Description: Generate cleaning operations based on data integrity analysis results. If the analysis results show the data is complete, do not generate JSON.

Input Data:
1. Data integrity analysis results: {integrity_result}

Note:
Do not generate row deletion cleaning operations if the data description does not specify data range

Output Data:
Cleaning operations in JSON format:
```json
[
  {{
    "operation_type": "Fill unknown entries with missing values",
    "operation_method": "If any observational units are missing from the data, add them to the data and populate unknown entries with missing values"
  }}
]
```
"""

# 10. Hypothesis Validation Code Generation
HYPOTHESIS_VALIDATION_CODE_TEMPLATE = """Task: Hypothesis Validation Code Generation
Task Description: Generate Python code to validate hypotheses based on the provided hypothesis and validation method. The code should store analysis results in the result variable.

Input Data:
1. Hypothesis content: {hypothesis}
2. Validation method: {validation_method}
3. CSV file path: {csv_path}

Output Data:
Python code for hypothesis validation in the following format:
```python
import pandas as pd
data = pd.read_csv('csv_path')
result = ''
# Validation logic
result += 'Validation results: ...'
print(result)
```
"""

# 11. Hypothesis Validation Results Analysis
HYPOTHESIS_VALIDATION_ANALYSIS_TEMPLATE = """Task: Hypothesis Validation Results Analysis
Task Description: Based on the local Python code execution results, determine whether the hypothesis holds and generate new hypothesis conclusions.

Input Data:
1. Validation results: {validation_result}

Output Data:
Analysis results in JSON format:
```json
[
  {{
    "hypothesis": "Hypothesis content",
    "validation_method": "Hypothesis validation method",
    "conclusion": "Whether the hypothesis holds or not"
  }}
]
```
"""

# 12. Data Cleaning Function Generation
DATA_CLEANING_FUNCTION_TEMPLATE = """Task: Data Cleaning Function Generation
Task Description: Generate a data cleaning Python function based on the provided cleaning operation list. For data imputation, pay attention to the imputation logic. For example, for country-year data, each variable should be imputed by country. The function should allow customization of tool function parameters for each operation.

Input Data:
1. CSV file path: {csv_path}
2. Data background description: {context_description}
3. Variable descriptions: {var_descriptions}
4. Data unit: {check_unit}
5. Data information: {data_info}
6. Cleaning operations list: {cleaning_operations}

Available Tool Functions:
{tools}

Tool Function Descriptions:
{tool_descriptions}

Output Requirements:
Implement a data processing function with multiple optional parameters, meeting the following requirements:
- Tool functions are imported by default, no need to re-import in your code.
- Each tool function call should store the returned DataFrame in a variable to ensure data processing continuity.
- Function should distinguish between necessary data cleaning steps and optional data preprocessing steps. Data integrity processing should be executed first, followed by data cleaning, and finally data preprocessing.
- Control execution of processing operations through boolean parameters.
- For fill_missing and handle_outliers function calls, allow parameter customization through parameter dictionaries.
- Rules for storing processed data files:
  - Location: 'clean_dataset' subdirectory under the original file directory (create if not exists)
  - Example: Original file '/path/data.csv' -> Processed file '/path/clean_dataset/cleaned_data.csv'
  - Note: Data set saving should be done after function call, not within the function
- Must include optional parameter `remove_with_missing`: Controls whether to remove rows still containing missing values after all data cleaning operations, according to data structure. Default is True. When True, first forward-fill remaining missing values, then delete all rows still containing missing values, ultimately generating a dataset with no missing values.
- Must include optional parameter `convert_categorical`: Controls whether to convert categorical variables to numeric encoding, preferably using category encoding, not saving original columns, no conversion needed for prediction variables.
- Function parameter names should avoid conflicts with tool function parameters.
- Call the function directly at the end of the code without using if __name__ == "__main__", store the dataset in the calling section, and store the dataset path in the result variable.
- Function should return the processed DataFrame.
- Based on operations, if several variables have the same fill_missing or handle_outliers method, combine these variables into variable groups.
- fill_missing and handle_outliers boolean parameters control filling and outlier handling, fill_missing_params and outlier_params are dictionary parameters, set filling methods and thresholds in the calling section based on operation suggestions.
- handle_outliers strategy parameter defaults to clip.
- Do not delete ID type variables if they exist in the dataset.

Format Example:
```python
def clean_data(data_path, 
    [optional operation parameters],
    remove_with_missing=True,
    convert_categorical=True,
    [tool function parameter dictionaries, fill_missing_params=None, outlier_params=None]
  ):
    data = pd.read_csv(data_path)
    
    [write related code based on operations]
    
    # Data cleaning steps
    if fill_missing:
        for method, columns in fill_missing_params.items():
            if isinstance(columns, str):
                columns = [columns]
            params = {{'method': method}}
            data = fill_missing_values_tools(data, target_columns=columns, **params)

    if handle_outliers:
        for group_params in outlier_params:
            method = group_params.get('method', 'iqr')
            columns = group_params.get('columns', [])
            sensitivity = group_params.get('sensitivity', 'medium')
            params = {{'method': method, 'strategy': 'clip', 'sensitivity': sensitivity}}
            data = handle_outliers_tools(data, target_columns=columns, **params)
    
    [handle missing values and numeric conversion of discrete data]
    
    return data

import os
data_path = [csv_file_path]

[Define multiple variable groups group1,group2,group3... by combining similar operations for missing value filling and outlier handling according to operations]
fill_missing_params = [fill default parameters according to operations]
outlier_params = [fill method parameters according to operations, note that this parameter does not include clip]

'''
Example (this part is for reference, do not include in generated code:
group1 = ['GDP', 'CPI', 'per_capita_income']
group2 = ['unemployment_rate', 'inflation_rate']
group3 = ['population']
group = ['GDP', 'CPI']

fill_missing_params = {{
    'bfill': group_linear,
    'ffill': group_ffill,
    'mean': group_mean
}}

outlier_params = [
  {{'method': 'iqr', 'columns': group_outliers}}
]
'''

cleaned_data = clean_data(
    data_path=data_path,
    [optional operation parameters],
    remove_with_missing=True,
    convert_categorical=True,
    [tool function parameter dictionaries, fill_missing_params=None, outlier_params=None]
)

# Save cleaned dataset
output_dir = os.path.join(os.path.dirname(data_path), 'clean_dataset')
os.makedirs(output_dir, exist_ok=True)
[filename should reflect both function parameters and tool function parameters]
output_path = os.path.join(output_dir, [generated filename])
cleaned_data.to_csv(output_path, index=False)
result = output_path
```
"""

# 13. EDA Questions Generation
EDA_QUESTIONS_TEMPLATE = """Task: Exploratory Data Analysis (EDA) Questions Generation
Task Description: Generate specific exploratory data analysis questions based on the cleaned data and data science project description.

Note:
1. Do not propose more than three questions.
2. For prediction problems, questions can involve feature extraction, such as analyzing correlations between all relevant variables and response variables.
3. Since the provided dataset is cleaned, column names in EDA questions may have changed, please ensure to use correct column names.

Input Data:
1. Data science project description: {problem_description}
2. Cleaned data structure information: {data_structure}
3. Cleaned data variable preview: {data_preview}

Output Data:
List of EDA questions in JSON format (including questions and their brief descriptions):
```json
[
  {{
    "question1": "Question 1 description",
    "conclusion": "Pending resolution"
  }}
]
```
"""

# 14. EDA Question Solution Code Generation
EDA_CODE_TEMPLATE = """Task: EDA Question Solution Code Generation
Task Description: Generate Python code to solve the provided EDA question. Code can include analysis, plotting, and other operations. The code should store analysis results and generated image path list in the result variable, and save images in the 'eda_plots' folder under the CSV file directory; create the folder if it doesn't exist. Note that the result variable should contain data-driven descriptions of generated images, not just simple path information. Use os.path.join function to dynamically generate paths, avoid hardcoding paths directly.
Ensure the code generates no more than five images, and use English for legends and titles;
The generated_plots list should only contain complete image path strings, no other information.

Input Data:
1. Cleaned data path: {csv_path}
2. Specific question description: {question}
3. Data structure information: {data_structure}
4. Data variable preview: {data_preview}

Output Data:
Python code, including analysis and visualization logic:
```python
import pandas as pd
import os

# Create directory for saving plots
csv_dir = os.path.dirname('path')
plot_dir = os.path.join(csv_dir, 'eda_plots')
os.makedirs(plot_dir, exist_ok=True)

data = pd.read_csv('path')
result = {{
    'text_result': '',  # Text analysis results, containing specific data descriptions
    'generated_plots': []  # List of image paths, e.g.: ['path/to/plot1.png', 'path/to/plot2.png']
}}
# Analysis and visualization code
print(result)
```
"""

# 15. EDA Analysis Results Feedback
EDA_ANALYSIS_TEMPLATE = """Task: EDA Analysis Results Feedback
Task Description: Update EDA question conclusions based on code execution results and image analysis results.

Input Data:
1. EDA question description: {question}
2. Code execution results: {result}
3. Image analysis results: {image_analysis}

Output Data:
Updated EDA questions and conclusions (JSON format):
```json
[
  {{
    "question": "Original question description",
    "conclusion": "Analysis results and conclusions"
  }}
]
```
"""

# 16. PCS Evaluation Code Generation
PCS_EVALUATION_TEMPLATE = """Task: PCS (Predictability, Computability, Stability) Evaluation Code Generation
Task Description: Generate Python code to evaluate predictability, computability, and stability based on provided conclusions. Code can include analysis, plotting, and other operations. The code should store analysis results and generated image path list in the result variable, and save images in the same directory as the CSV file. Use os.path.join function to dynamically generate paths, avoid hardcoding paths directly.
Ensure the code generates no more than 3 images, and use English for legends and titles;
The generated_plots list should only contain complete image path strings, no other information.

Input Data:
1. Cleaned data path: {csv_path}
2. Conclusions to evaluate: {conclusions}
3. Data structure information: {data_structure}
4. Data variable preview: {data_preview}

Output Data:
Python code, evaluation includes:
- Predictability assessment of conclusions. Predictability refers to the reoccurrence of these data-driven results in new contexts
- Stability of conclusions after data perturbation
- Stability of subjective judgments in visualization generation

Format as follows:
```python
import pandas as pd
result = ''
...
# Evaluation code
result = {{
    'text_result': '',  # Text analysis results, containing specific data descriptions
    'generated_plots': []  # List of image paths, e.g.: ['path/to/plot1.png', 'path/to/plot2.png']
}}
print(result)
```
"""

# 17. Discrete Variable Numerization Assessment
DISCRETE_VAR_CHECK_TEMPLATE = """Task: Discrete Variable Numerization Assessment
Task Description: Determine whether discrete variables in the data need to be numerized based on the provided data path and problem description.

Input Data:
1. Cleaned data path: {csv_path}
2. Problem description: {problem_description}

Output Data:
Assessment results in JSON format:
```json
[
  {{
    "needs_numerization": true
  }}
]
```
"""

# 18. Discrete Variable Numerization Code Generation
DISCRETE_VAR_CODE_TEMPLATE = """Task: Discrete Variable Numerization Code Generation
Task Description: Generate Python code for numerization based on the provided data path and list of discrete variables.

Input Data:
1. CSV path: {csv_path}
2. Discrete variables and their unique values: {discrete_vars}

Output Data:
Python code for numerization processing:
- Code should convert all discrete variables in the data to numeric variables
- Cleaned data should be saved as a new CSV file, with path generated based on the original data path
```python
import pandas as pd
data = pd.read_csv('path')
# Numerization logic
data.to_csv('new_file.csv', index=False)
```
"""

# Data Cleaning Task List Generation
TASK_LIST_TEMPLATE = """Task: Data Cleaning Task List Generation
Task Description: Before executing data cleaning tasks, you need to first generate an ordered list of data cleaning-related tasks based on the provided data science project description, data background description, observation unit, data variable description, and designed hypotheses.

Available Task Types (select only from the following options):
- dimension_analysis_and_problem_list_generation
- invalid_value_analysis_and_problem_list_generation
- missing_value_analysis_and_cleaning_operation_generation
- data_integrity_analysis_and_cleaning_operation_generation

Note:
- Only list task names, no additional description needed
- Strictly select from the above task types
- Each task should only contain the "task_name" field
- Arrange tasks in execution order, the generated task list will guide subsequent data cleaning operations

Input Data:
1. Data science project description: {problem_description}
2. Data background description: {context_description}
3. Observation unit: {check_unit}
4. Data variable description: {var_json}

Output Data:
Please output the task list strictly in the following JSON format:
```json
[
  {{"task_name": "dimension_analysis_and_problem_list_generation"}},
  {{"task_name": "invalid_value_analysis_and_problem_list_generation"}},
  {{"task_name": "missing_value_analysis_and_cleaning_operation_generation"}},
  {{"task_name": "data_integrity_analysis_and_cleaning_operation_generation"}}
]
```
"""

# 19. EDA Summary Generation
EDA_SUMMARY_TEMPLATE = """Task: Generate EDA Analysis Summary
Task Description: Based on the completed EDA questions and their conclusions, generate a comprehensive exploratory data analysis summary report.

Input Data:
1. EDA question and conclusion list: {eda_results}
2. Data science project description: {problem_description}

Output Requirements:
1. The summary should include the following content:
   - Main findings of data exploration
   - Key relationships between variables
   - Implications for subsequent modeling
2. Use professional but easy-to-understand language
3. Ensure the summary is relevant to the project goal
5. Adopt a coherent paragraph format, not a list or JSON format
6. The summary should be a complete narrative text

Output Format Example:
Through data exploration, we found the following key insights: [Main Findings]. In terms of variable relationships, [Describe Important Variable Relationships]. These findings have important implications for subsequent modeling, [Modeling Suggestions].
""" 