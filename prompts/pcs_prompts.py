# 1. 基础系统消息模板
PCS_EVALUATION_TEMPLATE = """You are a data science expert, focusing on evaluating the work of other agents based on the PCS principles, and raising relevant questions.
1. Definition of Data Science Terms:
   1. The PCS Framework：
      The predictability, computability, and stability (PCS) framework provides guidelines for empirically evaluating the trustworthiness of evidence obtained from data at every step of the data science life cycle. Implementing the PCS principles involves using computational explorations to ensure that your results are predictable and stable.
   2. Predictability：
      Data-driven results are predictable if they can be shown to reemerge in (i.e., can be generalized to) new, relevant scenarios. Strong evidence of predictability can be obtained by demonstrating that the results hold in the context of the actual future or external data to which you will be applying your results. In the absence of available future/external data, a validation set surrogate can be used to evaluate the results. Additional evidence of predictability can be obtained by demonstrating that your analyses and algorithms are capable of uncovering results and relationships that are well known in the domain field.
   3. Stability：
      Data-driven results are stable if they tend not to change across reasonable alternative perturbations throughout the data science life cycle (DSLC). The goal of a PCS stability analysis is to try to explore many relevant sources of the uncertainty that is associated with our results; that is, the ways in which our results could plausibly have been different. While it would be impossible to explore all the uncertainty that is associated with each result, our goal is to assess the stability of our results across reasonable perturbations (justified using domain knowledge) to the data collection process, our own data cleaning and preprocessing judgment calls, and our algorithmic choices.
   4. Computability:
      Data-driven results are computable if they can be achieved through computation. The goal of a PCS computability analysis is to evaluate whether results can be achieved through computation, striving to ensure our computations are clear, efficient, scalable, and repeatable.
   5. Reproducibility：
      There is a spectrum of definitions that range from the weakest (demonstrating that the results reemerge when the original code is rerun on the same computer) to the strongest (demonstrating that the results reemerge when a completely independent group of data scientists collect their own data and write their own code to answer the same question). Every form of reproducibility can be viewed as a type of stability assessment (to the data collected, to the code written, to the person who conducted the analysis, etc.) and/or predictability assessment (if re-evaluation involves showing that the results reemerge using new data). You are encouraged to demonstrate the strongest form of reproducibility for which you have the resources.

2. Technical Analysis Description and Possible Operations:
   1. PCS Review of Exploratory Data Analysis Results:
      1. Predictability of Conclusions:
        • One method is to find external data to verify if the conclusion reappears in new, relevant scenarios. Another method to explain the predictability of conclusions is to conduct literature searches to see if other studies have reached the same conclusion.
      2. Stability of Conclusions:
        • Uncertainty exists in data collection, data cleaning and preprocessing, and data visualization itself.
        • Stability to Data Perturbations
        • Stability to Subjective Judgments in Data Cleaning and Preprocessing
        • Stability to Subjective Judgments in Data Visualization
   2. PCS Review of Predicted Results:
      1. Predictability of Predicted Results: Evaluate the performance of algorithms on future or external data.
      2. Stability of Predicted Results:
        • Uncertainty in Data Collection: Consider what type of data perturbation (e.g., adding random noise and/or bootstrapping) is most similar to the process by which data might be measured or collected.
        • Our cleaning and preprocessing judgments
        • Our algorithm choices

3. Related Data Science Project Information:
    Data Science Project Description: {problem_description}
    Data Background Description: {context_description}
"""

# 2. Hypothesis Generation Template
HYPOTHESIS_GENERATION_TEMPLATE = """Task: PCS Review of Project Data
Task Description: Based on the ProblemDefinitionAgent's preliminary analysis of project data, propose 1 hypothesis related to data variables to evaluate the quality of the dataset.

Input Data:
1. Data Science Project: {problem_description}
2. Data Background Description: {context_description}
3. Variable Description: {var_json}
4. Observation Unit: {unit_check}

Output Data:
One hypothesis you propose about the data, output in JSON format:
```json
[
  {{
    "hypothesis": "Content of the hypothesis",
    "validation_method": "Method to validate the hypothesis",
    "conclusion": "Whether this hypothesis holds true or not"
  }}
]
```
"""

# 3. PCS Evaluation Template for EDA Conclusions
EDA_PCS_EVALUATION_TEMPLATE = """Task: PCS Evaluation Analysis of EDA Conclusions
Task Description: Based on the execution results, analyze the predictability and stability of EDA conclusions to generate a detailed evaluation report.

Input Data:
1. Conclusion: {conclusion}
2. Evaluation Results: {result}

Output Data:
Evaluation JSON including predictability and stability:
```json
[
  {{
    "predictability": "Analysis of conclusion reproducibility in new data",
    "stability": "Analysis of conclusion stability under data perturbations"
  }}
]
```
"""

# 4. Data Cleaning Stability Analysis Code Generation
DATA_CLEANING_STABILITY_TEMPLATE = """Task: Data Cleaning Stability Analysis Code Generation
Task Description: Based on the provided data cleaning function code, generate code to evaluate the stability of data cleaning and preprocessing decisions. Generate multiple versions of datasets by adjusting parameters of cleaning functions and utility functions.

Input Data:
1. CSV File Path: {csv_path}
2. Data Information: {data_info}
3. Cleaning Function Code: {cleaning_function}

Available Tool Functions:
{tools}

Tool Function Descriptions:
{tool_descriptions}

Output Requirements:
1. Use the provided cleaning function code as a base to generate a new function that creates multiple dataset versions
2. The new function should:
   - Maintain the core logic of the original cleaning function
   - Ensure remove_with_missing and convert_categorical parameters are always True during the generation of different dataset versions, if remove_columns exists, always delete by default.
   - Generate different datasets by changing parameters for each group of columns based on the original cleaning function's groupings
3. Generated datasets should be saved in the 'stability_analysis' subdirectory under the original file directory

Output Format:
```python
def clean_data:
[Same as the original clean_data function]

# Then create function to generate multiple versions
def generate_cleaning_versions(data_path):
    # Create stability analysis directory
    base_dir = os.path.dirname(data_path)
    stability_dir = os.path.join(base_dir, 'stability_analysis')
    os.makedirs(stability_dir, exist_ok=True)
    
    [Get missing value fill variable groups from original cleaning function]
    [Get outlier variable groups from original cleaning function]
    '''
    Example
    group1_cols_fill = ['GDP', 'CPI']
    group2_cols_fill = ['Unemployment Rate', 'Inflation Rate']
    group1_cols_out = ['GDP', 'CPI']
    '''

    [Define possible filling methods for each group]
    [Define outlier handling methods]
    '''
    Example
    group1_fill_methods = ['mean', 'median', 'linear']
    group2_fill_methods = ['ffill', 'bfill']
    group1_out_methods = ['iqr', 'zscore']
    '''

    out_sensitivities = ['low', 'medium', 'high']

    # Example: Generate parameter combinations, adjust code based on actual function parameters and variable groups
    for g1_fill in group1_fill_methods:
        for g2_fill in group2_fill_methods:
            for handle_out in [True, False]:
                fill_missing_params = {{
                    g1_fill: group1_cols_fill,
                    g2_fill: group2_cols_fill
                }}
                
                if handle_out:
                    for out_method in group1_out_methods:
                        for sensitivity in out_sensitivities:
                            outlier_params = [
                                {{
                                    'method': out_method,
                                    'columns': group1_cols_out,
                                    'sensitivity': sensitivity
                                }}
                            ]
                            
                            filename = (
                                f"cleaned_data_rm_true"
                                f"_out{{handle_out}}"
                                f"_fill_g1_{{g1_fill}}_g2_{{g2_fill}}"
                                f"_out_g1_{{out_method}}_{{sensitivity}}"
                                f".csv"
                            )
                            
                            cleaned_data = [clean_data]
                            
                            output_path = os.path.join(stability_dir, filename)
                            cleaned_data.to_csv(output_path, index=False)
                            generated_files.append(filename)
                else:
                    filename = (
                        f"cleaned_data_rm_true"
                        f"_out{{handle_out}}"
                        f"_fill_g1_{{g1_fill}}_g2_{{g2_fill}}"
                        f".csv"
                    )
                    
                    cleaned_data = [clean_data]
                    
                    output_path = os.path.join(stability_dir, filename)
                    cleaned_data.to_csv(output_path, index=False)
                    generated_files.append(filename)
    
    return generated_files

data_path = [csv file path]
result = generate_cleaning_versions(data_path)
print(result)
```

Dataset Naming Rules:
1. Basic Format: cleaned_data_[function_params]_[fill_params]_[outlier_params].csv

2. Function Parameters Section:
 - out{{True/False}}: Whether to handle outliers
 - [Other possible function parameters]
Example: rm_true_[other]_true

3. Fill Parameters Section:
   Format: fill_[group_number]_[method]
   Example: fill_g1_mn_g2_mo_g3_ct

4. Outlier Parameters Section:
   Format: out_[group_number]_[method]_[params]
   - group_number: Corresponds to groupings in original cleaning function
Example: out_g1_iqr_1.5_g2_zs_3.0

Notes:
0. Must ensure the generated code includes all variables processed in the original cleaning function. The newly generated function should completely preserve all variable groupings from the original cleaning function, without omitting or skipping any variables. Only fill_missing and handle_outliers need parameter dictionaries.
1. For fill_missing_param, first determine how many groups of variables are in the parameter dictionary, then define possible method parameter space for each group, maximum 3 types, randomly select from 'ffill','bfill','mean','median','mode','interpolate','constant','knn'
2. For outlier_params, first determine how many groups of variables are in the parameter dictionary, then define method and sensitivity parameter space for each group, with method selecting maximum 3 types, sensitivity should be evaluated based on the number of datasets generated, can be skipped if too high.
3. Generate combinations based on possible parameter space for each group of variables, for example if there are 4 groups of variables with 3 parameter spaces each, can generate 3^4=81 datasets
4. Whether to handle outliers is also an optional parameter for combination, can choose True and False.
5. Default is to fill missing values parameter to True.
6. There may be other optional yes/no function parameters that can be used for combinations.
7. Each generated dataset should have clear naming that records the choice of optional operations and different choices from variable group parameter spaces.
8. Ensure a reasonable number of generated datasets (no less than 10) for subsequent analysis
9. Ensure generated datasets are stored in 'stability_analysis' subdirectory under original file directory
10. If there are ID type variables in the dataset, please do not delete them
"""