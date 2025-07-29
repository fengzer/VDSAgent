PROBLEM_DEFINITION_TEMPLATE = """You are a data science expert, focusing on reviewing data context and domain information.
"""

VARIABLE_ANALYSIS_TEMPLATE = """First, please analyze the variables in my data.
Data Science Project: {problem_description}
Data Context Description: {context_description}
Variable Preview: {variable_info}
First Five Rows: {data_preview}
"""

VARIABLE_ANALYSIS_TEMPLATE = """Please analyze the variables in your data. Analyze each variable to understand its meaning.
Output format: Output the name and description of each variable in JSON format, as follows:
```json
[
  {{"name": "Variable Name", "description": "Variable Description"}}
]
```

Data Science Project: {problem_description}
Data Context Description: {context_description}
Variable Preview: {variable_info}
First Five Rows: {data_preview}
"""

OBSERVATION_UNIT_TEMPLATE = """Please detect the observation units in your data. Observation units correspond to the entities from which measurements are collected, such as people, countries, or years. Identify the response variable that needs to be predicted, which is the target variable that needs to be predicted in the prediction problem.
Please describe the observation units in text format."""

VARIABLE_RELEVANCE_TEMPLATE = """Please determine the variable relevance, evaluate whether each variable is relevant to the current data science project.
Variable Description: {variable_descriptions}
Please provide a conclusion in text format, explaining whether each variable supports the project goal."""