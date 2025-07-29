from .base_agent import BaseDSLC_Agent
from prompts.pcs_prompts import *
import json
from tools import ImageToTextTool
import pandas as pd
import re
from langchain.memory import ConversationBufferMemory
import os
import shutil
from typing import List

class PCSAgent(BaseDSLC_Agent):
    def __init__(self, problem_description, context_description, memory=None, llm=None):
        """
        Initialize PCS evaluation agent.
        
        Parameters:
        - problem_description: Data science project description
        - context_description: Data background description
        - var_json: Variable description (JSON format)
        - check_unit: Observation unit
        - memory: Optional memory component
        - llm: Optional language model component
        """
        system_message = PCS_EVALUATION_TEMPLATE.format(
            problem_description=problem_description,
            context_description=context_description
        )
        super().__init__(
            name="PCS Evaluation Agent",
            system_message=system_message,
            memory=memory,
            llm=llm
        )
        self.image_tool = ImageToTextTool(llm=self.llm)
        self.context_description = context_description
        self.problem_description = problem_description
        self.logger.info("PCS Evaluation Agent initialized")
    
    def analyze_image(self, image_path: str, prompt: str = "Please analyze this image") -> str:
        """Analyze image content"""
        return self.image_tool.run(tool_input={
            "image_path": image_path,
            "prompt": prompt
        })
    
    def analyze_pcs_evaluation_result(self, conclusion, result):
        """
        Perform PCS evaluation on conclusions based on code execution results.

        Parameters:
        - conclusion: Original conclusion.
        - result: Code execution result.

        Returns:
        - PCS evaluation report (JSON format).
        """
        self.logger.info("Starting PCS evaluation analysis")
        try:
            result_dict = eval(result)
            image_analysis_results = []
            for plot_path in result_dict.get('generated_plots', []):
                if os.path.exists(plot_path):
                    self.logger.info(f"Analyzing plot: {plot_path}")
                    image_analysis = self.analyze_image(
                    image_path=plot_path,
                    prompt="Please analyze this data visualization chart, describe its main trends, patterns, and key findings. Expected return format: This is an analysis of {chart name}, the main trend is {trend description}, the pattern is {pattern description}, and key findings are {key findings}."
                    )
                    image_analysis_results.append(image_analysis)

            combined_result = {
                'text_result': result_dict.get('text_result', ''),
                'image_analysis': '\n'.join(image_analysis_results)
            }

            input_data = EDA_PCS_EVALUATION_TEMPLATE.format(
                conclusion=conclusion,
                result=combined_result
            )

            response = self.execute(input_data)
            pcs_json = self.parse_llm_json(response)
            
            if pcs_json:
                self.logger.info("Successfully completed PCS evaluation")
                return pcs_json
            else:
                self.logger.warning("Failed to parse PCS evaluation result")
                return {"predictability": "Unable to evaluate", "stability": "Unable to evaluate"}
        except Exception as e:
            self.logger.error(f"Error in PCS evaluation: {str(e)}")
            return {"predictability": "Error in evaluation process", "stability": f"Error message: {str(e)}"}


    def evaluate_problem_definition(self, problem_description, context_description, var_json, unit_check):
        """
        Evaluate problem definition and assumptions based on PCS principles.
        
        Parameters:
        - problem_description: Data science project description
        - context_description: Data background description
        - var_json: Variable description (JSON format)
        - unit_check: Observation unit
        
        Returns:
        - Generated hypothesis list (JSON format)
        - If parsing fails, returns an error message string
        """
        self.logger.info("Starting problem definition evaluation")
        try:
            input_data = HYPOTHESIS_GENERATION_TEMPLATE.format(
                problem_description=problem_description,
                context_description=context_description,
                var_json=json.dumps(var_json, ensure_ascii=False, indent=2),
                unit_check=unit_check
            )

            response = self.execute(input_data)
            parsed_hypothesis = self.parse_llm_json(response)
            
            if parsed_hypothesis:
                self.logger.info("Successfully evaluated problem definition")
                return parsed_hypothesis
            else:
                self.logger.warning("Failed to generate valid hypothesis")
                return "Unable to generate valid hypothesis."
        except Exception as e:
            self.logger.error(f"Error evaluating problem definition: {str(e)}")
            return f"Error generating hypothesis: {str(e)}"

    def generate_stability_analysis_code(self, csv_file_path, cleaning_code):
        """
        Generate data cleaning stability analysis code.

        Parameters:
        - csv_file_path: Original data CSV file path
        - operations: JSON list containing cleaning operation items
        - cleaning_code: Original cleaning function code

        Returns:
        - Generated complete Python code string (including setup code)
        """
        self.logger.info(f"Generating stability analysis code for {csv_file_path}")
        try:
            # Read data information
            data = pd.read_csv(csv_file_path)
            from io import StringIO
            buffer = StringIO()
            data.info(buf=buffer)
            datainfo = buffer.getvalue()

            # Define available tool functions
            tools = [
                'fill_missing_values_tools',
                'remove_columns_tools',
                'handle_outliers_tools',
                'encode_categorical_tools'
            ]

            # Read tool function documentation
            doc_path = 'tools/ml_tools_doc/data_cleaning_tools.md'
            with open(doc_path, 'r', encoding='utf-8') as f:
                tool_descriptions = f.read()

            # Prepare template input data
            input_data = DATA_CLEANING_STABILITY_TEMPLATE.format(
                csv_path=csv_file_path,
                data_info=datainfo,
                cleaning_function=cleaning_code,
                tools=', '.join(tools),
                tool_descriptions=tool_descriptions
            )

            # Generate code
            generated_code, _ = self.chat_with_memory(input_data, ConversationBufferMemory())
            
            # Extract code part
            code_match = re.search(r"```python\n(.*?)\n```", generated_code, re.DOTALL)
            if code_match:
                extracted_code = code_match.group(1)
                
                # Add necessary imports and path settings
                path_setup_code = '''import os,sys,re
current_path = os.path.abspath(__file__)
match = re.search(r'(.*VDSAgent)', current_path)
if not match:
    raise FileNotFoundError("Could not find VDSAgent directory")
sys.path.append(match.group(1))
from tools.ml_tools import *

'''
                final_code = path_setup_code + extracted_code
                self.logger.info("Successfully generated stability analysis code")
                return f"```python\n{final_code}\n```"
            else:
                self.logger.warning("Failed to extract Python code from generated text")
                return "Unable to generate valid Python code."
        except Exception as e:
            self.logger.error(f"Error generating stability analysis code: {str(e)}")
            return f"Error generating code: {str(e)}"

    def execute_stability_analysis(self, csv_file_path: str,  cleaning_code: str):
        """
        Execute data cleaning stability analysis, generate multiple data set versions, and validate them.

        Args:
            csv_file_path: Original data CSV file path
            cleaning_code: Original cleaning function code

        Returns:
            Returns a list of validated data set paths
        """
        max_retries = 8
        retry_count = 0
        self.logger.info(f"Starting stability analysis for {csv_file_path}")
        
        # Set directory paths
        csv_dir = os.path.dirname(csv_file_path)
        stability_dir = os.path.join(csv_dir, 'stability_analysis')
        code_dir = os.path.join(csv_dir, 'code')  # Code is saved in the same directory as the original dataset under a code folder
        
        while retry_count < max_retries:
            self.logger.info(f"Attempt {retry_count + 1} of {max_retries}")
            try:
                if os.path.exists(stability_dir):
                    shutil.rmtree(stability_dir)
                os.makedirs(stability_dir, exist_ok=True)
                
                os.makedirs(code_dir, exist_ok=True)
                
                generated_code = self.generate_stability_analysis_code(
                    csv_file_path=csv_file_path,
                    cleaning_code=cleaning_code,
                )
                
                if "Unable to generate" in generated_code:
                    retry_count += 1
                    self.logger.warning(f"Failed to generate code, attempt {retry_count}")
                    continue
                
                # Set code save path
                py_filename = os.path.basename(csv_file_path).replace('.csv', '_stability.py')
                py_file_path = os.path.join(code_dir, py_filename)
                
                # Execute code and get generated data set list
                self.logger.info("Executing stability analysis code")
                result = self.execute_generated_code(generated_code, save_path=py_file_path)
                
                # Validate result format
                stability_dir = os.path.join(os.path.dirname(csv_file_path), 'stability_analysis')
                if not os.path.exists(stability_dir):
                    if os.path.exists(py_file_path):
                        os.remove(py_file_path)

                    if os.path.exists(stability_dir):
                        shutil.rmtree(stability_dir)
                    retry_count += 1
                    self.logger.warning("Stability directory not created, retrying")
                    continue
                
                # Get all generated CSV files
                generated_datasets = [
                    os.path.join(stability_dir, f) 
                    for f in os.listdir(stability_dir) 
                    if f.endswith('.csv')
                ]
                
                if not generated_datasets:
                    if os.path.exists(py_file_path):
                        os.remove(py_file_path)
                    retry_count += 1

                    if os.path.exists(stability_dir):
                        shutil.rmtree(stability_dir)
                    self.logger.warning("No datasets generated, retrying")
                    continue

                # If dataset exceeds 100, randomly select 100 to keep
                if len(generated_datasets) > 100:
                    import random
                    datasets_to_keep = random.sample(generated_datasets, 100)
                    for dataset_path in generated_datasets:
                        if dataset_path not in datasets_to_keep and os.path.exists(dataset_path):
                            os.remove(dataset_path)
                    generated_datasets = datasets_to_keep
                    self.logger.info("Reduced initial number of datasets to 100")
                
                # Remove duplicate datasets before unit tests
                self.logger.info("Removing duplicate datasets")
                i = 0
                while i < len(generated_datasets):
                    if not os.path.exists(generated_datasets[i]):
                        i += 1
                        continue
                        
                    try:
                        df1 = pd.read_csv(generated_datasets[i])
                        
                        # Check if dataset has only one row
                        if len(df1) <= 1:
                            os.remove(generated_datasets[i])
                            generated_datasets.pop(i)
                            self.logger.warning(f"Removed single-row dataset: {generated_datasets[i]}")
                            continue
                            
                        df1_sorted = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
                        
                        # Compare with subsequent datasets
                        j = i + 1
                        while j < len(generated_datasets):
                            if not os.path.exists(generated_datasets[j]):
                                j += 1
                                continue
                                
                            try:
                                df2 = pd.read_csv(generated_datasets[j])
                                
                                # Check if second dataset has only one row
                                if len(df2) <= 1:
                                    os.remove(generated_datasets[j])
                                    generated_datasets.pop(j)
                                    self.logger.warning(f"Removed single-row dataset: {generated_datasets[j]}")
                                    continue
                                    
                                df2_sorted = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)
                                
                                # If content is the same (ignoring order), delete the second dataset
                                if df1_sorted.equals(df2_sorted):
                                    os.remove(generated_datasets[j])
                                    generated_datasets.pop(j)
                                    self.logger.info(f"Removed duplicate dataset: {generated_datasets[j]}")
                                else:
                                    j += 1
                            except Exception as e:
                                self.logger.error(f"Error comparing datasets: {str(e)}")
                                j += 1
                                
                    except Exception as e:
                        self.logger.error(f"Error processing dataset {generated_datasets[i]}: {str(e)}")
                        i += 1
                        continue
                    
                    i += 1
                
                # Run unit tests on each dataset
                self.logger.info("Running unit tests on generated datasets")
                valid_datasets = []
                for dataset_path in generated_datasets:
                    passed, report = self.run_unit_tests(
                        cleaned_data_path=dataset_path,
                        original_data_path=csv_file_path
                    )
                    
                    if passed:
                        valid_datasets.append(dataset_path)
                        self.logger.info(f"Dataset passed unit tests: {dataset_path}")
                    else:
                        if os.path.exists(dataset_path):
                            os.remove(dataset_path)
                            self.logger.warning(f"Removed dataset that failed unit tests: {dataset_path}")
                
                # Check number of datasets
                num_datasets = len(valid_datasets)
                self.logger.info(f"Generated {num_datasets} valid datasets")
                
                if num_datasets < 3:
                    # Too few datasets, retry
                    for dataset_path in valid_datasets:
                        if os.path.exists(dataset_path):
                            os.remove(dataset_path)
                    if os.path.exists(py_file_path):
                        os.remove(py_file_path)
                    retry_count += 1
                    self.logger.warning("Too few valid datasets, retrying")
                    continue
                    
                elif num_datasets > 50:
                    # Randomly select 50 datasets to keep
                    import random
                    datasets_to_keep = random.sample(valid_datasets, 50)
                    for dataset_path in valid_datasets:
                        if dataset_path not in datasets_to_keep and os.path.exists(dataset_path):
                            os.remove(dataset_path)
                    valid_datasets = datasets_to_keep
                    self.logger.info("Reduced number of datasets to 50")
                
                # Return directly the list of validated datasets
                self.logger.info("Stability analysis completed successfully")
                return valid_datasets
                
            except Exception as e:
                # Clean up files on error
                if 'py_file_path' in locals() and os.path.exists(py_file_path):
                    os.remove(py_file_path)
                if os.path.exists(stability_dir):
                    shutil.rmtree(stability_dir)
                
                retry_count += 1
                self.logger.error(f"Error during stability analysis: {str(e)}")
                if retry_count == max_retries:
                    return f"Error after {max_retries} attempts: {str(e)}"
                continue
        
        self.logger.error(f"Failed to complete stability analysis after {max_retries} attempts")
        return f"Unable to generate valid stability analysis results after {max_retries} attempts."