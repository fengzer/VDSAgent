from .base_agent import BaseDSLC_Agent
import json
import pandas as pd
import os
import re
from langchain.memory import ConversationBufferMemory
from prompts.prediction_inference_prompts import *
from tools import *

class PredictionAndInferenceAgent(BaseDSLC_Agent):
    def __init__(self, problem_description, context_description, eda_summary, memory=None, llm=None):
        self.problem_description = problem_description
        self.context_description = context_description
        self.eda_summary = eda_summary
        self.response_variable = None

        system_message_content = PREDICTION_INFERENCE_TEMPLATE.format(
            problem_description=problem_description,
            context_description=context_description,
            eda_summary=eda_summary 
        )

        super().__init__(
            name="Prediction and Inference",
            system_message=system_message_content,
            memory=memory,
            llm=llm
        )
        self.logger.info("PredictionAndInferenceAgent initialized")

    def suggest_modeling_methods(self):
        """
        Generate up to three modeling methods from best to worst based on problem description, background information, and updated problems.
        
        Returns:
        - JSON format list containing up to three modeling methods.
        - If generation fails, returns error message string.
        """
        self.logger.info("Generating modeling method suggestions")
        try:
            input_data = MODELING_METHODS_TEMPLATE.format(
                problem_description=self.problem_description,
                context_description=self.context_description,
                eda_summary=self.eda_summary
            )

            response = self.execute(input_data)
            modeling_methods = self.parse_llm_json(response)
            
            if modeling_methods:
                self.logger.info(f"Successfully generated {len(modeling_methods)} modeling methods")
                return modeling_methods
            else:
                self.logger.warning("Failed to generate valid modeling methods list")
                return "Failed to generate valid modeling methods list."
        except Exception as e:
            self.logger.error(f"Error suggesting modeling methods: {str(e)}")
            return f"Error during execution: {str(e)}"
        

    def suggest_feature_engineering_methods(self, data_path: str):
        """
        Generate up to three feature engineering method suggestions based on problem description, background information, and dataset column names.
        
        Parameters:
        - data_path: Dataset file path
        
        Returns:
        - JSON format list containing up to three feature engineering methods.
        - If generation fails, returns error message string.
        """
        self.logger.info(f"Generating feature engineering method suggestions for {data_path}")
        try:
            # Read dataset to get column names
            df = pd.read_csv(data_path)
            column_names = list(df.columns)
            
            input_data = FEATURE_ENGINEERING_TEMPLATE.format(
                problem_description=self.problem_description,
                context_description=self.context_description,
                column_names=", ".join(column_names)  # Convert column names list to string
            )

            response = self.execute(input_data)
            feature_engineering_methods = self.parse_llm_json(response)
            
            if feature_engineering_methods:
                self.logger.info(f"Successfully generated {len(feature_engineering_methods)} feature engineering methods")
                return feature_engineering_methods
            else:
                self.logger.warning("Failed to generate valid feature engineering methods list")
                return "Failed to generate valid feature engineering methods list."
        except Exception as e:
            self.logger.error(f"Error suggesting feature engineering methods: {str(e)}")
            return f"Error during execution: {str(e)}"

    def identify_response_variable(self, data_path):
        """
        Analyze and identify response variables based on problem description, background information, and dataset column names.
        
        Returns:
        - JSON format dictionary containing response variable information.
        - If identification fails, returns error message string.
        """
        self.logger.info(f"Identifying response variable from {data_path}")
        try:
            # Read dataset to get column names
            df = pd.read_csv(data_path)
            column_names = list(df.columns)
            
            input_data = RESPONSE_VARIABLE_TEMPLATE.format(
                problem_description=self.problem_description,
                context_description=self.context_description,
                column_names=column_names
            )

            response = self.execute(input_data)
            response_var_list = self.parse_llm_json(response)
            
            # Validate if response variables are in dataset column names
            if isinstance(response_var_list, list):
                valid_responses = []
                for var_info in response_var_list:
                    if var_info and var_info["Response Variable"] in column_names:
                        column = df[var_info["Response Variable"]]
                        
                        if var_info["Variable Type"] == "Discrete":
                            # For discrete variables, store all possible values
                            unique_values = column.unique().tolist()
                            # Convert to regular Python numeric types if numeric
                            unique_values = [item.item() if hasattr(item, 'item') else item for item in unique_values]
                            var_info["Possible Values"] = sorted(unique_values)
                        else:  # Continuous
                            # For continuous variables, store min and max values
                            var_info["Value Range"] = {
                                "Minimum": float(column.min()),
                                "Maximum": float(column.max())
                            }
                        
                        valid_responses.append(var_info)
                        self.logger.info(f"Successfully identified response variable: {var_info}")
                
                if valid_responses:
                    self.response_variable = json.dumps(valid_responses, ensure_ascii=False, indent=2)
                    return valid_responses
                else:
                    self.logger.warning("No valid response variables found in dataset")
                    return "No valid response variables identified, or identified response variables not in dataset."
            else:
                self.logger.warning("Response is not a list format")
                return "Response format error, should be list format."
        except Exception as e:
            self.logger.error(f"Error identifying response variable: {str(e)}")
            return f"Error identifying response variable: {str(e)}"

    def generate_combined_model_code(self, csv_path, model_methods, feature_engineering_methods=None):
        """
        Generate a unified training and evaluation code file for multiple models.

        Parameters:
        - csv_path: File path containing cleaned data.
        - model_methods: List of model methods, each containing method and description information.
        - feature_engineering_methods: List of feature engineering methods, each containing name and description.

        Returns:
        - Generated complete Python code string.
        - If generation fails, returns error message string.
        """
        self.logger.info(f"Generating combined model code for {len(model_methods)} models")
        try:
            data = pd.read_csv(csv_path)
            
            # Define available feature engineering tool functions
            tools = [
                'transform_features',
                'reduce_dimensions',
                'select_features',
                'discretize_features',
                'create_polynomial_features'
            ]

            # Read tool function documentation
            doc_path = 'tools/ml_tools_doc/feature_engineering_tools.md'
            with open(doc_path, 'r', encoding='utf-8') as f:
                tool_descriptions = f.read()
            
            # Build input containing all model information
            models_json = json.dumps(model_methods, ensure_ascii=False, indent=2)
            
            input_data = COMBINED_MODEL_CODE_TEMPLATE.format(
                models=models_json,
                feature_engineering_methods=json.dumps(feature_engineering_methods, ensure_ascii=False, indent=2) if feature_engineering_methods else "[]",
                problem_description=self.problem_description,
                context_description=self.context_description,
                csv_path=csv_path,
                csv_columns=', '.join(data.columns),
                tools=json.dumps(tools, ensure_ascii=False, indent=2),
                tool_descriptions=tool_descriptions,
                response_variable=self.response_variable
            )

            generated_code, _ = self.chat_with_memory(input_data, ConversationBufferMemory())
            code_match = re.search(r"```python\n(.*?)\n```", generated_code, re.DOTALL)
            
            if code_match:
                extracted_code = code_match.group(1)
                
                # Add path setup code
                path_setup_code = '''import os,sys,re
current_path = os.path.abspath(__file__)
match = re.search(r'(.*VDSAgent)', current_path)
if not match:
    raise FileNotFoundError("Could not find VDSAgent directory")
sys.path.append(match.group(1))
from tools.ml_tools import *

'''
                final_code = path_setup_code + extracted_code
                self.logger.info("Successfully generated combined model code")
                return f"```python\n{final_code}\n```"
            self.logger.warning("Failed to extract Python code from generated text")
            return "No code generated."
        except Exception as e:
            self.logger.error(f"Error generating combined model code: {str(e)}")
            return f"Error during code generation: {str(e)}"

    def train_and_evaluate_combined_models(self, csv_path, model_methods, feature_engineering_methods=None, max_attempts=5):
        """
        Generate and execute unified code containing all models.

        Parameters:
        - csv_path: File path containing cleaned data.
        - model_methods: List of model methods, each containing method and description information.
        - feature_engineering_methods: List of feature engineering methods, each containing name and description.
        - max_attempts: Maximum number of code generation attempts

        Returns:
        - JSON list containing all model training and evaluation results.
        """
        self.logger.info(f"Starting combined model training and evaluation for {csv_path}")
        data_dir = os.path.dirname(os.path.dirname(csv_path))
        code_dir = os.path.join(data_dir, 'code')
        os.makedirs(code_dir, exist_ok=True)

        code_file = os.path.join(code_dir, "train_models.py")

        for attempt in range(max_attempts):
            try:
                self.logger.info(f"Attempt {attempt + 1}/{max_attempts} to generate and execute code")
                self.logger.info("Generating combined model code")
                code = self.generate_combined_model_code(
                    csv_path=csv_path,
                    model_methods=model_methods,
                    feature_engineering_methods=feature_engineering_methods
                )
                
                if "No code generated" in code:
                    self.logger.warning("Failed to generate combined model code")
                    continue

                self.logger.info("Executing combined model code")
                result = self.execute_generated_code(code, save_path=code_file)
                if "Code execution failed, maximum retries reached." in result:
                    self.logger.warning(f"Code execution failed after max retries, attempting regeneration")
                    continue  # Try regenerating code
                
                self.logger.info("Successfully executed combined model code")
                return {"Result": result}
                
            except Exception as e:
                self.logger.error(f"Error in attempt {attempt + 1}: {str(e)}")
                if attempt == max_attempts - 1:
                    return {"Result": f"Still failed after {max_attempts} attempts: {str(e)}"}
                continue  # Try next attempt

        return {"Result": f"Failed to generate and execute code successfully after {max_attempts} attempts."}

    def generate_batch_evaluation_code(self, datasets_dir, model_code_path):
        """
        Generate code for batch evaluation of datasets.
        
        Parameters:
        - datasets_dir: Directory path containing multiple datasets
        - model_code_path: Path to existing model training code file
        
        Returns:
        - Generated Python code string
        """
        self.logger.info(f"Generating batch evaluation code for {datasets_dir}")
        try:
            # Read model training code
            with open(model_code_path, 'r', encoding='utf-8') as f:
                model_code = f.read()
            
            input_data = BATCH_EVALUATION_TEMPLATE.format(
                datasets_dir=datasets_dir,
                model_code=model_code,
                problem_description=self.problem_description,
                context_description=self.context_description,
            )
            
            generated_code, _ = self.chat_with_memory(input_data, ConversationBufferMemory())
            code_match = re.search(r"```python\n(.*?)\n```", generated_code, re.DOTALL)
            
            if code_match:
                extracted_code = code_match.group(1)
                path_setup_code = '''import os,sys,re
current_path = os.path.abspath(__file__)
match = re.search(r'(.*VDSAgent)', current_path)
if not match:
    raise FileNotFoundError("Could not find VDSAgent directory")
sys.path.append(match.group(1))
from tools.ml_tools import transform_features,reduce_dimensions,select_features,discretize_features,create_polynomial_features

'''
                final_code = path_setup_code + extracted_code
                self.logger.info("Successfully generated batch evaluation code")
                return f"```python\n{final_code}\n```"
            self.logger.warning("Failed to extract Python code from generated text")
            return "No code generated."
        except FileNotFoundError:
            self.logger.error(f"Model code file not found: {model_code_path}")
            return f"Model code file not found: {model_code_path}"
        except Exception as e:
            self.logger.error(f"Error generating batch evaluation code: {str(e)}")
            return f"Error during code generation: {str(e)}"


    def execute_batch_evaluation(self, datasets_dir, model_code_path, max_attempts=5):
        """
        Generate and execute batch evaluation code.
        
        Parameters:
        - datasets_dir: Directory path containing multiple datasets
        - model_code_path: Path to existing model training code file
        - max_attempts: Maximum number of code generation attempts
        
        Returns:
        - Evaluation results dictionary
        """
        self.logger.info(f"Starting batch evaluation for {datasets_dir}")
        
        code_dir = os.path.dirname(model_code_path)
        code_file = os.path.join(code_dir, "batch_evaluation.py")

        for attempt in range(max_attempts):
            try:
                self.logger.info(f"Attempt {attempt + 1}/{max_attempts} to generate and execute code")
                
                # Generate batch evaluation code
                code = self.generate_batch_evaluation_code(
                    datasets_dir=datasets_dir,
                    model_code_path=model_code_path
                )
                
                if "No code generated" in code :
                    self.logger.warning("Failed to generate batch evaluation code")
                    continue
                
                # Execute code and get results
                self.logger.info("Executing batch evaluation code")
                result = self.execute_generated_code(code, save_path=code_file)
                
                # Simple validation: check if contains 5 results
                if "'dataset':" in result and result.count("'dataset':") >= 5:
                    self.logger.info("Successfully executed batch evaluation code with 5 or more results")
                    return result
                else:
                    self.logger.warning(f"Result contains fewer than 5 datasets, retrying...")
                    continue
                
            except Exception as e:
                self.logger.error(f"Error in attempt {attempt + 1}: {str(e)}")
                if attempt == max_attempts - 1:
                    return {"Result": f"Still failed after {max_attempts} attempts: {str(e)}"}
                continue

        return {"Result": f"Failed to generate results containing 5 datasets after {max_attempts} attempts."}

    def summarize_evaluation_results(self, results: dict, csv_path: str = None) -> str:
        """
        Convert model evaluation results to readable text format and optionally save to file
        
        Args:
            results: Dictionary containing evaluation results
            csv_path: Original CSV file path, used to determine where to save the summary
            
        Returns:
            str: Formatted results summary text
        """
        self.logger.info("Generating evaluation results summary")
        try:
            input_data = RESULT_SUMMARY_TEMPLATE.format(
                results=json.dumps(results, ensure_ascii=False, indent=2),
            )
            
            summary, _ = self.chat_with_memory(input_data, ConversationBufferMemory())
            code_match = re.search(r"```markdown\n(.*?)\n```", summary, re.DOTALL)
            if code_match:
                summary = code_match.group(1).strip()
            self.logger.info("Successfully generated evaluation summary")
            
            # If CSV path provided, save summary to same directory
            if csv_path:
                try:
                    # Get CSV directory
                    output_dir = os.path.dirname(csv_path)
                    # Create summary filename
                    summary_filename = "model_evaluation_summary.md"
                    summary_path = os.path.join(output_dir, summary_filename)
                    
                    # Save summary to file
                    with open(summary_path, 'w', encoding='utf-8') as f:
                        f.write(summary)
                    
                    self.logger.info(f"Evaluation summary saved to: {summary_path}")
                except Exception as e:
                    self.logger.error(f"Error saving summary to file: {str(e)}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating evaluation summary: {str(e)}")
            return f"Error generating evaluation summary: {str(e)}"