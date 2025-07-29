# agents/results_evaluation_agent.py
from .base_agent import BaseDSLC_Agent
from langchain.memory import ConversationBufferMemory
from prompts.results_evaluation_prompts import *
import re
import os

class ResultsEvaluationAgent(BaseDSLC_Agent):
    def __init__(self, problem_description, context_description, best_five_result, memory=None, llm=None):
        self.problem_description = problem_description
        self.context_description = context_description
        self.best_five_result = best_five_result
        
        system_message_content = RESULTS_EVALUATION_TEMPLATE.format(
            problem_description=problem_description,
            context_description=context_description,
            best_five_result=best_five_result
        )
        
        super().__init__(
            name="Results Evaluation",
            system_message=system_message_content,
            memory=memory,
            llm=llm
        )
        self.logger.info("ResultsEvaluationAgent initialized")

    def generate_test_datasets_code(self, multiple_datasets_code_path, original_dataset_path):
        """
        Generate code to create only the five best-fitting datasets based on the code for generating multiple datasets.
        
        Parameters:
        - multiple_datasets_code_path: Code file path for stability datasets
        - original_dataset_path: Original dataset path
        
        Returns:
        - Generated Python code string for creating five best-fitting datasets
        """
        self.logger.info(f"Generating code for test datasets based on {multiple_datasets_code_path}")
        try:
            # Read code for generating multiple datasets
            with open(multiple_datasets_code_path, 'r', encoding='utf-8') as f:
                multiple_datasets_code = f.read()
            
            input_data = BEST_FIVE_DATASETS_TEMPLATE.format(
                original_dataset_path=original_dataset_path,
                multiple_datasets_code=multiple_datasets_code
            )
            
            generated_code, _ = self.chat_with_memory(input_data, ConversationBufferMemory())
            code_match = re.search(r"```python\n(.*?)\n```", generated_code, re.DOTALL)
            
            if code_match:
                extracted_code = code_match.group(1)
                final_code = extracted_code
                self.logger.info("Successfully generated test datasets code")
                return f"```python\n{final_code}\n```"
            self.logger.warning("Failed to extract Python code from generated text")
            return "No code generated."
        except FileNotFoundError:
            self.logger.error(f"Multiple datasets code file not found: {multiple_datasets_code_path}")
            return f"Multiple datasets code file not found: {multiple_datasets_code_path}"
        except Exception as e:
            self.logger.error(f"Error generating best five datasets code: {str(e)}")
            return f"Error during code generation: {str(e)}"

    def generate_and_execute_test_datasets(self, multiple_datasets_code_path, original_dataset_path, data_dir, max_attempts=5):
        """
        Generate and execute code to create the best five datasets, and clean up datasets based on stability analysis results.

        Parameters:
        - multiple_datasets_code_path: Code file path for generating multiple datasets
        - original_dataset_path: Original dataset path
        - data_dir: Data root directory path
        - max_attempts: Maximum number of code generation attempts

        Returns:
        - Dictionary containing code execution results
        """
        self.logger.info(f"Starting test datasets generation for {original_dataset_path}")
        data_dir = os.path.dirname(original_dataset_path)
        code_dir = os.path.join(data_dir, 'code')
        os.makedirs(code_dir, exist_ok=True)

        code_file = os.path.join(code_dir, "generate_best_five_datasets.py")
        stability_dir = os.path.join(data_dir, 'stability_analysis')
        dataset_dir = os.path.join(data_dir, 'dataset')

        for attempt in range(max_attempts):
            try:
                self.logger.info(f"Attempt {attempt + 1}/{max_attempts} to generate and execute code")
                self.logger.info("Generating best five datasets code")
                code = self.generate_test_datasets_code(
                    multiple_datasets_code_path=multiple_datasets_code_path,
                    original_dataset_path=original_dataset_path
                )
                
                if "No code generated" in code:
                    self.logger.warning("Failed to generate test datasets code")
                    continue

                self.logger.info("Executing test datasets code")
                result = self.execute_generated_code(code, save_path=code_file)
                if "Code execution failed, maximum retries reached." in result:
                    self.logger.warning(f"Code execution failed after max retries, attempting regeneration")
                    continue
                
                # Check if stability analysis directory exists
                if not os.path.exists(stability_dir):
                    self.logger.warning("Stability analysis directory not found")
                    continue
                
                # Get CSV filenames from stability analysis directory
                stability_files = [f for f in os.listdir(stability_dir) if f.endswith('.csv')]
                if not stability_files:
                    self.logger.warning("No CSV files found in stability analysis directory")
                    continue
                
                # Clean up dataset directory
                if os.path.exists(dataset_dir):
                    dataset_files = os.listdir(dataset_dir)
                    for file in dataset_files:
                        if file.endswith('.csv'):
                            # Only keep files that appear in stability analysis
                            if file not in stability_files:
                                file_path = os.path.join(dataset_dir, file)
                                try:
                                    os.remove(file_path)
                                    self.logger.info(f"Removed unused dataset: {file}")
                                except Exception as e:
                                    self.logger.error(f"Error removing file {file}: {str(e)}")
                
                self.logger.info("Successfully executed test datasets code and cleaned up datasets")
                return {"Result": result}
                
            except Exception as e:
                self.logger.error(f"Error in attempt {attempt + 1}: {str(e)}")
                if attempt == max_attempts - 1:
                    return {"Result": f"Still failed after {max_attempts} attempts: {str(e)}"}
                continue

        return {"Result": f"Failed to generate and execute code successfully after {max_attempts} attempts."}

    def generate_model_evaluation_code(self, model_training_code_path, train_dataset_path, eval_dataset_path):
        """
        Generate model evaluation code using the best-fitting datasets and algorithms for training and evaluation.
        
        Parameters:
        - model_training_code_path: Original modeling code file path
        - train_dataset_path: Training dataset path
        - eval_dataset_path: Evaluation dataset path
        
        Returns:
        - Generated Python code string for training models and evaluating datasets
        """
        self.logger.info(f"Generating model evaluation code for {eval_dataset_path}")
        try:
            # Read original modeling code
            with open(model_training_code_path, 'r', encoding='utf-8') as f:
                model_training_code = f.read()
            
            input_data = MODEL_EVALUATION_TEMPLATE.format(
                train_dataset_path=train_dataset_path,
                eval_dataset_path=eval_dataset_path,
                model_training_code=model_training_code
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
from tools.ml_tools import *
from tools.ml_tools import transform_features,reduce_dimensions,select_features,discretize_features,create_polynomial_features

'''
                final_code = path_setup_code + extracted_code
                self.logger.info("Successfully generated model evaluation code")
                return f"```python\n{final_code}\n```"
            self.logger.warning("Failed to extract Python code from generated text")
            return "No code generated."
        except FileNotFoundError:
            self.logger.error(f"Model training code file not found: {model_training_code_path}")
            return f"Model training code file not found: {model_training_code_path}"
        except Exception as e:
            self.logger.error(f"Error generating model evaluation code: {str(e)}")
            return f"Error during code generation: {str(e)}"

    def generate_and_execute_model_evaluation(self, model_training_code_path, train_dataset_path, eval_dataset_path, max_attempts=5):
        """
        Generate and execute model evaluation code.

        Parameters:
        - model_training_code_path: Original modeling code file path
        - train_dataset_path: Training dataset path
        - eval_dataset_path: Evaluation dataset path
        - max_attempts: Maximum number of code generation attempts

        Returns:
        - Dictionary containing code execution results
        """
        self.logger.info(f"Starting model evaluation for {eval_dataset_path}")
        data_dir = os.path.dirname(eval_dataset_path)
        code_dir = os.path.join(data_dir, 'code')
        os.makedirs(code_dir, exist_ok=True)

        code_file = os.path.join(code_dir, "model_evaluation.py")

        for attempt in range(max_attempts):
            try:
                self.logger.info(f"Attempt {attempt + 1}/{max_attempts} to generate and execute code")
                self.logger.info("Generating model evaluation code")
                code = self.generate_model_evaluation_code(
                    model_training_code_path=model_training_code_path,
                    train_dataset_path=train_dataset_path,
                    eval_dataset_path=eval_dataset_path
                )
                
                if "No code generated" in code:
                    self.logger.warning("Failed to generate model evaluation code")
                    continue

                self.logger.info("Executing model evaluation code")
                result = self.execute_generated_code(code, save_path=code_file)
                if "Code execution failed, maximum retries reached." in result:
                    self.logger.warning(f"Code execution failed after max retries, attempting regeneration")
                    continue  # Try regenerating code
                
                self.logger.info("Successfully executed model evaluation code")
                return {"Result": result}
                
            except Exception as e:
                self.logger.error(f"Error in attempt {attempt + 1}: {str(e)}")
                if attempt == max_attempts - 1:
                    return {"Result": f"Still failed after {max_attempts} attempts: {str(e)}"}
                continue  # Try next attempt

        return {"Result": f"Failed to generate and execute code successfully after {max_attempts} attempts."}
