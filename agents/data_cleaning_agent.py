# agents/data_cleaning_agent.py
from .base_agent import BaseDSLC_Agent
from langchain_core.messages import SystemMessage
import json
from langchain.memory import ConversationBufferMemory
import pandas as pd
import re
from prompts.data_cleaning_prompts import *
from tools import *
import os
from langchain_openai import ChatOpenAI
from config import get_multimodal_llm

class DataCleaningAndEDA_Agent(BaseDSLC_Agent):
    def __init__(self, problem_description, context_description, check_unit, var_json, hyp_json, memory=None, llm=None):
        system_message_content = DATA_CLEANING_TEMPLATE.format(
            problem_description=problem_description,
            context_description=context_description,
            check_unit=check_unit,
            var_json=json.dumps(var_json, ensure_ascii=False, indent=2),
            hyp_json=json.dumps(hyp_json, ensure_ascii=False, indent=2)
        )
        
        super().__init__(
            name="Data Cleaning and EDA",
            system_message=system_message_content,
            memory=memory,
            llm=llm
        )
        
        self.context_description = context_description
        self.var_json = var_json
        self.problem_description = problem_description
        self.check_unit = check_unit
        
        self.image_tool = ImageToTextTool(llm=get_multimodal_llm())
        self.logger.info("DataCleaningAndEDA_Agent initialized with tools")
    
    def generate_cleaning_task_list(self):
        """
        Generate a sequential list of data cleaning tasks.
        """
        self.logger.info("Generating cleaning task list")
        try:
            input_data = TASK_LIST_TEMPLATE.format(
                problem_description=self.problem_description,
                context_description=self.context_description,
                check_unit=self.check_unit,
                var_json=json.dumps(self.var_json, ensure_ascii=False, indent=2)
            )

            response = self.execute(input_data)
            parsed_json = self.parse_llm_json(response)

            if parsed_json:
                self.logger.info(f"Successfully generated {len(parsed_json)} cleaning tasks")
                return parsed_json
            else:
                self.logger.warning("No cleaning tasks detected")
                return "No data cleaning tasks detected."
        except Exception as e:
            self.logger.error(f"Error generating cleaning task list: {str(e)}")
            raise

    def generate_dimension_check_code(self, csv_file_path):
        """
        Generate Python code for checking data dimensions.
        """
        self.logger.info(f"Generating dimension check code for {csv_file_path}")
        try:
            input_data = DIMENSION_CHECK_TEMPLATE.format(
                csv_path=csv_file_path,
                context_description=self.context_description
            )
            generated_code, _ = self.chat_with_memory(input_data,ConversationBufferMemory())
            self.logger.info("Successfully generated dimension check code")
            return generated_code
        except Exception as e:
            self.logger.error(f"Error generating dimension check code: {str(e)}")
            raise
        
    def analyze_data_dimension(self, result):
        """
        Use LLM to analyze Python code execution results and generate a problem list.
        """
        self.logger.info("Analyzing data dimensions")
        try:
            input_data = DIMENSION_ANALYSIS_TEMPLATE.format(
                result=result
            )
            
            response, _ = self.chat_with_memory(input_data,ConversationBufferMemory())
            parsed_json = self.parse_llm_json(response)
            
            if parsed_json:
                self.logger.info(f"Found {len(parsed_json)} dimension-related issues")
                return parsed_json
            else:
                self.logger.info("No dimension issues found")
                return "No issues found, data dimensions meet expectations."
        except Exception as e:
            self.logger.error(f"Error analyzing data dimensions: {str(e)}")
            raise

    def check_for_invalid_values(self, csv_file_path):
        """
        Use LLM to check for invalid or unreasonable values in data description.
        """
        self.logger.info(f"Checking for invalid values in {csv_file_path}")
        try:
            data = pd.read_csv(csv_file_path)
            data_describe = data.select_dtypes('number').describe()
            data_describe = data_describe.to_string()

            input_data = INVALID_VALUE_TEMPLATE.format(
                data_describe=data_describe
            )
            
            response, _ = self.chat_with_memory(input_data,ConversationBufferMemory())
            parsed_json = self.parse_llm_json(response)
            
            if parsed_json:
                self.logger.info(f"Found {len(parsed_json)} invalid value issues")
                return parsed_json
            else:
                self.logger.info("No invalid values detected")
                return "No issues found, no invalid or unreasonable values detected in the data."
        except Exception as e:
            self.logger.error(f"Error checking for invalid values: {str(e)}")
            raise
    
    def generate_missing_value_analysis_code(self, csv_file_path):
        """
        Generate Python code for missing value analysis.
        """
        self.logger.info(f"Generating missing value analysis code for {csv_file_path}")
        try:
            data = pd.read_csv(csv_file_path)
            missing_data = data.isna().sum() / len(data.index)
            missing_data_sorted = missing_data.sort_values()    
            missing_data_str = missing_data_sorted.to_string()

            input_data = MISSING_VALUE_CODE_TEMPLATE.format(
                missing_ratio=missing_data_str,
                csv_path=csv_file_path
            )
            
            generated_code, _ = self.chat_with_memory(input_data,ConversationBufferMemory())
            self.logger.info("Successfully generated missing value analysis code")
            return generated_code
        except Exception as e:
            self.logger.error(f"Error generating missing value analysis code: {str(e)}")
            raise

    def analyze_missing_values_result(self, result, csv_file_path):
        """
        Use LLM to analyze missing value analysis results and generate cleaning operations JSON.
        """
        self.logger.info("Analyzing missing values result")
        try:
            data = pd.read_csv(csv_file_path)
            missing_data = data.isna().sum() / len(data.index)
            missing_data_sorted = missing_data.sort_values()    
            missing_data_str = missing_data_sorted.to_string()

            input_data = MISSING_VALUE_CLEANING_TEMPLATE.format(
                missing_ratio=missing_data_str,
                missing_analysis=result,
                data_unit=self.check_unit
            )
            
            response, _ = self.chat_with_memory(input_data,ConversationBufferMemory())
            parsed_json = self.parse_llm_json(response)
            if parsed_json:
                self.logger.info(f"Generated {len(parsed_json)} missing value cleaning operations")
                return parsed_json
            else:
                self.logger.warning("No cleaning operations generated for missing values")
                return "No JSON cleaning operations generated."
        except Exception as e:
            self.logger.error(f"Error analyzing missing values result: {str(e)}")
            raise

    def generate_data_integrity_check_code(self, csv_file_path):
        """
        Generate Python code for checking data integrity.
        """
        self.logger.info(f"Generating data integrity check code for {csv_file_path}")
        try:
            input_data = DATA_INTEGRITY_CHECK_TEMPLATE.format(
                csv_path=csv_file_path,
            )
            generated_code, _ = self.chat_with_memory(input_data,ConversationBufferMemory())
            self.logger.info("Successfully generated data integrity check code")
            return generated_code
        except Exception as e:
            self.logger.error(f"Error generating data integrity check code: {str(e)}")
            raise
    
    def analyze_and_generate_fillna_operations(self, result):
        """
        Use LLM to analyze data integrity check results and generate cleaning operations.
        """
        self.logger.info("Analyzing data integrity check results")
        try:
            input_data = DATA_INTEGRITY_CLEANING_TEMPLATE.format(
                integrity_result=result,
            )
            response, _ = self.chat_with_memory(input_data,ConversationBufferMemory())
            parsed_json = self.parse_llm_json(response)
            if parsed_json:
                self.logger.info(f"Generated {len(parsed_json)} data integrity cleaning operations")
                return parsed_json
            else:
                self.logger.info("No cleaning operations needed for data integrity")
                return "Data is complete, no cleaning operations needed."
        except Exception as e:
            self.logger.error(f"Error analyzing data integrity: {str(e)}")
            raise

    def generate_cleaning_operations(self, pro_json):
        """
        Generate cleaning operation suggestions by creating a merged query content based on the problem list (pro_json)
        and asking LLM in one go. Returns a merged JSON of cleaning operations.
        """
        self.logger.info("Generating cleaning operations from problem list")
        try:
            input_data = CLEANING_OPERATIONS_TEMPLATE.format(
                problem_list=json.dumps(pro_json, ensure_ascii=False, indent=2)
            )

            response, _ = self.chat_with_memory(input_data,ConversationBufferMemory())
            json_output = self.parse_llm_json(response)
            if json_output:
                self.logger.info(f"Generated {len(json_output)} cleaning operations")
                return json_output
            else:
                self.logger.warning("No cleaning operations generated")
                return "No cleaning operations generated."
        except Exception as e:
            self.logger.error(f"Error generating cleaning operations: {str(e)}")
            return f"Execution error: {str(e)}"

    def generate_hypothesis_validation_code(self, csv_file_path, hypothesis):
        """
        Generate Python code for validating a single hypothesis.
        """
        self.logger.info(f"Generating hypothesis validation code for {csv_file_path}")
        try:
            input_data = HYPOTHESIS_VALIDATION_CODE_TEMPLATE.format(
                csv_path=csv_file_path,
                hypothesis=hypothesis['hypothesis'],
                validation_method=hypothesis['validation_method']
            )
            generated_code, _ = self.chat_with_memory(input_data,ConversationBufferMemory())
            self.logger.info("Successfully generated hypothesis validation code")
            return generated_code
        except Exception as e:
            self.logger.error(f"Error generating hypothesis validation code: {str(e)}")
            raise
    
    def analyze_hypothesis_validation_result(self, validation_result):
        """
        Analyze validation results to determine if the hypothesis holds true and generate updated hypothesis conclusions.
        """
        self.logger.info("Analyzing hypothesis validation results")
        try:
            input_data = HYPOTHESIS_VALIDATION_ANALYSIS_TEMPLATE.format(
                validation_result=validation_result
            )

            response, _ = self.chat_with_memory(input_data,ConversationBufferMemory())
            updated_hypothesis = self.parse_llm_json(response)
            if updated_hypothesis:
                self.logger.info("Successfully analyzed hypothesis validation results")
                return updated_hypothesis
            else:
                self.logger.warning("Failed to analyze hypothesis validation results")
                return None
        except Exception as e:
            self.logger.error(f"Error analyzing hypothesis validation: {str(e)}")
            raise

    def generate_cleaning_code(self, csv_file_path, operations):
        """
        Generate data cleaning code.

        Parameters:
        - csv_file_path: Path to the original CSV data file
        - operations: List of cleaning operations in JSON format

        Returns:
        - Complete Python code string (including setup code)
        """
        self.logger.info(f"Generating cleaning code for {csv_file_path}")
        try:
            data = pd.read_csv(csv_file_path)
            from io import StringIO
            buffer = StringIO()
            data.info(buf=buffer)
            datainfo = buffer.getvalue()

            tools = [
                'fill_missing_values_tools',
                'remove_columns_tools',
                'handle_outliers_tools',
                'encode_categorical_tools'
            ]

            doc_path = 'tools/ml_tools_doc/data_cleaning_tools.md'
            with open(doc_path, 'r', encoding='utf-8') as f:
                tool_descriptions = f.read()

            input_data = DATA_CLEANING_FUNCTION_TEMPLATE.format(
                csv_path=csv_file_path,
                context_description=self.context_description,
                var_descriptions=json.dumps(self.var_json, ensure_ascii=False, indent=2),
                check_unit=self.check_unit,
                data_info=datainfo,
                cleaning_operations=json.dumps(operations, ensure_ascii=False, indent=2),
                tools=', '.join(tools),
                tool_descriptions=tool_descriptions
            )

            generated_code, _ = self.chat_with_memory(input_data,ConversationBufferMemory())
            code_match = re.search(r"```python\n(.*?)\n```", generated_code, re.DOTALL)
            if code_match:
                self.logger.info("Successfully generated cleaning code")
                extracted_code = code_match.group(1)
                path_setup_code = '''import os,sys,re
current_path = os.path.abspath(__file__)
match = re.search(r'(.*VDSAgent)', current_path)
if not match:
    raise FileNotFoundError("Could not find VDSAgent directory")
sys.path.append(match.group(1))
from tools.ml_tools import *

'''
                final_code = path_setup_code + extracted_code
                return f"```python\n{final_code}\n```"
            else:
                self.logger.warning("Failed to extract Python code from generated text")
                return "Unable to generate valid Python code."
        except Exception as e:
            self.logger.error(f"Error generating cleaning code: {str(e)}")
            raise

    def execute_cleaning_operations(self, csv_file_path, operations):
        """
        Execute data cleaning operations and return the path to the cleaned dataset.
        """
        self.logger.info(f"Starting cleaning operations for {csv_file_path}")
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            self.logger.info(f"Attempt {retry_count + 1} of {max_retries}")
            generated_code = self.generate_cleaning_code(csv_file_path, operations)
            if "Unable to generate" in generated_code:
                retry_count += 1
                self.logger.warning(f"Failed to generate code, attempt {retry_count} of {max_retries}")
                if retry_count == max_retries:
                    return f"Unable to generate valid code after {max_retries} attempts."
                continue

            csv_dir = os.path.dirname(csv_file_path)
            code_dir = os.path.join(csv_dir, 'code')
            os.makedirs(code_dir, exist_ok=True)
            
            py_filename = os.path.basename(csv_file_path).replace('.csv', '_cleaning.py')
            py_file_path = os.path.join(code_dir, py_filename)
            
            result = self.execute_generated_code(generated_code, save_path=py_file_path)
            
            # Check if result contains dataset path
            if not isinstance(result, str) or 'dataset' not in result.lower() or '.csv' not in result:
                if os.path.exists(py_file_path):
                    os.remove(py_file_path)
                retry_count += 1
                self.logger.warning(f"Invalid result format, attempt {retry_count} of {max_retries}")
                if retry_count == max_retries:
                    return f"Unable to generate result containing dataset path after {max_retries} attempts."
                continue
            
            try:
                # Run unit tests
                cleaned_data_path = result
                passed, report = self.run_unit_tests(
                    cleaned_data_path=cleaned_data_path,
                    original_data_path=csv_file_path
                )
                
                if passed:
                    self.logger.info("Cleaning operations completed successfully")
                    return result
                else:
                    self.logger.warning(f"Unit tests failed: {report}")
                    with open(py_file_path, 'r', encoding='utf-8') as f:
                        code_to_debug = f.read()
                    
                    debug_tool = DebugTool(llm=self.llm)
                    debug_result = debug_tool.run(tool_input={
                        "code": code_to_debug,
                        "error_message": "Unit tests failed",
                        "output_message": report,
                        "tools_description": "Data cleaning code debugging tool",
                        "is_test_error": True,
                    })
                    
                    if debug_result["status"] == "success":
                        self.logger.info("Debug successful, retrying with fixed code")
                        fixed_result = self.execute_generated_code(
                            f"```python\n{debug_result['fixed_code']}\n```",
                            save_path=py_file_path,
                            is_debug=True
                        )
                        
                        if isinstance(fixed_result, str) and '.csv' in fixed_result:
                            # Run unit tests on fixed code
                            fixed_passed, fixed_report = self.run_unit_tests(
                                cleaned_data_path=fixed_result,
                                original_data_path=csv_file_path
                            )
                            if fixed_passed:
                                self.logger.info("Fixed code passed unit tests")
                                return fixed_result
                
                # If debug failed, fixed code has issues, or unit tests failed, retry
                if os.path.exists(py_file_path):
                    os.remove(py_file_path)
                if os.path.exists(cleaned_data_path):
                    os.remove(cleaned_data_path)
                if 'fixed_result' in locals() and os.path.exists(fixed_result):
                    os.remove(fixed_result)
                
                retry_count += 1
                self.logger.warning(f"Debug attempt failed, retry {retry_count} of {max_retries}")
                if retry_count == max_retries:
                    return f"Unable to generate code meeting requirements after {max_retries} attempts. Last test report: {report}"
                continue
            
            except Exception as e:
                self.logger.error(f"Error during cleaning operation: {str(e)}")
                if os.path.exists(py_file_path):
                    os.remove(py_file_path)
                if 'cleaned_data_path' in locals() and os.path.exists(cleaned_data_path):
                    os.remove(cleaned_data_path)
                
                retry_count += 1
                if retry_count == max_retries:
                    return f"Error occurred after {max_retries} attempts: {str(e)}"
                continue
        
        return f"Unable to generate valid code after {max_retries} attempts."

    def generate_eda_questions(self, csv_file_path):
        """
        Generate data exploration related questions based on cleaned data.

        Parameters:
        - csv_file_path: Path to cleaned data.

        Returns:
        - Generated EDA questions (JSON format).
        """
        self.logger.info(f"Generating EDA questions for {csv_file_path}")
        try:
            data = pd.read_csv(csv_file_path)
            from io import StringIO
            buffer = StringIO()
            data.info(buf=buffer)
            datainfo = buffer.getvalue()

            input_data = EDA_QUESTIONS_TEMPLATE.format(
                problem_description=self.problem_description,
                data_structure=datainfo,
                data_preview=', '.join(data.columns)
            )

            response, _ = self.chat_with_memory(input_data,ConversationBufferMemory())
            parsed_json = self.parse_llm_json(response)
            if parsed_json:
                self.logger.info(f"Successfully generated {len(parsed_json)} EDA questions")
                return parsed_json
            else:
                self.logger.warning("Failed to extract valid JSON questions")
                return "Failed to extract valid JSON format questions from LLM response."
        except Exception as e:
            self.logger.error(f"Error generating EDA questions: {str(e)}")
            return f"Execution error: {str(e)}"

    def generate_eda_code(self, csv_file_path, question):
        """
        Generate Python code based on EDA question.
        
        Parameters:
        - csv_file_path: Path to the cleaned data file
        - question: Specific EDA question to analyze

        Returns:
        - Generated Python code
        """
        self.logger.info(f"Generating EDA code for question: {question}")
        try:
            data = pd.read_csv(csv_file_path)
            from io import StringIO
            buffer = StringIO()
            data.info(buf=buffer)
            datainfo = buffer.getvalue()

            input_data = EDA_CODE_TEMPLATE.format(
                csv_path=csv_file_path,
                question=question,
                data_structure=datainfo,
                data_preview=', '.join(data.columns)
            )
            response, _ = self.chat_with_memory(input_data,ConversationBufferMemory())
            self.logger.info("Successfully generated EDA code")
            return response
        except Exception as e:
            self.logger.error(f"Error generating EDA code: {str(e)}")
            raise

    def analyze_eda_result(self, question, result, image_analysis=None):
        """
        Update EDA question conclusions based on code execution results and image analysis.

        Parameters:
        - question: Original EDA question
        - result: Text result from code execution
        - image_analysis: Image analysis results (optional)

        Returns:
        - Updated EDA question and conclusions (JSON format)
        """
        self.logger.info(f"Analyzing EDA results for question: {question}")
        try:
            input_data = EDA_ANALYSIS_TEMPLATE.format(
                question=question,
                result=result,
                image_analysis=image_analysis if image_analysis else "No image analysis results"
            )

            response, _ = self.chat_with_memory(input_data,ConversationBufferMemory())
            updated_json = self.parse_llm_json(response)
            if updated_json:
                self.logger.info("Successfully analyzed EDA results")
                return updated_json
            else:
                self.logger.warning("Failed to analyze EDA results")
                return {"question": question, "conclusion": "Unable to update conclusion"}
        except Exception as e:
            self.logger.error(f"Error analyzing EDA results: {str(e)}")
            raise
    
    def solve_eda_questions(self, csv_file_path, eda_questions):
        """
        Main logic for solving EDA questions.
        
        Parameters:
        - csv_file_path: Path to the cleaned data file
        - eda_questions: List of EDA questions in JSON format

        Returns:
        - Updated list of EDA questions with conclusions
        """
        self.logger.info(f"Starting to solve {len(eda_questions)} EDA questions")
        try:
            updated_questions = []

            for question_item in eda_questions:
                for key, question in question_item.items():
                    if key.startswith("question"):
                        self.logger.info(f"Processing question: {question}")

                        generated_code = self.generate_eda_code(csv_file_path, question)
                        if not generated_code:
                            self.logger.warning(f"Failed to generate code for question: {question}")
                            updated_questions.append({
                                key: question,
                                "conclusion": "Failed to generate code."
                            })
                            continue

                        try:
                            result = self.execute_generated_code(generated_code=generated_code)
                            result_dict = eval(result)
                            
                            image_analysis_results = []
                            for plot_path in result_dict.get('generated_plots', []):
                                if os.path.exists(plot_path):
                                    self.logger.info(f"Analyzing plot: {plot_path}")
                                    image_analysis = self.analyze_image(
                                        image_path=plot_path,
                                        prompt="Please analyze this data visualization chart and describe the main trends, patterns, and key findings."
                                    )
                                    image_analysis_results.append(image_analysis)
                            
                            combined_result = {
                                'text_result': result_dict.get('text_result', ''),
                                'image_analysis': '\n'.join(image_analysis_results)
                            }

                            updated_question = self.analyze_eda_result(
                                question=question,
                                result=combined_result['text_result'],
                                image_analysis=combined_result['image_analysis']
                            )
                            updated_questions.append(updated_question)
                            self.logger.info(f"Successfully processed question: {question}")
                            
                        except Exception as exec_error:
                            self.logger.error(f"Error executing code for question {question}: {str(exec_error)}")
                            updated_questions.append({
                                key: question,
                                "conclusion": f"Code execution failed: {str(exec_error)}"
                            })

            return updated_questions
        except Exception as e:
            self.logger.error(f"Error solving EDA questions: {str(e)}")
            return f"Error occurred during execution: {str(e)}"
        
    def generate_pcs_evaluation_code(self, csv_file_path, conclusion):
        """
        Generate Python code for PCS evaluation based on conclusions.

        Parameters:
        - csv_file_path: Path to the cleaned data file
        - conclusion: Conclusion to be evaluated

        Returns:
        - Generated Python code
        """
        self.logger.info(f"Generating PCS evaluation code for {csv_file_path}")
        try:
            data = pd.read_csv(csv_file_path)
            from io import StringIO
            buffer = StringIO()
            data.info(buf=buffer)
            datainfo = buffer.getvalue()

            input_data = PCS_EVALUATION_TEMPLATE.format(
                csv_path=csv_file_path,
                conclusions=conclusion,
                data_structure=datainfo,
                data_preview=', '.join(data.columns)
            )

            response, _ = self.chat_with_memory(input_data, ConversationBufferMemory())
            self.logger.info("Successfully generated PCS evaluation code")
            return response
        except Exception as e:
            self.logger.error(f"Error generating PCS evaluation code: {str(e)}")
            raise
    
    def check_discrete_variables(self, csv_file_path, question):
        """
        Determine if discrete variables in the data need to be converted to numerical values.

        Parameters:
        - csv_file_path: Path to the cleaned data file
        - question: Problem to be solved

        Returns:
        - True if numerical conversion is needed
        - False if numerical conversion is not needed
        """
        self.logger.info(f"Checking discrete variables for {csv_file_path}")
        try:
            input_data = DISCRETE_VAR_CHECK_TEMPLATE.format(
                csv_path=csv_file_path,
                problem_description=question
            )
            response, _ = self.chat_with_memory(input_data, ConversationBufferMemory())
            response_json = self.parse_llm_json(response)

            if response_json and isinstance(response_json, list):
                result = response_json[0].get("needs_numerical_conversion", False)
                self.logger.info(f"Discrete variables check result: {result}")
                return result
            else:
                self.logger.warning("Failed to parse discrete variables check result")
                return False
        except Exception as e:
            self.logger.error(f"Error checking discrete variables: {str(e)}")
            return False
    
    def generate_discrete_variable_code(self, csv_file_path):
        """
        Generate Python code for processing discrete variables.

        Parameters:
        - csv_file_path: Path to the cleaned data file

        Returns:
        - Path to the generated Python file
        """
        self.logger.info(f"Generating discrete variable code for {csv_file_path}")
        try:
            data = pd.read_csv(csv_file_path)
            # Get discrete variables and their unique values
            discrete_variables = {
                col: data[col].unique().tolist()
                for col in data.select_dtypes(include=['object', 'category']).columns
            }
            discrete_info = json.dumps(discrete_variables, ensure_ascii=False, indent=2)

            input_data = DISCRETE_VAR_CODE_TEMPLATE.format(
                csv_path=csv_file_path,
                discrete_vars=discrete_info
            )
            response, _ = self.chat_with_memory(input_data, ConversationBufferMemory())
            code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)

            if not code_match:
                self.logger.warning("No valid code content generated by LLM")
                return "LLM failed to generate valid code content."

            extracted_code = code_match.group(1)
            py_filename = os.path.basename(csv_file_path).replace('.csv', '_transformed.py')
            py_file_path = os.path.join(os.path.dirname(csv_file_path), py_filename)

            with open(py_file_path, 'w', encoding='utf-8') as py_file:
                py_file.write(extracted_code)

            self.logger.info(f"Successfully generated discrete variable code: {py_file_path}")
            return py_file_path
        except Exception as e:
            self.logger.error(f"Error generating discrete variable code: {str(e)}")
            return f"Error occurred during execution: {str(e)}"

    def load_and_compare_data(self, csv_file_path):
        """
        加载 CSV 文件的前10行和随机10行，将这20行数据发送给 LLM 进行对比并验证数据是否正确加载。
        """
        self.logger.info(f"Loading and comparing data from {csv_file_path}")
        try:
            data = pd.read_csv(csv_file_path)
            first_10_rows = data.head(10)
            random_10_rows = data.sample(10)

            first_10_dict = first_10_rows.to_dict(orient="records")
            random_10_dict = random_10_rows.to_dict(orient="records")
            first_10_str = json.dumps(first_10_dict, ensure_ascii=False, indent=2)
            random_10_str = json.dumps(random_10_dict, ensure_ascii=False, indent=2)

            input_data = DATA_LOADING_TEMPLATE.format(
                first_10_rows=first_10_str,
                random_10_rows=random_10_str
            )
            response, _ = self.chat_with_memory(input_data, ConversationBufferMemory())
            self.logger.info("Successfully compared data samples")
            return response
        except Exception as e:
            self.logger.error(f"Error loading and comparing data: {str(e)}")
            return f"Error occurred while loading and comparing data: {str(e)}"
        
    def execute_cleaning_tasks(self, task_list, csv_file_path):
        """
        Execute data cleaning tasks sequentially based on task list, including problem list generation and cleaning operations generation.
        
        Parameters:
            task_list (list): List containing task names
            csv_file_path (str): Path to the data file
        
        Returns:
            list: Merged list of cleaning operations in JSON format
            list: Error logs
        """
        self.logger.info(f"Starting execution of {len(task_list)} cleaning tasks")
        problem_list = []  # Store problem list
        cleaning_operations = []  # Store cleaning operations
        error_logs = []  # Store error logs

        for task in task_list:
            task_name = task.get('task_name')
            self.logger.info(f"Processing task: {task_name}")

            try:
                if task_name == 'dimension_analysis_and_problem_list_generation':
                    self.logger.info("Executing dimension analysis task")
                    dimension_check_code = self.generate_dimension_check_code(csv_file_path=csv_file_path)
                    dimension_check_result = self.execute_generated_code(dimension_check_code)
                    dimension_problems = self.analyze_data_dimension(dimension_check_result)
                    if isinstance(dimension_problems, list):
                        self.logger.info(f"Found {len(dimension_problems)} dimension-related problems")
                        problem_list.extend(dimension_problems)

                elif task_name == 'invalid_value_analysis_and_problem_list_generation':
                    self.logger.info("Executing invalid value analysis task")
                    invalid_value_problems = self.check_for_invalid_values(csv_file_path=csv_file_path)
                    if isinstance(invalid_value_problems, list):
                        self.logger.info(f"Found {len(invalid_value_problems)} invalid value problems")
                        problem_list.extend(invalid_value_problems)

                elif task_name == 'missing_value_analysis_and_cleaning_operation_generation':
                    self.logger.info("Executing missing value analysis task")
                    missing_value_code = self.generate_missing_value_analysis_code(csv_file_path=csv_file_path)
                    missing_value_result = self.execute_generated_code(missing_value_code)
                    missing_value_cleaning = self.analyze_missing_values_result(missing_value_result, csv_file_path=csv_file_path)
                    if isinstance(missing_value_cleaning, list):
                        print(missing_value_cleaning)
                        self.logger.info(f"Generated {len(missing_value_cleaning)} missing value cleaning operations")
                        cleaning_operations.extend(missing_value_cleaning)

                elif task_name == 'data_integrity_analysis_and_cleaning_operation_generation':
                    self.logger.info("Executing data integrity analysis task")
                    data_integrity_code = self.generate_data_integrity_check_code(csv_file_path=csv_file_path)
                    data_integrity_result = self.execute_generated_code(data_integrity_code)
                    data_integrity_cleaning = self.analyze_and_generate_fillna_operations(data_integrity_result)
                    if isinstance(data_integrity_cleaning, list):
                        self.logger.info(f"Generated {len(data_integrity_cleaning)} data integrity cleaning operations")
                        cleaning_operations.extend(data_integrity_cleaning)

                else:
                    error_message = f"Task {task_name} not implemented or unknown"
                    self.logger.warning(error_message)
                    error_logs.append({
                        "task_name": task_name,
                        "error_message": error_message
                    })

            except Exception as e:
                error_message = f"Error in task {task_name}: {str(e)}"
                self.logger.error(error_message)
                error_logs.append({
                    "task_name": task_name,
                    "error_message": error_message
                })

        if problem_list:
            try:
                self.logger.info("Generating cleaning operations from problem list")
                problem_cleaning_operations = self.generate_cleaning_operations(problem_list)
                if isinstance(problem_cleaning_operations, list):
                    self.logger.info(f"Generated {len(problem_cleaning_operations)} additional cleaning operations")
                    cleaning_operations.extend(problem_cleaning_operations)
            except Exception as e:
                error_message = f"Error generating cleaning operations from problem list: {str(e)}"
                self.logger.error(error_message)
                error_logs.append({
                    "task_name": "generate_cleaning_operations_from_problem_list",
                    "error_message": error_message
                })

        return cleaning_operations, error_logs

    def analyze_image(self, image_path: str, prompt: str = "请分析这个图片") -> str:
        """Analyze image content"""
        return self.image_tool.run(tool_input={
            "image_path": image_path,
            "prompt": prompt
        })

    def generate_eda_summary(self, eda_results):
        """
        Generate a summary report based on the completed EDA question list.

        Parameters:
        - eda_results: A list of EDA results containing questions and conclusions

        Returns:
        - A complete EDA summary text
        """
        self.logger.info("Generating EDA summary from results")
        try:
            input_data = EDA_SUMMARY_TEMPLATE.format(
                eda_results=json.dumps(eda_results, ensure_ascii=False, indent=2),
                problem_description=self.problem_description
            )
            
            response, _ = self.chat_with_memory(input_data, ConversationBufferMemory())
            self.logger.info("Successfully generated EDA summary")
            return response if response else "Unable to generate a valid EDA analysis summary."
        except Exception as e:
            self.logger.error(f"Error generating EDA summary: {str(e)}")
            return "Unable to generate a valid EDA analysis summary."