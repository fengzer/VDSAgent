from .base_agent import BaseDSLC_Agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
import pandas as pd
import json
from prompts.problem_definition_prompts import *


class ProblemDefinitionAndDataCollectionAgent(BaseDSLC_Agent):
    def __init__(self, memory=None, llm=None):
        super().__init__(
            name="Problem Definition and Data Collection",
            system_message=PROBLEM_DEFINITION_TEMPLATE,
            memory=memory,
            llm=llm
        )
        self.var_json = None
        self.unit_check = None
        self.data_corr = None
        self.logger.info("ProblemDefinitionAndDataCollectionAgent initialized")

    def load_data_preview(self, csv_file_path):
        """Load the first five rows of the CSV file and extract variable names and data preview"""
        self.logger.info(f"Loading data preview from {csv_file_path}")
        try:
            data = pd.read_csv(csv_file_path)
            preview = data.head().to_dict(orient="records")
            variables = list(data.columns)
            self.logger.info(f"Successfully loaded data preview with {len(variables)} variables")
            return preview, variables
        except Exception as e:
            self.logger.error(f"Error loading data preview: {str(e)}")
            return str(e), None

    def analyze_variables(self, csv_file_path, problem_description, context_description):
        """Analyze variables and get their descriptions"""
        self.logger.info(f"Analyzing variables for {csv_file_path}")
        try:
            preview, variables = self.load_data_preview(csv_file_path)
            
            if variables is None:
                self.logger.error(f"Failed to load data: {preview}")
                return f"Failed to load data: {preview}"

            input_data = VARIABLE_ANALYSIS_TEMPLATE.format(
                problem_description=problem_description,
                context_description=context_description,
                variable_info=', '.join(variables),
                data_preview=json.dumps(preview, ensure_ascii=False, indent=2)
            )

            #response = self.execute(input_data)
            response,_  = self.chat_with_memory(input_data,ConversationBufferMemory())
            self.var_json = self.parse_llm_json(response)
            if self.var_json:
                self.logger.info(f"Successfully analyzed {len(self.var_json)} variables")
            else:
                self.logger.warning("Failed to parse variable analysis response")
            return self.var_json
        except Exception as e:
            self.logger.error(f"Error analyzing variables: {str(e)}")
            raise

    def detect_observation_unit(self):
        """Detect the observation unit of the data"""
        self.logger.info("Detecting observation unit")
        try:
            #response = self.execute(OBSERVATION_UNIT_TEMPLATE)
            response ,_ = self.chat_with_memory(OBSERVATION_UNIT_TEMPLATE,ConversationBufferMemory())
            self.unit_check = response
            self.logger.info("Successfully detected observation unit")
            return self.unit_check
        except Exception as e:
            self.logger.error(f"Error detecting observation unit: {str(e)}")
            raise

    def evaluate_variable_relevance(self):
        """Evaluate if variables are relevant to data science projects"""
        self.logger.info("Evaluating variable relevance")
        try:
            if not self.var_json:
                self.logger.warning("Variable descriptions missing, cannot evaluate relevance")
                return "Variable descriptions missing, cannot evaluate relevance"
            
            input_data = VARIABLE_RELEVANCE_TEMPLATE.format(
                variable_descriptions=json.dumps(self.var_json, ensure_ascii=False, indent=2)
            )
            #response = self.execute(input_data)
            response,_ = self.chat_with_memory(input_data,ConversationBufferMemory())
            self.data_corr = response
            self.logger.info("Successfully evaluated variable relevance")
            return self.data_corr
        except Exception as e:
            self.logger.error(f"Error evaluating variable relevance: {str(e)}")
            raise

    def execute_problem_definition(self, csv_file_path, problem_description, context_description):
        """Overall execution function, execute steps 1, 2, and 3 in order"""
        self.logger.info(f"Starting problem definition execution for {csv_file_path}")
        try:
            # Step 1: Variable analysis
            self.logger.info("Step 1: Variable analysis")
            self.analyze_variables(csv_file_path, problem_description, context_description)
            
            # Step 2: Observation unit detection
            self.logger.info("Step 2: Observation unit detection")
            self.detect_observation_unit()
            
            # Step 3: Variable relevance evaluation
            self.logger.info("Step 3: Variable relevance evaluation")
            self.evaluate_variable_relevance()
            
            result = {
                "variable_descriptions": self.var_json,
                "observation_unit": self.unit_check,
                "relevance_evaluation": self.data_corr
            }
            self.logger.info("Successfully completed problem definition execution")
            return result
        except Exception as e:
            self.logger.error(f"Error in problem definition execution: {str(e)}")
            raise