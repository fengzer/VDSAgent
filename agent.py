from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory,ConversationBufferWindowMemory
from agents import (ProblemDefinitionAndDataCollectionAgent, DataCleaningAndEDA_Agent, 
                    PCSAgent, PredictionAndInferenceAgent, 
                    ResultsEvaluationAgent)
import os
import shutil
import argparse
from config import get_llm

parser = argparse.ArgumentParser(description='Run Titanic Survival Prediction Experiment')
parser.add_argument('--csv_path', type=str, required=True, help='Path to training data CSV file')
parser.add_argument('--problem_description', type=str, required=True, help='Problem description')
parser.add_argument('--context_description', type=str, required=True, help='Context description')
args = parser.parse_args()

csv_file_path = args.csv_path
problem_description = args.problem_description
context_description = args.context_description

llm = get_llm()


DATA_DIR = os.path.dirname(os.path.abspath(csv_file_path))
CODE_DIR = os.path.join(DATA_DIR, 'code')


problem_agent = ProblemDefinitionAndDataCollectionAgent(memory=ConversationBufferMemory(), llm=llm)
pcs_agent = PCSAgent(memory=ConversationBufferMemory(), llm=llm, problem_description=problem_description, context_description=context_description)

problem_analysis_result = problem_agent.execute_problem_definition(
    csv_file_path, 
    problem_description, 
    context_description
)

pcs_hypotheses = pcs_agent.evaluate_problem_definition(
    problem_description=problem_description,
    context_description=context_description,
    var_json=problem_analysis_result["variable_descriptions"],
    unit_check=problem_analysis_result["observation_unit"]
)

clean_agent = DataCleaningAndEDA_Agent(
    memory=ConversationBufferMemory(),
    llm=llm,
    problem_description=problem_description,
    context_description=context_description,
    check_unit=problem_analysis_result["observation_unit"],
    var_json=problem_analysis_result["variable_descriptions"],
    hyp_json=pcs_hypotheses
)

cleaning_task_list = clean_agent.generate_cleaning_task_list()

cleaning_operations, data_cleaning_logs = clean_agent.execute_cleaning_tasks(
    cleaning_task_list, 
    csv_file_path
)

hypothesis_validation_results = []
for hypothesis in pcs_hypotheses:
    validation_code = clean_agent.generate_hypothesis_validation_code(
        csv_file_path=csv_file_path,
        hypothesis=hypothesis
    )
    code_execution_result = clean_agent.execute_generated_code(validation_code)
    hypothesis_conclusion = clean_agent.analyze_hypothesis_validation_result(code_execution_result)
    
    if hypothesis_conclusion:
        hypothesis_validation_results.extend(hypothesis_conclusion)
    else:
        hypothesis_validation_results.append(
            f"Hypothesis validation failed: '{hypothesis['hypothesis']}'"
        )

dataset_name = os.path.splitext(os.path.basename(csv_file_path))[0]

def get_code_path(operation_type: str) -> str:
    return os.path.join(CODE_DIR, f"{dataset_name}_{operation_type}.py")

clean_csv_file_path = clean_agent.execute_cleaning_operations(
    csv_file_path=csv_file_path,
    operations=cleaning_operations
)

cleaning_code_path = get_code_path('cleaning')
with open(cleaning_code_path, 'r', encoding='utf-8') as f:
    cleaning_code = f.read()

list1 = pcs_agent.execute_stability_analysis(csv_file_path=csv_file_path,cleaning_code=cleaning_code)

eda_problem = clean_agent.generate_eda_questions(csv_file_path=clean_csv_file_path)
eda_problem = clean_agent.solve_eda_questions(eda_questions=eda_problem,csv_file_path=clean_csv_file_path)
eda_summary = clean_agent.generate_eda_summary(eda_results=eda_problem)

prediction_agent = PredictionAndInferenceAgent(
    problem_description=problem_description,
    context_description=context_description,
    eda_summary=eda_summary,
    memory=ConversationBufferMemory(),
    llm=llm
)

response_var = prediction_agent.identify_response_variable(data_path=clean_csv_file_path)
feature_engineering_methods = prediction_agent.suggest_feature_engineering_methods(data_path=clean_csv_file_path)
model_method = prediction_agent.suggest_modeling_methods()

model_evaluation_report = prediction_agent.train_and_evaluate_combined_models(
    model_methods=model_method,
    feature_engineering_methods=feature_engineering_methods,
    csv_path=clean_csv_file_path
)

stability_analysis_dir = os.path.join(DATA_DIR, 'stability_analysis')
model_code_path = os.path.join(CODE_DIR,'train_models.py')

batch_evaluation_results = prediction_agent.execute_batch_evaluation(
    datasets_dir=stability_analysis_dir,
    model_code_path=model_code_path
)

prediction_agent.summarize_evaluation_results(batch_evaluation_results, clean_csv_file_path)

output_path = os.path.join(DATA_DIR, 'clean_dataset', 'model_evaluation_summary.md')
with open(output_path, 'r', encoding='utf-8') as f:
    report = f.read()

results_agent = ResultsEvaluationAgent(
    problem_description=problem_description,
    context_description=context_description,
    best_five_result=report,
    memory=ConversationBufferMemory(),
    llm=llm
)

multiple_datasets_code_path = os.path.join(CODE_DIR, 'train_stability.py')
original_dataset_path = os.path.join(DATA_DIR, 'test.csv')

test_datasets_results = results_agent.generate_and_execute_test_datasets(
    multiple_datasets_code_path=multiple_datasets_code_path,
    original_dataset_path=original_dataset_path,
    data_dir=DATA_DIR,
)

info = results_agent.generate_and_execute_model_evaluation(
    model_training_code_path=os.path.join(CODE_DIR, 'train_models.py'),
    train_dataset_path=os.path.join(DATA_DIR, 'stability_analysis'),
    eval_dataset_path=os.path.join(DATA_DIR, 'dataset')
)