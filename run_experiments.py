import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def setup_experiments(num_experiments, train_path, test_path, exp_root='titanic'):
    """
    Set up experiment environment, create folders and copy data
    Args:
        num_experiments: Number of experiments
        train_path: Training data file path
        test_path: Test data file path
        exp_root: Experiment root directory path
    """
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Original data files do not exist")
    
    for exp_num in range(1, num_experiments + 1):
        exp_folder = f'{exp_root}/{exp_num}'
        data_folder = f'{exp_root}/{exp_num}/data'
        os.makedirs(data_folder, exist_ok=True)
        
        shutil.copy2(train_path, os.path.join(data_folder, 'train.csv'))
        shutil.copy2(test_path, os.path.join(data_folder, 'test.csv'))
        
        print(f"Experiment {exp_num} folder and data preparation completed")

def run_single_experiment(exp_num, exp_root, problem_description, context_description):
    print(f"\nStarting experiment {exp_num}")
    csv_path = f'{exp_root}/{exp_num}/data/train.csv'
    
    try:
        subprocess.run([
            'python', 'agent.py', 
            '--csv_path', csv_path,
            '--problem_description', problem_description,
            '--context_description', context_description
        ], check=True)
        print(f"Experiment {exp_num} completed")
    except subprocess.CalledProcessError as e:
        print(f"Experiment {exp_num} failed: {e}")

def run_experiments(num_experiments, train_path='online_shopping/data/shopping_train.csv', 
                   test_path='online_shopping/data/shopping_test.csv', exp_root='titanic',
                   problem_description=None, context_description=None):
    """
    Run specified number of experiments
    Args:
        num_experiments: Number of experiments
        train_path: Training data file path
        test_path: Test data file path
        exp_root: Experiment root directory path
        problem_description: Description of the problem to solve
        context_description: Description of the dataset context
    """
    setup_experiments(num_experiments, train_path, test_path, exp_root)

    if not problem_description or not context_description:
        raise ValueError("problem_description and context_description must be provided")
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_single_experiment, exp_num, exp_root, problem_description, context_description) for exp_num in range(1, num_experiments + 1)]
        
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    NUM_EXPERIMENTS = 1
    COMPETITION_NAME = 'bank_churn'
    TRAIN_PATH = f'data_science_project/{COMPETITION_NAME}/data/train.csv'
    TEST_PATH = f'data_science_project/{COMPETITION_NAME}/data/test.csv'
    EXP_ROOT = f'data_science_project/{COMPETITION_NAME}'
    
    # Define problem and context descriptions
    PROBLEM_DESCRIPTION = """In this competition, you need to predict whether customers will retain their accounts or close them (i.e., churn). The prediction results will be evaluated using the area under the ROC curve to assess the relationship between predicted probabilities and actual targets. For each id in the test set, you need to predict the probability of the target variable Exited."""

    CONTEXT_DESCRIPTION = """The dataset for this competition (training and test sets) was generated using a deep learning model trained on the Bank Customer Churn Prediction dataset. The feature distributions are similar but not identical to the original dataset. The bank customer churn dataset contains information about bank customers who have left the bank or continue as customers, including the following attributes:

1. id: Unique identifier
2. CustomerId: Customer unique identifier
3. Surname: Customer surname
4. CreditScore: Customer credit score
5. Geography: Customer's country (France, Spain, or Germany)
6. Gender: Customer gender (Male/Female)
7. Age: Customer age
8. Tenure: Number of years with the bank
9. Balance: Account balance
10. NumOfProducts: Number of bank products used (e.g., savings account, credit card)
11. HasCrCard: Whether the customer has a credit card (1=Yes, 0=No)
12. IsActiveMember: Whether an active member (1=Yes, 0=No)
13. EstimatedSalary: Estimated customer salary
14. Exited: Whether the customer has churned (1=Yes, 0=No)"""
    
    print(f"Starting setup for {NUM_EXPERIMENTS} experiments...")
    run_experiments(NUM_EXPERIMENTS, TRAIN_PATH, TEST_PATH, EXP_ROOT,
                   PROBLEM_DESCRIPTION, CONTEXT_DESCRIPTION)