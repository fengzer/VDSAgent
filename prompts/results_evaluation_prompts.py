# Model Results Evaluation Template
RESULTS_EVALUATION_TEMPLATE = """You are a data science expert, focusing on evaluating machine learning model performance on test sets using PCS principles.
1. Definition of Data Science Terms:
    1. PCS Predictive Fit：
      In the PCS framework, a predictive fit (or just a "fit") refers to the pairing of an algorithm and a particular cleaned/preprocessed training dataset used for training the algorithm.
    2. The single PCS predictive fit：
      science analog of the traditional approach, this approach involves identifying the single PCS predictive fit with the highest validation set predictive performance. Future predictions for new data points are then computed using the chosen predictive fit. If the best-performing fit does not pass a problem-specific predictability screening on the validation set (i.e., if the best-performing algorithm does not achieve adequate performance as dictated by the domain problem), then no predictive fit should be used. By considering the PCS predictive fits that span a range of alternative algorithmic and cleaning/preprocessing options, this approach takes into account the uncertainty arising from the algorithm and cleaning/preprocessing choices when computing the prediction. However, by presenting just a single prediction value as the final output, this uncertainty is not conveyed to the user.
    3. The PCS ensemble：
      Instead of choosing just one PCS predictive fit, this approach creates an ensemble from the PCS predictive fits that pass a problem-specific predictability screening on the validation set. One way to compute the ensemble's response prediction for a future data point is by computing the average response prediction for continuous responses or a majority vote for binary responses. By considering a range of alternative algorithmic and cleaning/preprocessing judgment call options, this approach takes into account the uncertainty arising from the algorithmic and cleaning/preprocessing judgment calls when computing the prediction. However, by presenting a single prediction value as the output, this approach also does not convey this uncertainty to the user.
    4. The PCS calibrated PPIs:
      Rather than providing a single prediction for each new data point (e.g., using a single fit or an ensemble), this approach creates an interval using the predictions from the fits that pass a problem-specific predictability screening on the validation set. These intervals are called prediction perturbation intervals (PPIs). Because the length of the interval will typically depend on the number of predictions used to create it, the intervals are calibrated to achieve a prespecified coverage level (e.g., the lengths of the intervals are modified to ensure that 90 percent of the validation set intervals contain the observed response). Currently, this interval-based approach is only designed for continuous response predictions.1 By presenting an interval based on the range of alternative algorithmic and cleaning/preprocessing options, this approach can convey the underlying prediction uncertainty to the user.
    5. Training Set, Validation Set, Test Set
      Training and validation sets are data where we know the true labels, where we train on the training set and evaluate model performance on the validation set
      Test set is real data where we don't know the true labels

2. Technical Analysis Description and Relevant Operations:
    1. Using PCS to Select Single Predictive Fit
        1. Predictability screening: Identify which fit has the best validation set performance.
            a. Create and document several different versions of the cleaned and preprocessed training and validation sets using different combinations of cleaning and preprocessing judgment calls.2 Let K denote the number of cleaned/preprocessed datasets you end up with.  
            b. Train each predictive algorithm (e.g., LS, RF, and so on) using each of the K cleaned/preprocessed training datasets. If you have L different algorithms, then you will end up with K × L predictive fits.  
            c. Generate a response prediction for each validation set observation using each of the K × L predictive fits.
            d. Identify which predictive fit from among the K × L fits yields the best validation predictive performance. This is the best PCS fit.
        2. Computing predictions for new observations: Predictions for new observations can be computed using the best PCS fit.  
        3. Test set evaluation: Evaluate your best fit using the test set observations to provide a final independent assessment of its predictive performance.
    2. PCS Ensemble
        1. Predictability screening: Identify which fits will be included in the ensemble.  
            a. Create and document several different versions of the cleaned and preprocessed training, validation, and test sets using a range of combinations of cleaning and preprocessing judgment calls. Let K denote the number of cleaned/preprocessed datasets you end up with.
            b. Train each relevant predictive algorithm (e.g., LS, RF, etc.) using each of the K cleaned/preprocessed versions of the training data. If you have L different algorithms, then you will end up with K × L different predictive fits.  
            c. Using each of the K × L predictive fits, generate a response prediction for each validation set observation.
            d. Evaluate the validation set predictive performance for each of the K × L fits (e.g., using rMSE or correlation for continuous responses and accuracy or AUC for binary responses).
            e. Conduct predictability screening by keeping only the best fits (say, the top 10 percent) in terms of validation set predictive performance. This threshold (e.g., the top 10 percent of fits) can be based on domain knowledge or it can be tuned using the ensemble's predictive performance on the validation set.
        2. Compute predictions for new observations using the ensemble: Predictions from the ensemble can be computed based on the fits that pass the predictability screening step (e.g., the top 10 percent of fits) by averaging their continuous predicted responses or taking the majority class of their binary predicted responses.
        3. Test set evaluation: Evaluate your ensemble's predictive performance using the test set observations to provide a final independent assessment of its predictive performance.

3. In our predictive inference exploration, we have completed the following steps:
    1. We have created K cleaning/preprocessing judgment combinations through PCS principles and created L different algorithms
    2. We have divided the initial data into training and validation sets, and have trained and evaluated K × L predictive fits on the training set and validation set respectively
    3. We have obtained evaluation results of the best 5 predictive fits on the validation set
    4. We will provide relevant code for cleaning to obtain different datasets and batch training different datasets
    5. Next, I need your help to use PCS to select single predictive fit and PCS ensemble. We will provide a test set without known true labels, and use the best 5 predictive fits to train algorithms on the training set and complete fitting on the test set

4. Notes
  1. To avoid escape errors, please use / when defining paths, such as ...VDSAgent/obesity_risks/1/data/test.csv

5. Related Background
Data Science Project Description: {problem_description}
Data Background Description: {context_description}
Best Five Predictive Fits: {best_five_result}

"""

# Generate Best Five Datasets Code Template
BEST_FIVE_DATASETS_TEMPLATE = """Task: Generate Evaluation Dataset Code
Task Description: Create a new Python code that specifies the generation of evaluation datasets based on the training dataset generation code. If there are ID type variables in the dataset, please do not delete them.

Input Data:
1. Evaluation Dataset Path: {original_dataset_path}
2. Training Dataset Generation Code: 
```python
{multiple_datasets_code}
```

Requirements:
1. Maintain the complete logic of the original code, just replace the training dataset path with the evaluation dataset path
2. Generated datasets must be saved in the dataset folder under the evaluation dataset directory
   - For example: if evaluation dataset path is /path/to/original.csv
   - Then generated datasets should be saved in /path/to/dataset/ directory

Output Data:
```python
[Complete python code]
```
"""

# Model Evaluation Code Template
MODEL_EVALUATION_TEMPLATE = """Task: Generate Model Evaluation Code
Task Description: Use the best predictive fit's corresponding dataset and algorithm to train the model and evaluate the specified dataset.

Input Data:
1. Training Dataset Path: {train_dataset_path}
2. Evaluation Dataset Path: {eval_dataset_path}
3. Original Modeling Code: 
```python
{model_training_code}
```

Requirements:
0. The feature engineering utility functions in the original modeling code have been pre-declared using from tools.ml_tools import transform_features, reduce_dimensions, select_features, discretize_features, create_polynomial_features, just call them directly in the generated code without re-declaring and defining
1. From the best five fitting results, obtain:
   - Best algorithm selection
   - Corresponding training dataset version
2. First read the datasets:
   - Training dataset: Use the dataset version that achieved the best predictive fit from [train_dataset_path]
   - Evaluation dataset: Use the corresponding version from [eval_dataset_path] that matches the best predictive fit's training dataset
   Note: Make sure to use the exact same dataset version that produced the best results during model selection
3. Then, use the original modeling code to call feature engineering functions from tools.ml_tools to transform features for both training and evaluation datasets. Note that there are no prediction variables in the evaluation dataset, so no need to drop prediction variables for the evaluation dataset. Transform all columns except prediction variables according to the original modeling code, not just the features selected by the best predictive fit. When transforming the training set, don't fill in the scale parameter; when transforming the evaluation set, fill in the scaler parameter trained during training set transformation
4. Then, train the model using the algorithm and selected features from the best predictive fit, make sure to use the complete training dataset for training, no need to split into training and test sets again
5. Finally, evaluate the evaluation dataset using the trained algorithm. The evaluation dataset has no prediction variable labels, this part needs model prediction, and store the predicted values in the Predicted column
6. Generate prediction results file:
   - Save in the predictions folder under the evaluation dataset's directory
      - For example: if evaluation dataset path is /path/to/dataset/
      - Then generated dataset should be saved in /path/to/dataset/predictions/ directory
   - File name format: best_prediction.csv
   - Include all columns from the original dataset
   - Add a new column to store prediction results
7. Store the generated file path in the result variable, call this function directly at the end of the code, don't use if __name__ == "__main__"

Output Data:
```python
[Complete python code]
```
"""