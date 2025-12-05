# Titanic Survival Prediction (Machine Learning)

***Project purpose:*** To predict whether each passenger is survived or not from the titanic disaster using different models going through ML workflow

***Project target:*** To find out the best model in predicting survival condition of each passenger


**Model used:** logistic regression, Random Forest 

**Tool used:** Jupyter Notebook, Python, Streamlit

**Imported libraries:** pandas, numpy, matplotlib.pyplot, seaborn, train_test_split, LogisticRegression, accuracy_score, StandardScaler, RandomForestClassifier, cross_val_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score


## Process ##
Data Loading & Exploration ✅

Data Cleaning & Preprocessing ✅

Handling Missing Values ✅

Data Visualization ✅

Feature Engineering (Encoding) ✅

Train-Test Split ✅

Feature Scaling ✅

Model Training ✅

Cross-Validation ✅

Multiple Model Comparison ✅ 

Logistic Regression ✅

Random Forest ✅

Comprehensive Evaluation for BOTH Models ✅

Accuracy Scores ✅

Confusion Matrix ✅

Classification Report (Precision, Recall, F1-Score) ✅

predict probability (how sure model is about this prediction) ✅


## Findings ##
Given a dataset with few data columns such as  age, sex ,survived and more. Number of passengers in the titanic dataset given is 891. 

Titanic survival rate:
342/891 *100%= 38.4%

Titanic missing people rate:
177/891*100%=19.9%

Other missing data columns:
687 cabins and 2 Embarked

Ways to handle data:
-missing columns: replace null values of certain column with that column's mean value or remove the entire column
-data visualisation: plotted plots, histogram and more to visualize relationship between number of people survived/died against number of male/female
-encoding categorical variable/replace categorical value with digit
-train test split(features&label, 20% data to test, 80% to train)
-features scaling

Technique used:
-Two models (logistic regression and random forest)
-5-fold cross validation & hyperparameter tuning
-model evaluation such as confusion matirx , accuracy , f1-score and more.

Comparison between two models:

1. Confusion matrix
-From the view of confusion matrix, randomforest model correctly predicted 86 deaths and wrongly predicted 14 deaths while for survivors it correctly predicted 58 survivors and wrongly predicted 21 survivors, which means it got 144 passengers' survival prediction correct. Note that accuracy for randomforest model is 80.45% out of 179 test passengers, which means randomforest model correctly predicted 144 passengers' survival condition over 179  passengers.  Meanwhile for logistic regression model, its accuracy is 78.77%.

2. Classification report:
-Random Forest achieved 73% recall for survivor prediction
-Superior F1-score (0.77) indicating balanced performance
-Fewer false negatives (21 vs 28) - critical for survival prediction
Overall, the rate of precision , f1-score and recall percentage from randomforest model are higher or equal than logistic regression model.

Conclusion:
By comparing the models (RandomForest & LogisticRegression), RandomForest gives a better accuracy with 80.45% accuracy.


***Project conclusion:*** From the result of model evaluation, random forest is the best model

## Files
- Titanic survival prediction project.ipynb — full analysis + training + evaluation
- app.py — simple script to run predictions
- logistic regression summary.png — model summary screenshot



