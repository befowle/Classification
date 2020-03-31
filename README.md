Classifying Liver Transplant Candidates
-
This project uses the 'Hepatitis C Virus (HCV) for Egyptian Patients Data Set' from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Hepatitis+C+Virus+%28HCV%29+for+Egyptian+patients#)

Objective
-
- To classify patients as candidates for liver transplants based on the progression of Cirrhosis

- Please note that when being considered for a transplant, other factors determine patient candidacy aside from the progression of Cirrhosis, including:
  - ability to tolerate major surgery
  - understanding risk and long-term care/ financial responsibilities
  - no other life-threatening conditions/brain damage that cannot be treated or cured
  - history of medical noncompliance
  - untreated chronic and ongoing drug use, history of serious psychiatric disorders that cannot be treated before transplantation
  - high risk for ongoing or increased severity of the psychiatric disorder


Data Processing
-
- Create dummy variables for categorical features (gender, physical symptoms like headaches, nausea + vomiting, fatigues, aches, jaundice, and pain)

-Create Transplant Class variable using Baseline Histological Staging to determine whether or not a patient is an eligible candidate.


EDA
-

- Examine class balance (insert Class Balance bar plot)

- Check for significant correlations

- Run Principle Component Analysis (insert PCA bar plot)

- Examine Explained Variance (Add explained variance plot)

Classification Model
-

Model 1: K-Nearest Neighbors
- Baseline:
    - Precision: 0.29
    - Recall: 0.17
    - Accuracy: 0.66
    - F1 Score: 0.21
- With optimized Parameters (k = 1), F1 Score improved to 32%

Model 2: Decision Tree
- Cross-Validated Accuracy Score: 72%
  - Depth: 2
  - Feature Importance:
    1) RNA4
    2) AST1
    3) ALT48
    4) RNA EF
    5) WBC
    6) RNA EOT

- Test Accuracy Score: 73%
- Train Accuracy Score: 75%

Model 3: Random Forest
- Bagged Trees with Maximum Depth = 3
  - Training Accuracy: 74%
  - Testing Accuracy: 73%
- Feature Importance:
  1) RNA 4
  2) ALT 48
  3) RNA 12
  4) ALT 4
  5) BMI
  6) Age

- Though accuracy diminished, this model does a fair job identifying patients that are not transplant candidates

Refine Random Forest Model
- Ran GridSearch CV to find optimal hyperparameters
  - Mean Training Score: 79.06%
  - Mean Test Score: 71.47%
  - Best Parameter Combination:
    - criterion: entropy
    - max_depth: 4
    - min_samples_leaf: 2
    - min_samples_split: 5
- Rerun Random Forest with optimized parameters
  - Training Accuracy: 75%
  - Test Accuracy: 73%
  - Feature Importance:
    1) RNA Base
    2) WBC
    3) ALT 24
    4) AST 1
    5) RNA EOT
    6) ALT 4
- Insert Classification Report

Summary
-
- Best model at present: Random Forest
- Future work:
  - Eliminate extraneous features
  - SMOTE to improve class balance between 
  - Try other boosting algorithms (ADAboost, Gradient Boosting)
