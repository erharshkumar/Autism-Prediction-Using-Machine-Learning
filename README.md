## Project Overview

Autism Spectrum Disorder (ASD) is a neurodevelopmental condition that affects communication, social interaction, and behavior. Early identification of ASD plays a crucial role in providing timely intervention and improving long-term outcomes. Manual diagnosis often requires experienced clinicians and can be time-consuming, making automated screening tools valuable for supporting healthcare professionals.

This project develops an advanced machine learning pipeline capable of predicting the likelihood of Autism Spectrum Disorder using behavioral and demographic screening data. The project follows an end-to-end machine learning workflow, from data preprocessing to model deployment, emphasizing both predictive performance and real-world usability.

### Problem Statement

The objective of this project is to build a highly accurate and reliable machine learning model that can classify whether an individual is likely to have Autism Spectrum Disorder based on screening questionnaire responses and demographic information.

The project aims to:
- Improve early ASD screening
- Reduce diagnostic delays
- Support healthcare professionals with data-driven insights
- Create a deployable prediction system for practical applications

### Objectives
The primary objectives of this project are:
  - Perform comprehensive exploratory data analysis.
  - Clean and preprocess the dataset.
  - Handle missing values and duplicate records.
  - Encode categorical features.
  - Scale numerical variables.
  - Handle class imbalance using SMOTE.
  - Select the most informative features.
  - Train multiple machine learning models.
  - Optimize model performance using hyperparameter tuning.
  - Build an ensemble learning model.
  - Evaluate models using multiple performance metrics.
  - Optimize prediction threshold.
  - Calibrate predicted probabilities.
  - Save the trained model for deployment.
  - Develop a Streamlit application for real-world predictions.

### Dataset Overview
The dataset contains demographic, medical, and behavioral screening information collected from individuals undergoing Autism Spectrum Disorder screening.
#### Dataset Features
Typical attributes include:
  - Age
  - Gender
  - Ethnicity
  - Country of residence
  - Family history of autism
  - Jaundice at birth
  - Previous ASD diagnosis
  - Behavioral screening questionnaire scores (A1–A10)
  - Screening score
  - Age category
  - Relationship
  - Class label (ASD or Non-ASD)

The target variable predicts whether an individual has Autism Spectrum Disorder.

### Project Workflow
1. Data Collection
- Imported the autism screening dataset.
= Loaded data using Pandas.

2. Data Exploration
Performed:
  - Dataset inspection
  - Shape analysis
  - Data types
  - Missing value analysis
  - Duplicate detection
  - Statistical summary
  - Target class distribution

3. Exploratory Data Analysis (EDA)
Visualizations included:
  - Count plots
  - Histograms
  - Box plots
  - Correlation heatmap
  - Pair plots
  - Class distribution plots
  - Feature importance visualization
EDA helped identify:
  - Feature distributions
  - Outliers
  - Relationships among variables
  - Correlations
  - Class imbalance

4. Data Preprocessing
Performed several preprocessing steps:
- Missing Value Treatment
    - Filled missing values appropriately.
- Duplicate Removal
    - Removed duplicate records.
Encoding
- Applied encoding techniques for categorical variables.
Feature Scaling
- Used StandardScaler to normalize numerical variables.

5. Train-Test Split
The dataset was divided into:
  - Training Set
  - Testing Set
using stratified sampling to preserve class distribution.

6. Handling Class Imbalance
Applied:
- SMOTE (Synthetic Minority Over-sampling Technique)
Benefits:
  - Balanced class distribution
  - Reduced prediction bias
  - Improved recall for minority class
 
### Feature Engineering
Performed feature engineering to improve model performance.
This included:
  - Encoding
  - Scaling
  - Feature selection
  - Data transformation

### Machine Learning Models
Several algorithms were trained and compared.
#### Decision Tree
Advantages:
  - Easy interpretation
  - Fast training
  - Baseline model

#### Random Forest
Advantages:
  - Reduced overfitting
  - Better generalization
  - High robustness

#### XGBoost
Advantages:
  - Gradient boosting
  - Excellent predictive performance
  - Handles complex relationships efficiently

#### Stacking Ensemble (Final Model)
Base learners:
  - Decision Tree
  - Random Forest
  - XGBoost
Meta learner:
- Logistic Regression
The stacking model combines predictions from multiple algorithms to improve accuracy and stability.

### Hyperparameter Tuning
Optimized models using:
  - RandomizedSearchCV
  - Optuna
Parameters tuned included:
  - Number of estimators
  - Maximum depth
  - Learning rate
  - Minimum samples split
  - Minimum samples leaf
  - Subsample ratio
  - Column sampling
This significantly improved model performance.

### Model Evaluation
Models were evaluated using multiple metrics instead of relying only on accuracy.
Evaluation metrics included:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC Score
  - Sensitivity
  - Specificity

### Performance Visualization
Generated professional visualizations including:
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
  - Calibration Curve
  - Probability Distribution Plot
  - Feature Importance Plot
These plots provide detailed insights into model behavior.

### Threshold Optimization
Instead of using the default probability threshold (0.50), an optimal threshold was selected using the Precision-Recall Curve.
Benefits:
  - Higher sensitivity
  - Better ASD detection
  - Improved clinical usefulness

### Probability Calibration
Used calibration techniques to improve predicted probabilities.
Benefits:
  - More reliable confidence scores
  - Better decision-making support
  - Improved probability estimation

### Model Saving
Saved the trained model using Pickle.
Generated deployment artifacts including:
  - Trained Pipeline (.pkl)
  - Threshold Configuration (.json)
  - Model Card
  - Streamlit Application

### Deployment
Prepared the project for deployment using Streamlit.
The web application allows users to:
  - Enter patient information
  - Receive ASD prediction
  - View prediction probability
  - Support screening decisions

### Technologies Used
  - Programming Language - Python
  - Data Analysis - Pandas, NumPy
  - Data Visualization - Matplotlib, Seaborn
  - Machine Learning - Scikit-learn, XGBoost, Imbalanced-learn (SMOTE), Optuna
  - Model Deployment - Pickle, JSON, Streamlit

### Skills Demonstrated
This project demonstrates practical experience in:
  - Data Cleaning
  - Exploratory Data Analysis (EDA)
  - Feature Engineering
  - Data Visualization
  - Feature Scaling
  - Encoding Techniques
  - Handling Imbalanced Data
  - SMOTE
  - Machine Learning
  - Ensemble Learning
  - Hyperparameter Optimization
  - RandomizedSearchCV
  - Optuna Optimization
  - Model Evaluation
  - ROC-AUC Analysis
  - Precision-Recall Analysis
  - Probability Calibration
  - Threshold Optimization
  - Model Serialization
  - Deployment Preparation
  - Streamlit Development
  - End-to-End Machine Learning Pipeline

### Key Outcomes
- Developed a complete end-to-end machine learning pipeline for Autism Spectrum Disorder prediction.
- Improved predictive performance through hyperparameter tuning and ensemble learning.
- Addressed class imbalance using SMOTE to enhance minority class detection.
- Optimized classification thresholds to improve sensitivity and screening effectiveness.
- Evaluated models using comprehensive performance metrics and visualizations.
- Prepared the trained model for deployment with a reusable pipeline, configuration files, and a Streamlit application.

## Conclusion

This project demonstrates how advanced machine learning techniques can support the early screening of Autism Spectrum Disorder through accurate and data-driven predictions. By combining robust preprocessing, feature engineering, class imbalance handling, hyperparameter optimization, ensemble learning, probability calibration, and deployment preparation, the project delivers a practical and scalable prediction system.

While the model is not intended to replace professional medical diagnosis, it can serve as a valuable clinical decision-support tool by helping identify individuals who may benefit from further evaluation. Overall, this project showcases a complete end-to-end machine learning workflow and highlights the potential of AI in improving healthcare screening and early intervention.
