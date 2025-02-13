# Dataset Analysis Tool

## Overview

The **Dataset Analysis Tool** is a comprehensive Python application that automates key phases of the data science workflow. It provides interactive data loading, cleaning, exploratory analysis, visualization (including AI-assisted recommendations), categorical encoding, model training with AutoML, evaluation, and automated report generation—all in one tool.

This repository contains the source code and documentation needed to set up and use the tool effectively.

**(Please Create a huggingface inference API user access token and put it in the line 25 of dataset_analysis_tool.py for the tool to work)**

## Features

- **Data Loading:** Load datasets (CSV files) interactively.
- **Data Exploration:** View dataset information, head (first five rows), and summary statistics.
- **Data Cleaning:** Automatically handle missing values and remove outliers using the IQR method.
- **AI-Assisted Visualization:** Get visualization recommendations based on an AI model (using the Hugging Face Inference API).
- **Categorical Encoding:** Choose between Label Encoding and One-Hot Encoding for categorical columns.
- **Model Training and Evaluation:** Train an automated model (using TPOT) as well as a baseline decision tree model, with evaluation metrics such as accuracy, RMSE, and R².
- **Report Generation:** Generate a detailed analysis report covering data quality, exploration, model training, evaluation, and conclusions.
- **MLflow Integration:** Log model parameters, metrics, and artifacts for reproducibility.

## Prerequisites

- **Python:** Version 3.9 or above.
- **Required Libraries:**  
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `TPOT`
  - `mlflow`
  - `huggingface_hub`
  - Plus other standard libraries (e.g., `json`, `textwrap`, `warnings`)

You can install all dependencies using pip:

        pip install textwrap pandas numpy matplotlib seaborn json5 regex huggingface_hub tpot scikit-learn mlflow


# Usage Instructions

Once installed, run the tool via the command line. The tool provides an interactive menu that guides you through each step.

To Run the Application: 

      python data_analysis_tool.py

### Interactive Menu Options:

####After starting the application, you will see a menu with the following options:


  ##### 1. Load Dataset:
  
  What It Does: Prompts you to enter the file path of your CSV dataset and loads the data.
  
  User Action: Provide the full path to your CSV file (e.g., /path/to/data.csv).

  ##### 2. Explore Dataset:
  
  What It Does: Displays basic information about the dataset, including its structure, first five rows, and summary statistics.
  
  User Action: Simply review the output in the terminal.

  ##### 3. Clean Dataset:
  
  What It Does: Automatically handles missing values by dropping or filling them and removes outliers using the IQR method.
  
  User Action: Observe the cleaning messages printed to the console.

  ##### 4. Determine Target Variable:
  
  What It Does: Lists the columns in your dataset and asks you to select one as the target variable for model training.
  
  User Action: Enter the name of the target variable exactly as it appears in the dataset.

  ##### 5. Analyze Dataset:
  
  What It Does: Generates an AI-driven analysis prompt to recommend interesting visualizations based on relationships between dataset columns.
  
  User Action: Review the generated visualizations and recommendations.

  ##### 6. Encode Categorical Columns:
  
  What It Does: Detects categorical columns and prompts you to choose between Label Encoding and One-Hot Encoding.
  
  User Action: For each detected categorical column, enter your choice (1 for Label Encoding, 2 for One-Hot Encoding).

  ##### 7. Train Models and Evaluate:
  
  What It Does: Splits the data into training and testing sets, automatically trains an AutoML model (using TPOT) and a baseline decision tree model, then evaluates the performance.
  
  User Action: Review the model performance metrics printed to the console (e.g., accuracy, RMSE, R²).

  ##### 8. Generate Report:
  
  What It Does: Compiles the analysis, visualizations, and model results into a detailed report with sections (Introduction, Data Quality Checks, Data Exploration, Training a Model, Model Evaluation, Conclusion)
  and saves it to a text file.
  
  User Action: Open the generated dataset_analysis_report.txt to view the full report.

  ##### 9. Exit:
  
  What It Does: Exits the application.
  
  User Action: Select this option when you are finished.

  

### Workflow Example:

  - **Step 1:** Start by loading your dataset (option 1).
  - **Step 2:** Explore your dataset (option 2) to understand its structure.
  - **Step 3:** Clean the dataset (option 3) to handle missing values and outliers.
  - **Step 4:** Select the target variable (option 4) for model training.
  - **Step 5:** Analyze relationships and get visualization recommendations (option 5).
  - **Step 6:** Encode any categorical features as needed (option 6).
  - **Step 7:** Train and evaluate your models (option 7).
  - **Step 8:** Generate and review a comprehensive report (option 8).


Contributions to the Dataset Analysis Tool are welcome. If you have suggestions or improvements, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or issues, please contact [lukamarg@gmail.com].


