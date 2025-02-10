# Dataset Analysis Tool

## Overview

The **Dataset Analysis Tool** is a comprehensive Python application that automates key phases of the data science workflow. It provides interactive data loading, cleaning, exploratory analysis, visualization (including AI-assisted recommendations), categorical encoding, model training with AutoML, evaluation, and automated report generation—all in one tool.

This repository contains the source code and documentation needed to set up and use the tool effectively.

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

- **Python:** Version 3.7 or above.
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

You can install all dependencies using pip.


