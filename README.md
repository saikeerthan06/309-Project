# EGT 309 Project

## Overview
This project is part of the EGT309 Learning Unit and contributes 40% to the final grade. It involves performing Exploratory Data Analysis (EDA), building a machine learning pipeline, and presenting insights and model performance.

## Objectives
1) Perform EDA and document insights
2) Select and justify machine learning models
3) Build and execute a machine learning pipeline
4) Collaborate using version control
5) Present and explain code and results effectively

## Project Structure
project-root/
│
├── src/                   # Source code for pipeline
├── saved_model/           # Trained model(s) directory
├── eda.ipynb              # Jupyter notebook for EDA
├── eda.pdf                # Exported PDF version of EDA
├── requirements.txt       # Python dependencies
├── run.sh                 # Script to execute pipeline
├── README.md              # Project overview and instructions

## Installation 
1) Python 3.8+
2) Docker (if containerized)
3) Jupyter Lab (for EDA)

### Set up environment (bash)
git clone https://github.com/<your-team>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### To run the EDA notebook (bash)
jupyter lab eda.ipynb

### To run the pipeline (bash)
bash run.sh

## Team Members & Contributions
| Name     | Contribution                             |
| -------- | ---------------------------------------- |
| Member 1 | Data Cleaning, Feature Engineering       |
| Member 2 | Model Training, Hyperparameter Tuning    |
| Member 3 | EDA Visualizations, Pipeline Integration |
| Member 4 | Inference, Report Compilation            |

