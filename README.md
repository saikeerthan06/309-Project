# EGT 309 Project: EDA and ML Pipeline

## Project Description

### Synopsis
This project is part of the EGT309 Learning Unit and contributes 40% to the final grade. It involves performing Exploratory Data Analysis (EDA), building a machine learning pipeline using Kedro, and presenting insights and model performance.

### Objectives
1. Perform EDA and document insights
2. Select and justify machine learning models
3. Build and execute a machine learning pipeline (Kedro)
4. Collaborate using version control
5. Present and explain code and results effectively

---

## Project Structure

project-root/
│
├── src/ # Source code for Kedro pipeline and nodes
├── saved_model/ # Trained model(s) directory
│ └── stacked_ensemble_model.pkl
├── data/ # Raw, cleaned, and intermediate data
├── eda.ipynb # Jupyter notebook for EDA
├── eda.pdf # Exported PDF version of EDA
├── requirements.txt # Python dependencies
├── run.sh # Script to execute full pipeline
├── README.md # Project overview and instructions


---

## Outputs

- The main machine learning model is saved as `saved_model/stacked_ensemble_model.pkl` after running the pipeline.
- All intermediate and cleaned data are saved in the `data/` directory.

---

## Key EDA Findings

*(Summarize important findings and trends from your EDA, e.g. top-selling categories, repeat buyers’ behaviors, or interesting outliers.)*

---

## Pipeline Flow

- Data cleaning for all raw datasets
- Merging into a master dataset
- Feature engineering and target creation
- Data splitting (train, validation, test)
- Model training: RandomForest, XGBoost, Logistic Regression (with SMOTE)
- Evaluation on validation and test sets
- Model stacking/ensemble with Logistic Regression as meta-learner
- Model saved to `saved_model/`

---

## Explanation of Machine Learning Models

- **Random Forest:** Used for baseline classification of repeat buyers.
- **XGBoost:** Gradient boosting method for higher performance.
- **Logistic Regression (with SMOTE):** Handles class imbalance.
- **Stacked Ensemble:** Combines predictions from XGBoost and Logistic Regression using a meta-logistic regression model.

---

## Evaluation of Developed Models

*(Add your summary metrics/insights here. Example:)*

| Model              | Validation F1 | Test F1 | Notes                                  |
|--------------------|:-------------:|:-------:|----------------------------------------|
| Random Forest      | 0.68          | 0.67    | Good baseline                          |
| XGBoost            | 0.70          | 0.69    | Slightly better than Random Forest      |
| Logistic Regression (SMOTE) | 0.67 | 0.66    | Handles imbalance well                 |
| **Stacked Ensemble**| **0.72**     | **0.71**| Best overall, selected for deployment  |

---

## Challenges Faced

- Merging of all 9 datasets to form a master dataset table. Some datasets when merged, produced a one-to-many explosion of rows resulting in redundant duplicates (e.g., order_items_dataset, order_payments_dataset). Decided to preserve product-seller granularity to enable accurate future merging of datasets.

---

## Future Improvements

- Add advanced feature engineering and model tuning
- Automate data quality checks
- Deploy model as a REST API (bonus!)
- More fun EDA visualizations

---

## How to Install and Run the Project

### Dependencies
1. Python 3.8+
2. (Optional) Docker Desktop
3. Jupyter Lab

### Set up environment (terminal)
```bash
git clone https://github.com/saikeerthan06/309-Project.git
cd 309-Project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### To run the EDA notebook (terminal)
```
jupyter lab eda.ipynb
```
### To run the pipeline (terminal)
```
bash run.sh

## Credits
### Group Contributions
| Name                   | Student ID         | Contribution(s)                                             |
| ---------------------- | ------------------ |------------------------------------------------------------ |
| Sai Keerthan (Leader)  | 232594T            | Model & Pipeline Developer                                  |
| Leong Jun Ming         | 233079X            | Model, Kedro Pipeline , ReadME.md Developer                                  |               
| Richie Teo Wei Xuan    | 231944N            | Data Preprocessing, EDA Visualizations                      |
| Lee Xiu Wen            | 231867A            | Data Preprocessing, eda.ipynb, ReadME.md, PowerPoint Slides |

### Resources Used
- https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/
- https://www.markdownguide.org/basic-syntax/#lists-1