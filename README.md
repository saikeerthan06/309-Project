# EGT 309 Project: EDA and ML Pipeline

## Project Description

### Synopsis
This repository here hosts the entire version control of this project is part of the EGT309 Learning Unit and contributes 40% to the final grade. 
The Version Control involves: 
    1. Performing Data Cleaning & Exploratory Data Analysis (EDA) 
    2. Training 4 different models(Random Forest, LightGBM, XGBoost, CatBoost) & merging them into an ensemble stacked model. 
    3. Building a machine learning pipeline using Kedro, and presenting insights and model performance. 
    4. Lastly containerising the entire pipeline using Docker such that the project can be viewed on localhost.

This project's objectives is to predict the potential repeat buyers through our ensemble stacked model such that Olist will be able to launch targeted marketing campaigns. 

### Objectives
1. Perform EDA and document insights
2. Select and justify machine learning models
3. Build and execute a machine learning pipeline (Kedro) | Containerise Project through Docker
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

### To run the pipeline (terminal)

- git clone https://github.com/saikeerthan06/309-Project.git
- python -m venv venv | python3 -m venv venv (For MacOS)
- source venv/bin/activate   # on macOS/Linux
- .\venv\Scripts\activate    # on Windows
- bash run.sh

**To Use Docker and Containerise the pipeline**:
1. docker build -t (name) .
2. docker run -d -p 5050:5050 (name)
3. Go to browser and type in "localhost:5050" and the entire pipeline should show up in a Jupyter environment.

## Pipeline Flow

- Data cleaning for all raw datasets

- Merging into a master dataset

- Feature engineering and target creation

- Data Splitting
    - Stratified split into train, validation, and test sets.
- Train Data Balancing
    - Apply partial SMOTE (sampling_strategy=0.5) to balance the training set only.
- Train Base Models
    - Use 5-fold StratifiedKFold cross-validation to train:
        - XGBoost
        - CatBoost
        - LightGBM
        - Random Forest
        - Balanced Bagging Classifier (with RandomForest)
    - Collect out-of-fold predictions for meta-learner.
- Train Meta-Learner
    - Use XGBoost as meta-learner on base model out-of-fold predictions.
- Evaluate Stacked Ensemble
    - Generate predictions using stacked ensemble on:
        - Validation set
        - Test set
    - Compute precision, recall, F1-score, and determine optimal threshold.
- Model Saving
    - Save final stacked ensemble model to `saved_model/`.

## Challenges Faced

- Merging of all 9 datasets to form a master dataset table. Some datasets when merged, produced a one-to-many explosion of rows resulting in redundant duplicates (e.g., order_items_dataset, order_payments_dataset). Decided to preserve product-seller granularity to enable accurate future merging of datasets.
- Tackling the Class Imbalance based on our complex data pre-processing logic, despite using multiple different methods to tackle class imbalance like SMOTE + Undersampling/Stacked Model, Model still is not showing perfect results for Class 1, with either the precision or the recall falling behind. 


## Future Improvements

- Add advanced feature engineering and model tuning, including RandomSearchCV/GridSearchCV
- Automate data quality checks
- Deploy model as a REST API (bonus!)
- More fun EDA visualizations

## Credits
### Group Contributions
| Name                   | Student ID         | Contribution(s)                                             |
| ---------------------- | ------------------ |------------------------------------------------------------ |
| Sai Keerthan (Leader)  | 232594T            | Data Pre-Processing(Tackling class imbalance by re-distrubuting class weights, SMOTE + Sampling), Model Development(Trained XGBoost & CatBoost Model, combined all of the 4 models into one stacked model), Oversaw handling of the git repository including gitignore etc. , edited ReadMe.md & Oversaw Containerisation of the pipeline using Docker, edited relevant PowerPoint Slides|
| Leong Jun Ming         | 233079X            | Developed Kedro Pipeline (run.sh,requirements.txt, ReadMe.md, catalog.yml, parameters.yml, nodes.py, pipeline.py), Model Development (Splitting dataset, feature engineering, training of random forest and lightBGM), Slides|               
| Richie Teo Wei Xuan    | 231944N            | Data Preprocessing, EDA Visualizations                      |
| Lee Xiu Wen            | 231867A            | Data Preprocessing, eda.ipynb, ReadME.md, PowerPoint Slides |

### Resources Used
- https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/
- https://www.markdownguide.org/basic-syntax/#lists-1
