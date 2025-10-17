Overview
This repository implements a complete machine learning pipeline for classifying Iris flower species using the famous Iris dataset. It includes scripts for model training, prediction, testing, and CI/CD integration using DVC and GitHub Actions.

Directory Structure
text
├── .dvc/                     # DVC metadata and tracking directory  
├── .dvcignore                # Files and folders ignored by DVC  
├── .gitignore                # Git ignore configuration  
├── .github/
│   └── workflows/
│       └── ci.yml            # Continuous integration workflow definition  
├── config/                   # Configuration files for data paths, hyperparameters, etc.  
├── data/
│   └── iris.csv              # Original dataset file  
├── env/                      # Virtual environment folder  
│   ├── bin/                  # Executable scripts  
│   ├── lib/python3.11/site-packages/  # Installed dependencies  
│   ├── lib64/  
│   └── pyvenv.cfg  
├── model/
│   └── model.pkl             # Trained model file  
├── src/
│   ├── train.py              # Model training script  
│   ├── predict.py            # Model inference script  
│   └── __pycache__/          # Cached bytecode files  
├── tests/
│   ├── test_validation.py    # Input data validation tests  
│   ├── test_evaluation.py    # Model performance evaluation tests  
├── dvc.yaml                  # DVC pipeline definition  
└── README.md                 # Project documentation  
How to Run
Create Virtual Environment

bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
Reproduce Pipeline with DVC

bash
dvc repro
Train the Model

bash
python src/train.py
Make Predictions

bash
python src/predict.py --input data/iris.csv --output predictions.csv
Run Tests

bash
pytest tests/
CI/CD
ci.yml file defines automatic testing and validation using GitHub Actions.

On each commit, workflows ensure model reproducibility and data consistency.

Requirements
Python 3.11+

DVC

scikit-learn

pandas

pytest
