# Heart Disease Prediction

A machine learning project that predicts whether a patient is at risk of heart disease from clinical measurements. It covers exploratory data analysis, data cleaning, a comparison of five classification models, and a Streamlit web app that serves the best model.

> **Disclaimer:** This is an educational project and is not a medical diagnostic tool.

## Overview

The goal is to classify patients as having heart disease (1) or not (0) using 11 clinical features. I trained and compared five classifiers on the same train/test split, then saved the best one (KNN) and built a simple web interface for it.

## Dataset

- **File:** `heart.csv`
- **Size:** 918 records, 11 features plus 1 target (`HeartDisease`)
- **Class balance:** about 55% of patients have heart disease
- No null values and no duplicate rows

| Feature | Description |
|---|---|
| `Age` | Age in years |
| `Sex` | M / F |
| `ChestPainType` | ATA, NAP, TA, ASY |
| `RestingBP` | Resting blood pressure (mm Hg) |
| `Cholesterol` | Serum cholesterol (mg/dL) |
| `FastingBS` | Fasting blood sugar > 120 mg/dL (1 = yes, 0 = no) |
| `RestingECG` | Normal, ST, LVH |
| `MaxHR` | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina (Y / N) |
| `Oldpeak` | ST depression induced by exercise |
| `ST_Slope` | Up, Flat, Down |
| `HeartDisease` | Target: 1 = heart disease, 0 = normal |

## Workflow

### 1. Exploratory Data Analysis
- Checked shape, data types, summary statistics, nulls, and duplicates.
- Plotted the target distribution and histograms (with KDE) for `Age`, `RestingBP`, `Cholesterol`, and `MaxHR`.
- Compared `ChestPainType` and `FastingBS` against the target with count plots.
- Used a box plot (Cholesterol by target), a violin plot (Age by target), and a correlation heatmap.

### 2. Data Cleaning
- Found impossible zero values in `Cholesterol` and `RestingBP`, treated them as missing, and replaced them with the mean of the non-zero values.

### 3. Preprocessing
- One-hot encoded categorical features with `pd.get_dummies(drop_first=True)`.
- Split the data 80/20 with stratification on the target (`random_state=42`).
- Standardized features with `StandardScaler`, fitted on the training set only.

### 4. Modeling
Trained five classifiers with scikit-learn and compared them on the held-out test set:

| Model | Accuracy | F1 Score |
|---|---|---|
| **KNN** | **0.8859** | **0.8986** |
| Logistic Regression | 0.8750 | 0.8878 |
| Naive Bayes | 0.8696 | 0.8788 |
| SVM (RBF kernel) | 0.8641 | 0.8804 |
| Decision Tree | 0.7446 | 0.7513 |

KNN performed best, so it was saved with `joblib` along with the scaler and the list of feature columns.

### 5. Web App (Streamlit)
`app.py` loads the saved model, scaler, and column list. The user enters the clinical values in a form, and the app applies the same encoding and scaling as in training and shows a high-risk or low-risk result.

## Project Structure

```
├── HeartdiseaseFinal.ipynb   # EDA, cleaning, modeling
├── app.py                    # Streamlit app
├── heart.csv                 # Dataset
├── knn_heart_model.pkl       # Saved KNN model
├── heart_scaler.pkl          # Saved StandardScaler
├── heart_columns.pkl         # Expected feature columns
└── README.md
```

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Joblib, Streamlit

## How to Run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib streamlit
```

**Run the notebook:** open `HeartdiseaseFinal.ipynb` and run all cells. This regenerates the three `.pkl` files.

**Run the app:**
```bash
streamlit run app.py
```

## Possible Improvements
- Tune hyperparameters (for example the value of `k` in KNN) with `GridSearchCV`.
- Use cross-validation instead of a single train/test split.
- Report precision, recall, and a confusion matrix, since missing a true heart disease case matters more than a false alarm.
- Compare ensemble models such as Random Forest and Gradient Boosting.
- Deploy the app online with Streamlit Community Cloud.

