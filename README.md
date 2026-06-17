# Healthcare Premium Prediction

A machine learning web application that predicts a person's **annual health insurance cost** based on demographic, lifestyle, and medical-history inputs. Built with a **segmented two-model approach** (Linear Regression + XGBoost) and a Streamlit front end.

> This project was built as a hands-on implementation while following the **codebasics ML course**. I used it to learn regression modelling, customer segmentation, risk-score engineering, and ML app deployment.

---

## Demo

<!-- TODO: Add a screenshot of the running Streamlit app here, e.g.:
![App Screenshot](app_screenshot.png)
Run the app, take a screenshot, save it in the repo, and update this line. -->

*Screenshot coming soon.*

---

## What it does

The user enters details — age, dependants, income, genetic risk, insurance plan, employment, gender, marital status, BMI category, smoking status, region, and medical history. The app then:

1. Encodes categorical inputs and computes a **normalized medical risk score** from the applicant's conditions.
2. **Segments the customer by age** and routes them to the appropriate model and scaler.
3. Predicts the estimated insurance cost.

## Key design choice: segmented modelling

Rather than using one model for everyone, this project splits customers into two groups, each served by a different model:

| Segment | Model | Why |
|---|---|---|
| **Young** (age ≤ 25) | Linear Regression | Simpler relationship in this segment |
| **Rest** (age > 25) | XGBoost Regressor | Captures more complex, non-linear patterns |

Each segment has its **own fitted scaler**, so preprocessing matches the model it feeds. This is a practical example of improving accuracy by tailoring models to sub-populations rather than forcing a single model to fit everyone.

## Tech stack

- **Python**
- **scikit-learn** — Linear Regression, scaling
- **XGBoost** — gradient-boosted regression for the main segment
- **pandas** — data handling and feature encoding
- **Streamlit** — interactive web UI
- **joblib** — model serialization

## Feature engineering highlight

A **normalized risk score** is computed from the applicant's medical history. Conditions are assigned weights (e.g. heart disease = 8, diabetes = 6, high blood pressure = 6, thyroid = 5) and combinations are summed, then normalized to a 0–1 range so the model receives a single consistent risk signal.

## Project structure

```
Healthcare-Premium-Prediction/
├── main.py                     # Streamlit app (UI + input handling)
├── prediction_helper.py        # Encoding, risk scoring, segmented prediction
├── artifacts/
│   ├── model_young.joblib      # Linear Regression (age ≤ 25)
│   ├── model_rest.joblib       # XGBoost (age > 25)
│   ├── scaler_young.joblib     # Scaler for young segment
│   └── scaler_rest.joblib      # Scaler for rest segment
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup and run

1. **Clone the repository**
   ```bash
   git clone https://github.com/omshivbhosale/Healthcare-Premium-Prediction.git
   cd Healthcare-Premium-Prediction
   ```

2. **(Recommended) Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run main.py
   ```
   The app opens in your browser at `http://localhost:8501`.

## What I learned

- Building regression models for a real-world cost-prediction problem
- **Customer segmentation**: routing different sub-populations to different models for better accuracy
- Engineering a normalized risk score from categorical medical history
- Keeping preprocessing consistent between training and inference using saved scalers
- Comparing a linear model against a gradient-boosted model (XGBoost)

## Acknowledgements

Built by following the **codebasics** machine learning course (codebasics.io). The dataset and core project concept are from the course; this repository is my own implementation.
