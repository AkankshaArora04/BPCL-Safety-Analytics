# BPCL Safety Violation Intelligence System

A machine learning project built on real safety alert data from BPCL (Bharat Petroleum) units across India. The goal was to improve upon their existing violation detection system which had around 60% accuracy.

## What this project does

BPCL plants have CCTV cameras that detect safety violations — like workers not wearing helmets, safety belts, fire hazards etc. Each violation triggers an alert. This project takes 3 years of those real alerts (42,000+ records from 56 units) and trains ML models to automatically classify what type of violation occurred, faster and more accurately than before.

## Models Used

| Model | Accuracy |
|---|---|
| Random Forest | ~75% |
| XGBoost | ~74% |
| Logistic Regression | ~72% |
| Linear SVM | ~71% |

Previous baseline: ~60%

## Features

- Interactive dashboard built with Streamlit
- Trend analysis across 2021-2024
- Unit-wise and hourly violation breakdown
- Live prediction — input alert details, get violation type

## Tech Stack

Python, Pandas, Scikit-learn, XGBoost, Streamlit, Matplotlib, Seaborn

## Future Scope

- YOLO-based model using the actual CCTV video links present in the dataset
- Deploy on Streamlit Cloud
- Worker-level violation tracking