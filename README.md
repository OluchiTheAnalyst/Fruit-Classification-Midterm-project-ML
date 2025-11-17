### Note From the Author
This is my first ML model, built and deployed from scratch! This project was built with intention, to show that learning, consistency, and curiosity can turn complex ideas into something practical. I am leaving this note here because I want to comeback in the nearest future to see and remember where I started from and where I will be then. If you’re reading this, keep building, keep experimenting, and never underestimate what you can create from where you are. Thanks for stopping by !

## Problem Statement
The goal of this project is to build a machine learning model that classifies fruits based on their physical and sensory attributes. Using data on weight, size, color, shape, average price, and taste, the system predicts the correct fruit category. The model is then served as an API for real-time classification requests.

## Dataset Description
The dataset used contains information on various fruit samples.
Features include:
- `size (cm)` — diameter or length of the fruit
- `shape` — categorical (round, oval, irregular)
- `weight (g)` — fruit weight in grams
- `avg_price` — average retail price in local currency
- `color` — categorical (green, red, yellow, etc.)
- `taste` — categorical (sweet, sour, bland, etc.)

Target: `fruit_name` — the type of fruit (e.g., apple, mango, banana, orange).

## Exploratory Data Analysis
Here are some of the steps I took while exploring the data set;
- Checked for missing and duplicate values.
- Converted timestamp and numeric formats properly.
- Encoded categorical variables using `DictVectorizer`.
- Visualized class distribution and correlations between numeric features.
- Observed that weight and color were strong predictors of fruit type.

## Data Modeling
I trainned the data using 3-4 models then selected the best
- Model: `XGBoost Classifier` (after testing Random Forest)
- Label encoding for target variable (fruit_name)
- Evaluation metric: Accuracy and F1-score
- Achieved around 90% accuracy on test data
- Serialized trained model and vectorizer into `model.bin` using pickle

## How to run the codes
```
# Train model
python train.py

# Run prediction from script
python predict.py

```
## How to run the API
```
uvicorn serve:app --host 0.0.0.0 --port 9696
```
then open `http://localhost:9696` via web browser

## Run via Docker
```
docker build -t fruit-classification-api .
docker run -it -p 9696:9696 fruit-classification-api
```

## Limitations
- Dataset is small, too simple and limited in variety.
- Could expand with new features (texture, ripeness, or aroma).
- Plan to enhance model interpretability and retrain with larger data.
- Most of this cloud services are paid services, so i couldn't deploy for free.

## Project Architecture
```
Data → Exploratory Data Analysis → Feature Engineering → Model Training → Hyper parameter tuning → Model Retrain → FastAPI Service → Docker 

```



