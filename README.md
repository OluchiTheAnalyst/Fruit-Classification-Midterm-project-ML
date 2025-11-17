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


