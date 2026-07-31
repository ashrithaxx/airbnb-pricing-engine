# Airbnb Pricing Engine

## Overview

The Airbnb Pricing Engine is a machine learning-based decision support system designed to estimate optimal listing prices for Airbnb properties. The project applies regression techniques to historical listing data to identify pricing patterns and generate data-driven recommendations based on property characteristics.

The workflow includes data preprocessing, exploratory data analysis, feature engineering, predictive modeling, business intelligence dashboards, and deployment through an interactive Streamlit application.

---

## Objectives

* Develop an accurate machine learning model for Airbnb price prediction.
* Identify the factors that most significantly influence listing prices.
* Provide an interactive pricing recommendation tool for hosts.
* Visualize pricing trends using business intelligence dashboards.

---

## Dataset

The model was trained using a publicly available Airbnb listings dataset containing property characteristics such as:

* Location
* Room type
* Accommodation capacity
* Number of bedrooms and bathrooms
* Availability
* Review statistics
* Host information
* Pricing information

---

## Methodology

### Data Preprocessing

* Removed duplicate records
* Handled missing values
* Encoded categorical variables
* Scaled numerical features
* Removed irrelevant attributes

### Exploratory Data Analysis

Exploratory analysis was performed to understand:

* Price distributions
* Location-wise pricing trends
* Room type popularity
* Correlation between property features and listing price
* Distribution of reviews and availability

### Feature Engineering

Relevant features were extracted and transformed to improve model performance, including:

* Property characteristics
* Host-related features
* Review metrics
* Availability statistics
* Location-based information

---

## Machine Learning Pipeline

The project follows a complete supervised learning workflow:

1. Data preprocessing
2. Feature engineering
3. Train-test split
4. Model training
5. Model evaluation
6. Prediction
7. Deployment

Regression models were evaluated using standard performance metrics including:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## Technology Stack

**Languages**

* Python

**Libraries**

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

**Tools**

* Jupyter Notebook
* Power BI
* Streamlit
* Git

---

## Results

The developed model successfully predicts Airbnb listing prices based on property attributes and provides an interactive interface for estimating recommended prices.

The accompanying Power BI dashboard enables users to explore pricing trends, demand patterns, and location-based insights through interactive visualizations.

---

## Future Work

Potential improvements include:

* Integration of real-time Airbnb market data
* Incorporation of seasonal pricing trends
* Ensemble learning methods such as XGBoost and LightGBM
* Geographic visualization using mapping libraries
* Cloud deployment for scalable access

This repository contains the complete source code, preprocessing pipeline, trained model, Streamlit application, and supporting notebooks used in the development of the Airbnb Pricing Engine.
