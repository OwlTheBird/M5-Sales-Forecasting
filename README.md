# 🛒 M5 Sales Forecasting with PySpark

![Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white)

> **A distributed machine learning pipeline to forecast daily unit sales for Walmart retail goods using Apache Spark.**

---

## 📖 Overview

This repository hosts a solution for the **M5 Forecasting - Accuracy** competition organized by the Makridakis Open Forecasting Center (MOFC). 

The objective is to estimate point forecasts for unit sales of over 30,000 products sold by Walmart across three US States (California, Texas, and Wisconsin). Given the massive scale and hierarchical nature of the data, this project utilizes **PySpark** to handle data ingestion, feature engineering, and model training efficiently.

## 🎯 The Goal

The challenge is to predict daily sales for the next **28 days** (Forecast horizon F1-F28) for each unique item-store combination.

The data covers stores in:
* **CA** (California)
* **TX** (Texas)
* **WI** (Wisconsin)

Predictions are evaluated on the **Weighted Root Mean Squared Scaled Error (RMSSE)**, which penalizes errors on high-value items and accounts for the historical volatility of the series.

## 🗂️ Dataset

The dataset is hierarchical, including item level, department, product categories, and store details. It consists of three main files:

1.  **`calendar.csv`**: Contains information about the dates on which the products are sold (events, day of the week, SNAP food stamp allowability).
2.  **`sell_prices.csv`**: Contains information about the price of the products sold per store and date.
3.  **`sales_train_evaluation.csv`**: Contains the historical daily unit sales data for each product and store (d_1 - d_1941).

*Note: Data is provided by Walmart and the MOFC.*

## ⚙️ Technical Approach

Due to the size of the dataset (millions of rows when melted), traditional pandas-based approaches often hit memory limits. This project uses **PySpark** to:

1.  **Melt & Transform**: Convert wide-format time series (d_1...d_1941) into a long-format transactional dataset.
2.  **Distributed Feature Engineering**:
    * **Lag Features**: Sales from t-7, t-28, etc.
    * **Rolling Statistics**: Moving averages and standard deviations over various windows.
    * **Encoding**: Indexing categorical variables (State, Store, Category) efficiently.
3.  **Model Training**: Using Spark MLlib (or Spark-wrapped XGBoost/LightGBM) to train regressors on the distributed clusters.

## 📉 Evaluation Metric

The competition uses **Weighted RMSSE**.

$$
RMSSE = \sqrt{\frac{1}{n} \sum_{t=n+1}^{n+h} \frac{(Y_t - \hat{Y}_t)^2}{\frac{1}{n-1} \sum_{i=2}^{n} (Y_i - Y_{i-1})^2}}
$$

Where $Y_t$ is the actual sale, $\hat{Y}_t$ is the forecast, and the denominator represents the scaling factor based on the historical differenced variance.

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Apache Spark 3.0+
* Java 8 or 11

### Installation

Clone the repository:
```bash
git clone https://github.com/OwlTheBird/M5-Sales-Forecasting
cd m5-forecasting-pyspark
