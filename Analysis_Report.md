# Big Data Project: Sales Trend Analysis Report

## 1. Executive Summary
This project analyzes the `train.parquet` dataset to identify sales trends, seasonal cycles, and the impact of external factors like price and SNAP benefits.

### Key Findings

#### A. Global Trend (The "Pulse")
*   **Observation:** The daily sales aggregation reveals a consistent "heartbeat" of the business.
*   **Insight:** Sales drop to zero every Christmas. There is a general upward trend in sales volume over the 5-year period, indicating business growth.

#### B. Seasonality
*   **Weekly:** Sales peak significantly on weekends (Saturday/Sunday), indicating a strong weekly shopping cycle.
*   **Yearly:** Sales volume varies by month, with noticeable peaks likely driven by summer activities and holiday seasons.

#### C. External Factors
*   **Price Elasticity:** There is a clear inverse relationship between price and sales volume. Higher sales are concentrated at lower price points, and distinct pricing tiers (e.g., $X.99) are visible.
*   **SNAP Effect:** Analysis of the `FOODS` category in `CA` shows a measurable increase in average sales on days when SNAP benefits are active, quantifying the impact of government assistance on revenue.

---

## 2. Technical Implementation
The analysis is structured into two separate Jupyter Notebooks for modularity and clarity.

### Notebook 1: Time Series Analysis
**File:** `analysis_notebook.ipynb`
*   **Focus:** Temporal patterns (Trends, Cycles, Seasonality).
*   **Methodology:**
    *   Aggregates 58M+ rows using **DuckDB** for performance.
    *   **Global Trend:** Groups by day number (`d_x`) to show the full timeline.
    *   **Weekly Seasonality:** Groups by weekday (`wday`) to show average daily performance.
    *   **Yearly Seasonality:** Groups by month to show annual cycles.

### Notebook 2: Bivariate Analysis
**File:** `bivariate_analysis.ipynb`
*   **Focus:** Relationship between Sales and External Variables.
*   **Methodology:**
    *   **Price Elasticity:** Uses a **1% random sample** of the `FOODS_3` department to visualize the Price vs. Sales scatter plot without overplotting.
    *   **SNAP Effect:** Compares average sales on SNAP vs. Non-SNAP days for the `FOODS` category in `CA`.

## 3. Tools Used
*   **Language:** Python
*   **Data Engine:** DuckDB (for efficient SQL querying of Parquet files)
*   **Visualization:** Matplotlib & Seaborn
*   **Environment:** Jupyter Notebook
