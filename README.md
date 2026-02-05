# scowtt-ml-assessment
## ML Engineer Assessment

### Project Overview
This project focuses on identifying high-value customers from the Olist E-Commerce dataset. The primary objective was to build a machine learning pipeline capable of predicting which customers are likely to make a repeat purchase in the near future. This helps in targeting marketing efforts towards users with the highest propensity to convert.

### Methodology
I approached this problem by treating it as a binary classification task (Will the user buy again?) combined with a regression task (How much will they spend?).

#### Data Processing
I used the Olist dataset, which contains information on approximately 100k orders from 2016 to 2018. I cleaned the data by aggregating individual orders into a "Single User View", ensuring each row represents a unique customer with their entire history. I focused the training on a stable "healthy" period of data (up to June 2018) to avoid data quality issues and inconsistent reporting found in the final months of the dataset.

#### Feature Engineering
I engineered features based on Recency, Frequency, and Monetary (RFM) analysis, which are standard validation metrics in retail. I also included behavioral features such as average review scores, preferred payment methods, and delivery performance metrics to capture user satisfaction.

#### Modeling
I dealt with a significant class imbalance (very few repeat buyers compared to one-time buyers) by using **Class Weighting** and **SMOTE** (Synthetic Minority Over-sampling Technique). I experimented with several algorithms including Logistic Regression, Random Forest, and Gradient Boosting to find the best balance of precision and recall.

### Technology Stack
*   **Language**: Python
*   **Data Manipulation**: pandas, numpy
*   **Machine Learning**: scikit-learn (LogisticRegression, RandomForestClassifier), imbalanced-learn (SMOTE)
*   **Visualization**: matplotlib, seaborn

### Results
My final model achieved an **AUC of 0.71**. In practical business terms, the top 10% of users identified by the model are **2.88 times more likely** to make a repeat purchase than the average user. This demonstrates that the model successfully captures signals of customer loyalty despite the sparseness of the data.

## How to Run
To reproduce the analysis:
1.  Ensure you have the Olist dataset CSV files in a `datasets/` folder.
2.  Install the required packages: `pandas`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn`.
3.  Run the ipynb script: `scowtt-ml-assessment.ipynb`

This will generate the user rankings in `user_propensity_scores.csv` and performance plots in `model_plots.png`.
