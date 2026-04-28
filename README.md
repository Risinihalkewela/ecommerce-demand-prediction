# Predicting High-Demand Products in E-Commerce 🛒

This project applies **machine learning classification algorithms** to predict high-demand products in e-commerce using customer purchase behavior. It also introduces the **Smart Goods Recommendation System (SGRS)**, an interactive decision-support tool for sellers.

## 📊 Dataset
- **Source:** [Online Retail II Dataset (Kaggle)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data)
- **Records:** 1,048,575 (after preprocessing: 793,309)
- **Period:** Dec 2009 – Dec 2011
- **Features:** Invoice number, stock code, product description, quantity, invoice date, unit price, customer ID, country
- **Engineered Features:** total quantity sold, number of orders, average unit price, total revenue, unique customers, purchase frequency

## ⚙️ Methodology
- **Data Preprocessing:** missing value handling, outlier removal, feature scaling, revenue feature engineering
- **Exploratory Data Analysis (EDA):** histograms, scatter plots, bubble charts, correlation heatmaps
- **Models Implemented:**
  - Logistic Regression
  - Decision Tree
  - Random Forest (final model)

## 🏆 Results
- **Random Forest:** Accuracy = 95.68%, ROC-AUC = 0.993
- Strong discriminative power for imbalanced classes
- High-demand products show distinct patterns: sustained revenue + high customer engagement

## 🚀 Smart Goods Recommendation System (SGRS)
An interactive tool that allows sellers to:
- Apply contextual filters (time period, country, expected unit price, top-N selection)
- Identify products predicted to be in high demand
- Support inventory and promotion strategy decisions

## 🛠️ Tech Stack
- Python, Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Flask / Streamlit (for SGRS app)

✨ *This project bridges predictive analytics with practical e-commerce decision-making.*

✨ *This project bridges predictive analytics with practical e-commerce decision-making.*
