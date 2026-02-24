# 🏠 Outlier Analysis & Impact on Linear Regression — King County House Prices

A data analysis notebook that investigates how outliers affect machine learning model performance. Using the King County House Sales dataset, three progressive outlier filtering strategies are compared to demonstrate the real impact of data cleaning on regression accuracy.

---

## Overview

Outliers are data points that deviate significantly from the rest of the dataset. In machine learning, they can distort model training, inflate error metrics, and create misleading evaluation scores. This notebook provides a hands-on comparison of model performance before and after outlier removal, and examines a counterintuitive phenomenon: why R² can sometimes *decrease* after cleaning data.

---

## What's Inside

**1. Exploratory Data Analysis**
Distribution analysis of key features (`price`, `bedrooms`, `sqft_living`) using histograms and KDE plots. Log-scale transformations applied to handle right-skewed distributions.

**2. Baseline Model (with outliers)**
Linear regression trained on raw data including extreme values. Establishes a performance baseline for comparison.

**3. Progressive Outlier Filtering**
Three filtering strategies applied sequentially:

| Strategy | Filter Criteria | RMSE | MAE |
|---|---|---|---|
| No filtering (raw) | — | 212,539 | 127,493 |
| Moderate filtering | bedrooms ≤ 10, sqft ≤ 10,000, price ≤ $10M | 204,078 | 124,563 |
| Strict filtering | bedrooms ≤ 8, sqft ≤ 8,000, price ≤ $5M | 189,903 | 122,404 |

Each filtering step reduces both RMSE and MAE, confirming that extreme values harm prediction accuracy.

**4. The R² Paradox**
A key finding: R² score can appear to *decrease* after outlier removal, even when the model genuinely improves. This happens because:
- Outliers create high variance in the target variable
- A model that fits those outliers appears to "explain" more variance
- This is a form of false accuracy — the model is memorizing extreme values, not learning the underlying pattern
- After cleaning, the data becomes more homogeneous and true model quality becomes visible through MAE/RMSE

**5. Comparative Error Analysis**
Side-by-side comparison of all three scenarios using RMSE and MAE, with written interpretation of results.

---

## Dataset

**King County House Sales Dataset**

Download from Kaggle: [https://www.kaggle.com/datasets/harlfoxem/housesalesprediction](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)

After downloading, update the file path in the notebook:

```python
# Local environment
df = pd.read_csv('kc_house_data.csv')

# Google Colab
df = pd.read_csv('/content/drive/MyDrive/kc_house_data.csv')
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Key Takeaways

- Outlier removal consistently improved MAE and RMSE in this experiment
- Stricter filtering yielded better results, but comes with a trade-off: reduced dataset size means less training data
- R² is not always a reliable metric in the presence of outliers — MAE and RMSE provide a more honest picture of real-world prediction error
- Data cleaning decisions should be guided by domain knowledge, not just statistical thresholds

---

## Tech Stack

| | |
|---|---|
| Language | Python |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Modeling | scikit-learn (LinearRegression) |
| Environment | Google Colab / Jupyter Notebook |

---

## License

MIT
