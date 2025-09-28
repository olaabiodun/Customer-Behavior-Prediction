# Predicting Customer Attrition: Data Exploration and Machine Learning Models

## Project Summary

This analysis examines customer behavior patterns to develop predictive systems that identify clients likely to discontinue their services. Understanding why customers leave is crucial for maintaining business sustainability and requires sophisticated analytical approaches.

When customers end their relationship with a service provider, it creates significant financial challenges. Early detection of at-risk customers enables proactive retention strategies and helps preserve revenue streams. Poor service quality or negative experiences often drive this decision-making process.

## Data Source and Structure

The foundation of this study comes from a comprehensive dataset available at [Kaggle's Credit Card Customer Data Repository](https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction/data).

**Dataset Composition:**
- **Identification Fields:** Customer identifiers, sequence numbers, and demographic information
- **Financial Metrics:** Account balances, transaction history, and credit scores
- **Behavioral Indicators:** Account activity status, product ownership, and service usage patterns
- **Geographic Data:** Customer location information
- **Target Variable:** Service continuation status (primary focus: "Exited" column)

## Technical Prerequisites

To reproduce this analysis, ensure the following Python packages are installed:

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical graphics
- **scikit-learn** - Machine learning algorithms

## Analytical Framework

### Data Imbalance Management
The project addresses the challenge of uneven class distribution using synthetic sample generation techniques, specifically Synthetic Minority Oversampling Technique (SMOTE), which creates artificial examples of the minority class to improve model training effectiveness.

### Pattern Discovery
Through systematic data examination, we uncover underlying trends and behavioral patterns that influence customer decisions. This involves statistical analysis, correlation studies, and visual representation of key relationships that drive service abandonment.

### Predictive Modeling
Multiple algorithmic approaches are evaluated to forecast customer departure likelihood:

**Algorithm Portfolio:**
- Traditional statistical methods (logistic regression)
- Tree-based ensemble techniques (random forest)
- Instance-based learning (K-nearest neighbors)
- Maximum margin classifiers (support vector machines)
- Advanced gradient boosting implementations (XGBoost, Gradient Boosting)

Each model incorporates class rebalancing strategies to ensure fair representation of both customer segments.

## Performance Evaluation

| Algorithm | Overall Correctness | Sensitivity | Harmonic Mean | Area Under Curve |
|-----------|-------------------|-------------|---------------|------------------|
| Logistic Regression | 70.37% | 68.32% | 47.30% | 76.41% |
| Random Forest | 86.20% | 41.44% | 53.90% | 85.24% |
| K-Nearest Neighbors | 75.23% | 66.78% | 51.21% | 77.66% |
| Support Vector Machine | 78.57% | 66.27% | 54.62% | 82.25% |
| XGBoost | 83.30% | 60.96% | 58.70% | 84.18% |
| Gradient Boosting | 81.70% | 70.03% | 59.84% | 85.98% |

## Model Assessment

The ensemble learning approaches demonstrate superior capability in handling complex customer behavior patterns. Decision tree ensembles, particularly gradient boosting variants, show exceptional skill in distinguishing between customers who remain loyal and those likely to depart.

**Key Observations:**

1. **Gradient Boosting** achieves the most favorable trade-off between identifying at-risk customers and maintaining overall accuracy, with the strongest performance metrics across all evaluation criteria.

2. **XGBoost** provides competitive results with robust predictive capabilities, making it a reliable secondary option for customer retention modeling.

3. **Random Forest** excels at correctly classifying the majority customer segment but struggles with minority class detection, indicating limitations in handling imbalanced datasets.

4. **Support Vector Machines** and **K-Nearest Neighbors** offer moderate predictive performance, outperforming basic statistical methods but falling short of advanced ensemble techniques.

5. **Logistic Regression** shows the weakest predictive power among all tested algorithms, suggesting it may be inadequate for capturing the complex relationships in customer behavior data.

## Strategic Recommendations

The analysis reveals that advanced ensemble methods, particularly gradient boosting algorithms, provide the most effective approach for customer churn prediction. These models demonstrate superior ability to manage class imbalance while maintaining high predictive accuracy. Implementation of XGBoost or Gradient Boosting classifiers would likely yield the best results for real-world customer retention initiatives.



