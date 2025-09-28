#!/usr/bin/env python
# coding: utf-8

"""
Customer Behavior Analysis: Predictive Modeling for Service Discontinuation

This comprehensive study examines customer behavior patterns to develop sophisticated
predictive systems for identifying clients at risk of terminating their service relationships.
"""

# =============================================================================
# INITIALIZATION AND DATA ACQUISITION
# =============================================================================

# Core data processing and scientific computing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning and statistical modeling frameworks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, roc_auc_score

# Predictive algorithms collection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Specialized tools for imbalanced datasets
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Configuration for warning management
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA ACQUISITION AND INITIAL ASSESSMENT
# =============================================================================

def load_and_examine_data():
    """Load customer dataset and perform initial quality assessment."""

    # Load the customer dataset for comprehensive analysis
    customer_data = pd.read_csv('Churn_Modelling.csv')

    print("INITIAL DATA OVERVIEW:")
    print("=" * 50)
    print(customer_data.head())

    print("\nDATASET STRUCTURAL INFORMATION:")
    print("=" * 50)
    print(customer_data.info())

    print("\nNUMERICAL FEATURES STATISTICAL SUMMARY:")
    print("=" * 50)
    print(customer_data.describe())

    print("\nDATA QUALITY ASSESSMENT - MISSING VALUES:")
    print("=" * 50)
    print(customer_data.isnull().sum())

    return customer_data

# =============================================================================
# EXPLORATORY BEHAVIORAL PATTERN ANALYSIS
# =============================================================================

def analyze_service_distribution(customer_data):
    """Analyze the distribution of service continuation status."""

    plt.figure(figsize=(8, 8))

    # Calculate service status distribution
    service_counts = customer_data['Exited'].value_counts()

    # Create visual representation of service status proportions
    plt.pie(service_counts, labels=['Active Customers', 'Departed Customers'],
            autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'lightcoral'])
    plt.title('Service Continuation Status Distribution', fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.show()

    print("KEY OBSERVATION:")
    print("The service status shows significant imbalance, indicating that traditional")
    print("accuracy metrics may not be appropriate for model evaluation.")

def analyze_numerical_distributions(customer_data):
    """Examine distribution characteristics of key numerical attributes."""

    print("\nNUMERICAL FEATURES DISTRIBUTION ANALYSIS:")
    print("=" * 50)

    plt.figure(figsize=(16, 12))

    # Define key metrics for analysis
    key_numerical_features = ['Age', 'Tenure', 'EstimatedSalary']

    for idx, feature_name in enumerate(key_numerical_features):
        plt.subplot(2, 2, idx + 1)
        sns.boxplot(x=customer_data[feature_name], color='skyblue', width=0.6)
        plt.title(f'{feature_name} Distribution Pattern', fontsize=12)
        plt.xlabel('Value Spectrum')
        plt.ylabel('Distribution Density')

    plt.tight_layout()
    plt.show()

    print("DISTRIBUTION INSIGHTS:")
    print("- Age shows the most varied distribution pattern")
    print("- Tenure and salary demonstrate relatively normal distributions")
    print("- Limited extreme outliers observed across numerical features")

def analyze_categorical_patterns(customer_data):
    """Analyze categorical feature distributions and patterns."""

    print("\nCATEGORICAL FEATURES DISTRIBUTION ANALYSIS:")
    print("=" * 50)

    plt.figure(figsize=(16, 12))

    # Define categorical attributes for comprehensive analysis
    categorical_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

    for idx, feature_name in enumerate(categorical_features):
        plt.subplot(2, 2, idx + 1)
        sns.countplot(x=customer_data[feature_name], palette='Set2')
        plt.title(f'{feature_name} Distribution', fontsize=12)
        plt.xlabel('Categories')
        plt.ylabel('Customer Count')
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    print("CATEGORICAL INSIGHTS:")
    print("- Geographic distribution shows regional concentration patterns")
    print("- Gender representation appears well-balanced")
    print("- Credit card ownership varies significantly")
    print("- Account activity status shows different participation levels")

# =============================================================================
# COMPARATIVE BEHAVIORAL ANALYSIS
# =============================================================================

def comparative_feature_analysis(customer_data):
    """Compare feature distributions across service status groups."""

    print("\nCOMPARATIVE ANALYSIS BY SERVICE STATUS:")
    print("=" * 50)

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))

    # Financial and demographic comparisons
    sns.boxplot(data=customer_data, y="CreditScore", x="Exited", ax=axes[0, 0], palette="Set1")
    axes[0, 0].set_title("Credit Score Patterns by Service Status")

    sns.boxplot(data=customer_data, y="Age", x="Exited", ax=axes[0, 1], palette="Set1")
    axes[0, 1].set_title("Age Distribution by Service Status")

    # Relationship and account comparisons
    sns.boxplot(data=customer_data, y="Tenure", x="Exited", ax=axes[1, 0], palette="Set1")
    axes[1, 0].set_title("Tenure Analysis by Service Status")

    sns.boxplot(data=customer_data, y="Balance", x="Exited", ax=axes[1, 1], palette="Set1")
    axes[1, 1].set_title("Account Balance by Service Status")

    # Income comparison
    sns.boxplot(data=customer_data, y="EstimatedSalary", x="Exited", ax=axes[2, 0], palette="Set1")
    axes[2, 0].set_title("Income Analysis by Service Status")

    # Hide empty subplot
    axes[2, 1].axis("off")

    plt.tight_layout()
    plt.show()

    print("COMPARATIVE ANALYSIS RESULTS:")
    print("- Credit scores show similar patterns across service groups")
    print("- Age demonstrates more pronounced differences between groups")
    print("- Account tenure and balance exhibit distinct behavioral patterns")
    print("- Income levels appear relatively consistent across service status")

# =============================================================================
# FEATURE ENGINEERING AND SELECTION
# =============================================================================

def create_enhanced_features(customer_data):
    """Create additional features that may improve predictive power."""

    print("\nFEATURE ENGINEERING AND ENHANCEMENT:")
    print("=" * 50)

    # Create new calculated features for better analysis
    customer_data = customer_data.copy()

    # Financial behavior metrics
    customer_data['CreditUtilization'] = customer_data['Balance'] / customer_data['CreditScore']

    # Customer engagement scoring
    customer_data['EngagementScore'] = (customer_data['NumOfProducts'] +
                                       customer_data['HasCrCard'] +
                                       customer_data['IsActiveMember'])

    # Financial capacity indicators
    customer_data['BalanceToIncomeRatio'] = customer_data['Balance'] / customer_data['EstimatedSalary']

    # Demographic interaction features
    customer_data['CreditAgeInteraction'] = customer_data['CreditScore'] * customer_data['Age']

    print("New features created:")
    print("- CreditUtilization: Balance relative to credit score")
    print("- EngagementScore: Composite measure of customer activity")
    print("- BalanceToIncomeRatio: Financial capacity indicator")
    print("- CreditAgeInteraction: Demographic interaction feature")

    return customer_data

def analyze_feature_relationships(customer_data):
    """Analyze correlations and relationships between features."""

    print("\nFEATURE RELATIONSHIP ANALYSIS:")
    print("=" * 50)

    # Remove identifier columns for correlation analysis
    analysis_features = customer_data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Generate correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = analysis_features.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Analyze target variable correlations
    target_correlations = correlation_matrix['Exited'].sort_values(ascending=False)
    print("\nFEATURE CORRELATIONS WITH SERVICE STATUS:")
    print(target_correlations)

    print("\nKEY CORRELATION INSIGHTS:")
    print("- Age shows moderate positive correlation with service discontinuation")
    print("- Active membership demonstrates negative correlation with churn")
    print("- Credit utilization and product ownership show positive associations")

# =============================================================================
# PREDICTIVE MODELING FRAMEWORK
# =============================================================================

def prepare_modeling_data(customer_data):
    """Prepare and preprocess data for predictive modeling."""

    print("\nDATA PREPARATION FOR MODELING:")
    print("=" * 50)

    # Separate target variable and features
    target_variable = customer_data['Exited']
    feature_columns = customer_data.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Split data for training and validation
    training_features, validation_features, training_target, validation_target = train_test_split(
        feature_columns, target_variable, test_size=0.3, random_state=42, stratify=target_variable
    )

    print(f"Training set dimensions: {training_features.shape}")
    print(f"Validation set dimensions: {validation_features.shape}")

    # Identify categorical features for encoding
    categorical_features = ['Geography', 'Gender']

    # Apply label encoding to categorical features
    feature_encoder = LabelEncoder()
    for feature in categorical_features:
        training_features[feature] = feature_encoder.fit_transform(training_features[feature])
        validation_features[feature] = feature_encoder.transform(validation_features[feature])

    # Define numerical features for standardization
    numerical_features = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary',
                         'CreditUtilization', 'BalanceToIncomeRatio', 'CreditAgeInteraction']

    # Apply feature scaling
    feature_scaler = StandardScaler()
    training_features[numerical_features] = feature_scaler.fit_transform(training_features[numerical_features])
    validation_features[numerical_features] = feature_scaler.transform(validation_features[numerical_features])

    return training_features, validation_features, training_target, validation_target

def develop_predictive_models(training_features, validation_features, training_target, validation_target):
    """Develop and evaluate multiple predictive models."""

    print("\nPREDICTIVE MODEL DEVELOPMENT:")
    print("=" * 50)

    # Define model configurations with different algorithms
    predictive_models = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'K-Nearest Neighbors': Pipeline([
            ('sampling', SMOTE(random_state=42)),
            ('classification', KNeighborsClassifier())
        ]),
        'Support Vector Machine': Pipeline([
            ('sampling', SMOTE(random_state=42)),
            ('classification', SVC(probability=True, random_state=42))
        ]),
        'XGBoost': XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=(len(training_target) - sum(training_target)) / sum(training_target),
            random_state=42
        ),
        'Gradient Boosting': Pipeline([
            ('sampling', SMOTE(random_state=42)),
            ('classification', GradientBoostingClassifier(random_state=42))
        ])
    }

    # Initialize results storage
    model_performance_results = []

    # Evaluate each model
    for model_name, model_algorithm in predictive_models.items():
        print(f"\nEvaluating {model_name}:")
        print("-" * 30)

        # Train the model
        model_algorithm.fit(training_features, training_target)

        # Generate predictions
        validation_predictions = model_algorithm.predict(validation_features)

        # Calculate comprehensive performance metrics
        model_accuracy = accuracy_score(validation_target, validation_predictions)
        model_recall = recall_score(validation_target, validation_predictions)
        model_f1 = f1_score(validation_target, validation_predictions)

        # Calculate ROC AUC for probabilistic models
        if hasattr(model_algorithm, "predict_proba"):
            model_roc_auc = roc_auc_score(validation_target, model_algorithm.predict_proba(validation_features)[:, 1])
        else:
            model_roc_auc = None

        # Display detailed classification report
        print("Classification Performance:")
        print(classification_report(validation_target, validation_predictions))

        # Display confusion matrix
        print("Prediction Matrix:")
        print(confusion_matrix(validation_target, validation_predictions))

        # Store results for comparison
        model_performance_results.append({
            'Algorithm': model_name,
            'Accuracy': model_accuracy,
            'Recall': model_recall,
            'F1_Score': model_f1,
            'ROC_AUC': model_roc_auc
        })

    # Create performance comparison summary
    performance_summary = pd.DataFrame(model_performance_results)
    print("\nMODEL PERFORMANCE COMPARISON:")
    print("=" * 50)
    print(performance_summary.round(4))

    return performance_summary

# =============================================================================
# MAIN EXECUTION WORKFLOW
# =============================================================================

if __name__ == "__main__":
    print("CUSTOMER BEHAVIOR ANALYSIS: PREDICTIVE MODELING WORKFLOW")
    print("=" * 60)

    # Execute complete analysis workflow
    customer_dataset = load_and_examine_data()

    analyze_service_distribution(customer_dataset)
    analyze_numerical_distributions(customer_dataset)
    analyze_categorical_patterns(customer_dataset)
    comparative_feature_analysis(customer_dataset)

    enhanced_dataset = create_enhanced_features(customer_dataset)
    analyze_feature_relationships(enhanced_dataset)

    training_data, validation_data, training_labels, validation_labels = prepare_modeling_data(enhanced_dataset)
    final_results = develop_predictive_models(training_data, validation_data, training_labels, validation_labels)

    print("\nANALYSIS COMPLETION SUMMARY:")
    print("=" * 50)
    print("The comprehensive customer behavior analysis has been completed successfully.")
    print("All models have been trained and evaluated using the enhanced feature set.")
    print("Results indicate varying performance levels across different algorithmic approaches.")
