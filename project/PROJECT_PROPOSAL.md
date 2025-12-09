# AI 221 Final Project Proposal
## Philippine Family Income and Expenditure Analysis

**Student Name:** [Your Name]  
**Course:** AI 221 - Classical Machine Learning  
**Date Submitted:** December 9, 2025  
**Dataset Source:** [Family Income and Expenditure Survey - Kaggle](https://www.kaggle.com/datasets/grosvenpaul/family-income-and-expenditure)

---

## 1. Introduction

I apologize for my delayed response. Due to transitioning to a new laptop and missing some online class announcements, I was unable to receive timely updates. However, I have been working steadily on the Final Project and would like to submit my proposal for your approval.

---

## 2. Project Title

**"Socio-Economic Classification and Household Income Prediction in the Philippines Using Machine Learning"**

---

## 3. Problem Statement

The Philippine Statistics Authority conducts the Family Income and Expenditure Survey (FIES) to understand household income patterns and expenditure behaviors. This project aims to:

1. **Predict household income levels** based on expenditure patterns using multiple regression and classification methods
2. **Classify households into income brackets** based on their spending behavior
3. **Discover natural household segments** and unusual spending patterns through unsupervised learning
4. **Visualize the household landscape** in reduced dimensionality space

---

## 4. Data Source

**Dataset:** Family Income and Expenditure Survey (Philippine Statistics Authority)  
**Platform:** Kaggle - [Dataset Link](https://www.kaggle.com/datasets/grosvenpaul/family-income-and-expenditure)  
**Size:** 40,000+ households × 60 variables  
**Content:** Household income sources, expenditure categories, demographics, and socio-economic indicators

---

## 5. Proposed Methodology

### 5.1 Supervised Learning Tasks

#### Task 1: Income Regression
- **Objective:** Predict household total income from expenditure patterns
- **Methods to be used:**
  - Linear Regression (baseline)
  - Support Vector Regression (SVR)
  - Multi-Layer Perceptron (MLP) Neural Networks
  - Random Forest Regressor
  - Gradient Boosting Regressor

#### Task 2: Income Classification
- **Objective:** Classify households into income brackets (Low, Middle, High) or quintiles
- **Methods to be used:**
  - Logistic Regression (baseline)
  - Support Vector Machine (SVM) Classifier
  - MLP Neural Networks Classifier
  - Random Forest Classifier
  - Gradient Boosting Classifier

### 5.2 Unsupervised Learning Tasks

#### Task 1: Household Clustering and Segmentation
- **Objective:** Discover natural groups of households with similar expenditure patterns
- **Methods to be used:**
  - K-Means Clustering
  - Hierarchical Clustering (Agglomerative)
  - DBSCAN (Density-based)

#### Task 2: Dimensionality Reduction and Visualization
- **Objective:** Reduce 60 dimensions to 2D/3D space for visualization and analysis
- **Methods to be used:**
  - Principal Component Analysis (PCA)
  - t-SNE (t-Distributed Stochastic Neighbor Embedding)
  - Optional: UMAP

#### Task 3: Anomaly Detection
- **Objective:** Identify unusual household spending patterns
- **Methods to be used:**
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - Mahalanobis Distance
  - Z-Score based detection

---

## 6. Expected Outcomes

### Supervised Learning Results
- Regression models with R² scores and RMSE metrics
- Classification models with accuracy, precision, recall, and F1-scores
- Feature importance rankings showing key income drivers
- Model comparison and selection of best performer

### Unsupervised Learning Results
- Optimal number of household clusters (3-5 groups)
- Cluster profiles and interpretation (household types)
- Anomalies detected and analyzed
- Clear visualization in PCA and t-SNE spaces showing cluster separation
- Silhouette, Davies-Bouldin, and Calinski-Harabasz scores

### Key Insights
- Which expenditure categories are strongest predictors of income
- Natural household segments for targeted policy interventions
- Data quality issues identified through anomaly detection
- Dimensionality reduction showing main variance drivers

---

## 7. Alignment with AI 221 Curriculum

| Week | Topic | Methods to Apply |
|------|-------|-----------------|
| 2 | EDA | Data exploration and visualization |
| 3 | Linear Regression | Linear/Logistic Regression baseline models |
| 4 | Kernel Methods | Support Vector Regression/Machine |
| 5 | Cross-validation & Hyperparameter Tuning | Model optimization |
| 7 | Neural Networks | MLP for regression and classification |
| 8 | Ensemble Learning | Random Forest, Gradient Boosting |
| 9 | Linear Dimensionality Reduction | PCA for visualization |
| 10 | Nonlinear Dimensionality Reduction | t-SNE for visualization |
| 11 | Clustering & Anomaly Detection | K-Means, Hierarchical, DBSCAN, Isolation Forest, LOF |

---

## 8. Project Deliverables

1. **Jupyter Notebook** containing:
   - Complete exploratory data analysis
   - Data preprocessing and feature engineering
   - All supervised learning models with evaluations
   - All unsupervised learning analyses
   - Comprehensive visualizations
   - Summary dashboard and conclusions

2. **Written Report** including:
   - Problem formulation
   - Methodology explanation
   - Results and findings
   - Recommendations and insights

3. **Class Presentation** covering:
   - Problem statement and motivation
   - Methodology overview
   - Key findings and comparisons
   - Business implications for Philippines policy

---

## 9. Implementation Timeline

- **Week 1:** Data loading, EDA, and preprocessing
- **Week 2:** Supervised learning models (regression and classification)
- **Week 3:** Unsupervised learning (clustering and anomaly detection)
- **Week 4:** Dimensionality reduction and visualization
- **Week 5:** Results compilation, writing, and presentation preparation

---

## 10. Conclusion

This project comprehensively addresses both supervised and unsupervised learning requirements while providing practical insights into Philippine household economics. The use of multiple methods from the AI 221 curriculum allows for robust comparison and selection of optimal approaches for socio-economic classification.

I am ready to proceed with implementation and welcome your feedback on this proposal.

---

**Respectfully submitted,**

[Your Name]  
AI 221 - Classical Machine Learning  
December 9, 2025
