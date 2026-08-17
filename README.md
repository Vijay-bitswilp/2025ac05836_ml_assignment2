# Credit Card Default Prediction using Machine Learning

## 1. Problem Statement

Credit card default prediction is an important classification problem in the financial domain. Financial institutions need to identify customers who are likely to default on their credit card payments so that appropriate risk-control measures can be taken.

The objective of this project is to build and compare multiple machine learning classification models for predicting whether a credit card customer will default on payment in the following month.

Six classification algorithms are implemented and evaluated using the same train-test split. Their performance is compared using multiple evaluation metrics rather than accuracy alone.

The trained models are also integrated into an interactive Streamlit application where a user can:

- Upload a test dataset in CSV format
- Select a trained classification model
- Generate predictions
- View model evaluation metrics
- View the confusion matrix
- View the classification report

---

## 2. Dataset Description

### Dataset Name

**Default of Credit Card Clients**

### Source

UCI Machine Learning Repository

Dataset ID: **350**

### Dataset Size

- Total instances: **30,000**
- Input features: **23**
- Target variable: **1**
- Total columns after combining features and target: **24**
- Classification type: **Binary Classification**

### Target Variable

The target variable was renamed as:

`DEFAULT`

The classes are:

- `0` - Customer did not default
- `1` - Customer defaulted

### Target Class Distribution

The dataset is moderately imbalanced.

| Class | Meaning | Number of Records | Percentage |
|---|---|---:|---:|
| 0 | No Default | 23,364 | 77.88% |
| 1 | Default | 6,636 | 22.12% |

Because of this class imbalance, model performance was evaluated using Accuracy as well as AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

---

## 3. Features Used

The dataset contains demographic information, credit limits, repayment history, bill statement amounts and previous payment amounts.

### Demographic / Categorical Features

- SEX
- EDUCATION
- MARRIAGE

### Repayment Status / Ordinal Features

- PAY_0
- PAY_2
- PAY_3
- PAY_4
- PAY_5
- PAY_6

### Continuous / Numerical Features

- LIMIT_BAL
- AGE
- BILL_AMT1
- BILL_AMT2
- BILL_AMT3
- BILL_AMT4
- BILL_AMT5
- BILL_AMT6
- PAY_AMT1
- PAY_AMT2
- PAY_AMT3
- PAY_AMT4
- PAY_AMT5
- PAY_AMT6

---

## 4. Data Preprocessing

The dataset was inspected before model training.

### Missing Values

No missing values were present in the dataset.

### Duplicate Records

A total of **35 exact duplicate rows** were observed.

Since the dataset does not contain a unique customer identifier among the selected predictors and identical predictor values may correspond to different customers, these records were retained instead of removing them arbitrarily.

### Cleaning of Categorical Values

Some uncommon categorical codes were grouped into appropriate existing categories.

For `EDUCATION`:

- 0, 5 and 6 were mapped to category 4

For `MARRIAGE`:

- 0 was mapped to category 3

### Train-Test Split

The dataset was divided into:

- Training data: **80%**
- Testing data: **20%**

A stratified train-test split was used so that the proportion of default and non-default customers remained approximately the same in both the training and testing datasets.

Training instances: **24,000**

Testing instances: **6,000**

### Feature Transformation

For models sensitive to feature scale, preprocessing was performed using a Scikit-learn pipeline.

Continuous numerical variables were standardized using:

`StandardScaler`

Categorical variables were encoded using:

`OneHotEncoder`

Repayment-status ordinal variables were retained in their original ordered numerical form.

Tree-based models such as Decision Tree and Random Forest were trained using the cleaned raw feature values because feature scaling is not required for these algorithms.

---

## 5. Machine Learning Models Implemented

The following six classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine
6. Random Forest Classifier

All models were evaluated on the same test dataset so that their performances could be compared fairly.

---

## 6. Evaluation Metrics

The following evaluation metrics were calculated for each model:

### Accuracy

Measures the proportion of total predictions that were classified correctly.

### AUC

Area Under the ROC Curve measures how effectively the model separates the two classes across different classification thresholds.

### Precision

Measures the proportion of predicted default customers who actually defaulted.

### Recall

Measures the proportion of actual default customers correctly identified by the model.

### F1 Score

Represents the harmonic mean of Precision and Recall and is useful when classes are imbalanced.

### Matthews Correlation Coefficient (MCC)

MCC considers all four values of the confusion matrix and provides a balanced measure of classification performance, particularly when the target classes are imbalanced.

---

## 7. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8088 | 0.7100 | **0.6915** | 0.2449 | 0.3617 | 0.3304 |
| Decision Tree | 0.7195 | 0.6099 | 0.3766 | 0.4092 | 0.3922 | 0.2106 |
| K-Nearest Neighbors | 0.7950 | 0.7045 | 0.5566 | 0.3595 | 0.4368 | 0.3299 |
| Gaussian Naive Bayes | 0.6607 | 0.7248 | 0.3590 | **0.6805** | **0.4701** | 0.2822 |
| Support Vector Machine | **0.8153** | 0.7199 | 0.6603 | 0.3399 | 0.4488 | 0.3792 |
| Random Forest | 0.8142 | **0.7572** | 0.6429 | 0.3595 | 0.4611 | **0.3817** |

---

## 8. Model-wise Observations

### 8.1 Logistic Regression

Logistic Regression achieved an accuracy of **0.8088** and produced the highest Precision among the six models at **0.6915**.

However, its Recall was only **0.2449**, which indicates that the model failed to identify a large proportion of customers who actually defaulted.

Its confusion matrix was:

- True Negatives: 4528
- False Positives: 145
- False Negatives: 1002
- True Positives: 325

Therefore, Logistic Regression performs well when predicting a customer as a defaulter with relatively high precision, but it misses many actual default cases.

---

### 8.2 Decision Tree

Decision Tree achieved an accuracy of **0.7195**, which was lower than most of the other models.

Its Recall of **0.4092** was better than Logistic Regression, but its Precision was only **0.3766**.

The model produced:

- True Negatives: 3774
- False Positives: 899
- False Negatives: 784
- True Positives: 543

The Decision Tree generated considerably more false-positive predictions and obtained the lowest AUC and MCC among the evaluated models.

This suggests that a single Decision Tree was not able to generalize as effectively as the other classifiers.

---

### 8.3 K-Nearest Neighbors

K-Nearest Neighbors achieved:

- Accuracy: **0.7950**
- AUC: **0.7045**
- Precision: **0.5566**
- Recall: **0.3595**
- F1 Score: **0.4368**
- MCC: **0.3299**

Its confusion matrix was:

- True Negatives: 4293
- False Positives: 380
- False Negatives: 850
- True Positives: 477

KNN provided a better balance between Precision and Recall than Logistic Regression.

However, its overall Accuracy and AUC remained below those of the strongest-performing models such as SVM and Random Forest.

---

### 8.4 Gaussian Naive Bayes

Gaussian Naive Bayes produced the highest Recall and F1 Score among all evaluated models.

Its results were:

- Accuracy: **0.6607**
- AUC: **0.7248**
- Precision: **0.3590**
- Recall: **0.6805**
- F1 Score: **0.4701**
- MCC: **0.2822**

Its confusion matrix was:

- True Negatives: 3061
- False Positives: 1612
- False Negatives: 424
- True Positives: 903

The high Recall indicates that Naive Bayes was able to detect a large proportion of actual default customers.

However, this was accompanied by a large number of false-positive predictions, resulting in relatively low Precision and Accuracy.

Therefore, Naive Bayes may be useful when detecting as many potential defaulters as possible is more important than minimizing false alarms.

---

### 8.5 Support Vector Machine

Support Vector Machine achieved the highest overall Accuracy among the six models.

Its results were:

- Accuracy: **0.8153**
- AUC: **0.7199**
- Precision: **0.6603**
- Recall: **0.3399**
- F1 Score: **0.4488**
- MCC: **0.3792**

Its confusion matrix was:

- True Negatives: 4441
- False Positives: 232
- False Negatives: 876
- True Positives: 451

SVM demonstrated strong overall classification performance and high Precision.

However, its Recall remained relatively low, indicating that several actual default cases were still missed.

---

### 8.6 Random Forest

Random Forest achieved one of the highest overall Accuracy scores while also producing the best AUC and MCC.

Its results were:

- Accuracy: **0.8142**
- AUC: **0.7572**
- Precision: **0.6429**
- Recall: **0.3595**
- F1 Score: **0.4611**
- MCC: **0.3817**

Its confusion matrix was:

- True Negatives: 4408
- False Positives: 265
- False Negatives: 850
- True Positives: 477

Random Forest produced the highest **AUC (0.7572)** and highest **MCC (0.3817)** while maintaining an Accuracy of **0.8142**.

This indicates that Random Forest provided one of the strongest overall balances across the different evaluation criteria.

---

## 9. Comparison of Best Metric Values

The highest value obtained for each metric was:

| Evaluation Metric | Best Model | Score |
|---|---|---:|
| Accuracy | Support Vector Machine | 0.8153 |
| AUC | Random Forest | 0.7572 |
| Precision | Logistic Regression | 0.6915 |
| Recall | Gaussian Naive Bayes | 0.6805 |
| F1 Score | Gaussian Naive Bayes | 0.4701 |
| MCC | Random Forest | 0.3817 |

The results show that there is no single model that dominates every individual evaluation metric.

Different models provide different trade-offs between detecting default customers and minimizing incorrect default predictions.

---

## 10. Overall Best Model

Based on the overall comparison, **Random Forest was selected as the best overall model** for this experiment.

Although Support Vector Machine achieved a slightly higher Accuracy of **0.8153** compared with Random Forest's **0.8142**, Random Forest achieved:

- Highest AUC: **0.7572**
- Highest MCC: **0.3817**
- Competitive Accuracy: **0.8142**
- F1 Score: **0.4611**

Since the dataset contains an imbalanced target distribution, selecting a model only on the basis of Accuracy may not provide a complete representation of performance.

AUC and MCC provide additional information about discrimination ability and balanced classification performance.

For this reason, Random Forest was considered the strongest overall baseline model among the six classifiers evaluated.

However, the preferred model can depend on the business objective.

For example:

- If detecting as many actual defaulters as possible is the priority, Gaussian Naive Bayes provides the highest Recall.
- If minimizing false default predictions is important, Logistic Regression provides the highest Precision.
- If overall Accuracy is the main criterion, Support Vector Machine provides the highest Accuracy.
- For balanced overall discrimination across multiple metrics, Random Forest provides the strongest performance.

---

## 11. ROC Curve Analysis

ROC curves were generated for all six classification models.

The corresponding AUC scores were:

| Model | AUC |
|---|---:|
| Random Forest | 0.7572 |
| Gaussian Naive Bayes | 0.7248 |
| Support Vector Machine | 0.7199 |
| Logistic Regression | 0.7100 |
| K-Nearest Neighbors | 0.7045 |
| Decision Tree | 0.6099 |

Random Forest obtained the highest AUC, indicating the strongest overall ability among the evaluated models to distinguish default customers from non-default customers across different decision thresholds.

---

## 12. Streamlit Application

An interactive Streamlit application was developed for evaluating the trained machine learning models.

### Application Features

The application allows the user to:

1. Select one of the six trained classification models from a dropdown menu.
2. Upload the supplied test dataset in CSV format.
3. View a preview of the uploaded test data.
4. View the size of the uploaded dataset.
5. Generate predictions using the selected model.
6. Display the following evaluation metrics:
   - Accuracy
   - AUC
   - Precision
   - Recall
   - F1 Score
   - MCC
7. View the confusion matrix.
8. View the complete classification report.

The same test dataset can therefore be used to compare the results generated by different trained models interactively.

---

## 13. Streamlit Application Link

Live Streamlit Application:

https://2025ac05836mlassignment2-dl8ptmfy7xkp2r5fyoftwt.streamlit.app/

## 14. GitHub Repository

The complete project source code, trained model files, notebook, test dataset and deployment files are available in this repository.

**GitHub Repository Link:**

https://github.com/Vijay-bitswilp/2025ac05836_ml_assignment2

---

## 15. Repository Structure

```text
ML_Assignment2/
│
├── app.py
├── ML_assignment2.ipynb
├── test_data.csv
├── requirements.txt
├── README.md
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── svm.pkl
    └── random_forest.pkl
