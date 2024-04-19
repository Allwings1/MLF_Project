# **Usage of transaction data to predict credit event**

## Team Member

Student Name | Student ID
:---------  | ---------
Zhang Yuyang| 2301212424
Ge Ruiyang| 2301212326


## Project Introduction

This is a project about using transaction data to predict defaults. In this project, we use several machine learning methods to predict default events based on the transaction data of banks(?????????).


It should be noticed that the dataset is imbalanced (e.g., approximately 99% of the data is non-default data and only 1% of the data is default data), so our priority is to identify the majority of companies that are likely to default to recommend an additional screening, so recall should be our metric of choice rather than accuracy.

What's more, the count of some import and export transaction data is missing where there is transaction amount in the corresponding position.（缺失值、图标支撑、能构造的因子少比较单一、）There is not much valid information in this dataset, and there are many missing and error values, so caution is required when constructing features.


## Data Cleaning and Processing

First, let's focus on the data. In the dataset, we are able to obtain the amount and count of the import and export transactions. It is worth noting that the transaction data has two time indexes "COHORT_MONTH" and "IMAGE_DT", where each "COHORT_MONTH" is June 30 of a certain year which further contains the transaction information of the company in the past 12 months, and under each "COHORT_MONTH" records whether the company defaults or not (e.g. 0 or 1). On June 30th of every month, we need to predict whether the company will default or not based on data from the previous 12 months.

图片1

### 1. Handling of outliers and missing values
* The values in the count column of the transaction data will appear negative, and after verification, these negative values appear on the date that no transaction occurred, so we change them to 0. 
* Another problem of the dataset is that, the count of some import and export transaction data is missing where there is transaction amount in the corresponding position, we change the corresponding count to 1.

### 2. Time weighting: ewm

$\hat{\mu}_t = \sum_{\tau=0}^{k} (1 - \alpha)^\tau Y_{t - \tau}$ 

The transaction data has two time indexes "COHORT_MONTH" and "IMAGE_DT", where each "COHORT_MONTH" is June 30 of a certain year which further contains the transaction information of the company in the past 12 months. Under each "COHORT_MONTH" we record whether the company defaults or not (e.g. 0 or 1). Thus based on the different IMAGE_DT dates of the same COHORT_MONTH, we took into account how long they were from COHORT_MONTH and tried to construct meaningful factors. Specifically, we assign weights according to the length of time IMAGE_DT is from COHORT_MONTH, and the closer the date is to COHORT_MONTH, the greater the weight.

### 3. Aggregate data by category

Considering the limitations of the features of the original data,  for all IMAGE_DT data of each company and each COHORT_MONTH, we sum import_Product_Trans_amt_t, export_Product_Trans_amt_t, import_Product_Trans_cnt_t, export_Product_Trans_cnt_t respectively and take them into account as new factors.

## Descriptive statistics and correlation analysis

* After data processing, we once again emphasize that this data is seriously imbalance. As we can see from the graph of the kernel function, basically all the data is piled up at 0.

* In addition, we have done correlation analysis and made heat maps.

## Division of training & test set and feature selection

* In this section, we split the data set into training samples (70%) and test samples (30%). 
* Then we use random forest method for feature selection.



## Construction and training of different classification models

In this part, we use logistic regression, SVM, decision tree, random forest and LSTM methods to build models and train.


Limitation of our model.
