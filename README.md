# Patient Status Monitoring System
This project analyzes human vital signs from two distinct datasets to categorize patient health status using data preprocessing, exploratory data analysis, and K-Means clustering. The goal is to segment patients into meaningful groups, such as **_'Normal Vitals'_** or **_'Frequent Monitoring'_**, to aid in healthcare monitoring.
### Table of Contents
* [Project Overview](#project-overview)
* [Key Features](#key-features)
* [Datasets Used](#datasets-used)
* [Methodology](#methodology)
* [Results](#results)
* [How to Use](#how-to-use)
* [Libraries Used](#libraries-used)
### Project Overview
The primary objective of this data analysis project is to integrate and analyze human vital signs data to identify distinct patient health profiles. By applying K-Means clustering on key vital sign categories, the system groups patients into clusters representing different levels of health risk. This allows for a quick and automated assessment of a patient's condition, flagging those who may require immediate attention or more frequent monitoring.
### Key Features
* **Data Integration**: Combines two separate datasets on human vital signs into a single, cohesive dataset for analysis.
  
* **Data Preprocessing**: Includes cleaning the data, renaming columns for consistency, and handling outliers using the Interquartile Range (IQR) method.
  
* **Feature Engineering**:
  - Calculates _Mean Arterial Pressure (MAP)_ from systolic and diastolic blood pressure.
  - Creates categorical features for _Blood Pressure (BPCategory), Heart Rate (HRCategory), and Oxygen Saturation (OxyCategory)_ based on established medical ranges.

* **Exploratory Data Analysis (EDA)**: Visualizes data distributions with histograms and identifies correlations between vital signs using a heatmap.
  
* **K-Means Clustering**: Applies the K-Means algorithm to segment patients based on their vital sign categories.
  
* **Model Evaluation**:  
  * **Elbow Method**: Used to determine the optimal number of clusters for the K-Means algorithm.
  * **Silhouette Score**: Calculated to evaluate the quality and separation of the resulting clusters.

* **Prediction Function**: A simple model is defined to predict the health status of a new patient based on their vital sign inputs.
### Datasets Used  
The analysis is performed on two datasets:
1. **Human Vital Signs Dataset 2024**: Contains vital signs such as Heart Rate, Respiratory Rate, Body Temperature, Oxygen Saturation, and Blood Pressure, along with demographic data like age, gender, weight, and height.
  
2. **Health Monitor Dataset**: Provides additional health metrics and symptoms like Dehydration, Medicine Overdose, and Cough, alongside core vital signs.

The common vital sign columns from both datasets were integrated to create a comprehensive dataset for clustering.  
### Methodology
1. **Data Loading and Cleaning**: The two datasets are loaded into pandas DataFrames. Column names are standardized for easier merging (e.g., 'Heart Rate' becomes 'HeartRate').

2. **Feature Engineering & Integration**: Mean Arterial Pressure (MAP) is calculated. The common columns from both datasets are identified and concatenated into a single DataFrame. Outliers in the numerical columns are removed using the IQR method.

3. **Categorization**: Numerical vital signs (Heart Rate, Blood Pressure, Oxygen Saturation) are converted into categories ('Normal', 'High', 'Low', 'Elevated', etc.) using predefined functions based on medical guidelines.

4. **Clustering**:
   - The categorical features are encoded into numerical labels.
   - The Elbow Method is applied to the encoded categories, suggesting an optimal number of 3 clusters.
   - K-Means clustering is performed with n_clusters=3.

5. **Analysis and Interpretation**: The resulting clusters are analyzed and mapped to meaningful health statuses:
   - **_Cluster 0: Frequent Monitoring_**
   - **_Cluster 1: Normal Vitals_**
   - **_Cluster 2: Immediate Attention_**

6. **Evaluation**: A Silhouette Score is calculated on a sample of the data to assess the model's performance. The resulting score of 1.0 indicates that the clusters are perfectly distinct based on the input features.

7. **Prediction**: A function is created to take new patient vitals, categorize them, and predict their cluster and corresponding health status.

### Results
The K-Means clustering algorithm successfully partitioned the patient data into three distinct and meaningful clusters.
* **Cluster Distributions**: The distribution of patients across the generated categories for Heart Rate, Blood Pressure, and Oxygen Saturation was visualized to understand the composition of the dataset.

<img width="1502" height="2398" alt="image" src="https://github.com/user-attachments/assets/0f903026-89e6-4f1c-a49a-07ba3cbb75c7" />


* **Cluster Visualization**: A 3D scatter plot visualizes the three distinct clusters based on the encoded categories, showing clear separation between the groups.

<img width="934" height="983" alt="image" src="https://github.com/user-attachments/assets/0895f6da-3834-42a8-abbb-b892a99cdd97" />


* **Evaluation Score**: The Silhouette Score of 1.0 indicates excellent cluster separation, suggesting that the categorical features derived from the vital signs are strong differentiators of patient status.

### How to Use
1. **Prerequisites**: Ensure you have Python installed with the necessary libraries.
2. **Clone the Repository**:
```
git clone https://github.com/aldona-rose/patient-status-monitoring.git
cd patient-status-monitoring
```
3. **Install Libraries**:

`pip install pandas scikit-learn seaborn matplotlib`

4. **Run the Notebook**: Open and run the PatientStatusMonitoring.ipynb notebook in a Jupyter environment. Make sure the dataset files (human_vital_signs_dataset_2024.csv and Data-Table 1.csv) are in the correct paths as specified in the notebook.
### Libraries Used
* Pandas
* Scikit-learn
* Seaborn
* Matplotlib
