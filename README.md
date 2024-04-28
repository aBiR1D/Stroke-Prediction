# Stroke Prediction
![th](https://github.com/aBiR1D/Stroke-Prediction/assets/56883085/d9acac5e-57b1-4f24-9f42-4a89603d5bc6)

## CONTEXT
According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of 
total deaths.This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age,
various diseases, and smoking status. Each row in the data provides relavant information about the patient.

### Attribute Information

- 1.id: unique identifier
- 2.gender: "Male", "Female" or "Other"
- 3.age: age of the patient
- 4.hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
- 5.heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
- 6.ever_married: "No" or "Yes"
- 7.work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
- 8.Residence_type: "Rural" or "Urban"
- 9.avg_glucose_level: average glucose level in blood
- 10.bmi: body mass index
- 11.smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
- 12.stroke: 1 if the patient had a stroke or 0 if not

- *Note: "Unknown" in smoking_status means that the information is unavailable for this patient

### So, Why Catboost Or, Lightgbm ?
Boosting algorithms have become one of the most powerful algorithms for training on structural (tabular) data. The three most famous boosting algorithm
implementations that have provided various recipes for winning ML competitions are:
 - 1.XGBoost
 - 2.CatBoost
 - 3.LightGBM

We have a lot of categorical data. And Catboost And LightBGM are known to perform well in terms of tackling categorical data.CatBoost also provides 
significant performance potential as it performs remarkably well with default parameters, significantly improving performance when tuned.

Here, I'll be deploying both Of Them ! Let's See How They Perform.
