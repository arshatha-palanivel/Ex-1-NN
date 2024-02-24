<H3>ENTER YOUR NAME : ARSHATHA P</H3>
<H3>ENTER YOUR REGISTER NO: 212222230012</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 24/2/24</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle.

## EQUIPMENTS REQUIRED:
Hardware – PCs.
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook.

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
IMPORT LIBRARIES :
```py
#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```
READ THE DATA:
```py
df = pd.read_csv('Churn_Modelling.csv')
print(df)
```
CHECK DATA:
```py
df.head()
df.tail()
df.columns
```
CHECK THE MISSING DATA:
```py
print(df.isnull().sum())
```
ASSIGNING X:
```py
X = df.iloc[:, :-1].values
print(X)
```
ASSIGNING Y:
```py
y = df.iloc[:, :-1].values
print(y)
```
HANDLING MISSING VALUES:
```py
df.fillna(df.mean().round(1), inplace=True)
print(df.isnull().sum())
```
CHECK FOR OUTLIERS:
```py
df.describe()
```
DROPPING STRING VALUES DATA FROM DATASET: & CHECKING DATASETS AFTER DROPPING STRING VALUES DATA FROM DATASET:
```py
df1 = df.drop(['Surname','Geography','Gender'],axis=1)
df1.head()
```
NORMALIE THE DATASET USING (MinMax Scaler):
```py
scaler = MinMaxScaler()
df2 = pd.DataFrame(scaler.fit_transform(df1))
print(df2)
```
SPLIT THE DATASET:
```py
X = df.iloc[:, :-1].values
print(X)
y = df.iloc[:, :-1].values
print(y)
```
TRAINING AND TESTING MODEL:
```py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(X_train)
print("Length of X_train:",len(X_train))
print(X_test)
print("Length of X_test:",len(X_test))

## OUTPUT:
### DATA CHECKING:
![01](https://github.com/arshatha-palanivel/Ex-1-NN/assets/118682484/56925ffd-7b99-4809-9236-caffca8c39eb)

### MISSING DATA:
![02](https://github.com/arshatha-palanivel/Ex-1-NN/assets/118682484/e3362ea8-bc16-4f3a-b8f8-6b8a232dfddd)

### DUPLICATES IDENTIFICATION:
![03](https://github.com/arshatha-palanivel/Ex-1-NN/assets/118682484/e0c737dd-e87b-48ed-a33c-0cec8862443f)

### VALUE OF Y:
![04](https://github.com/arshatha-palanivel/Ex-1-NN/assets/118682484/51d5f62c-d427-49bd-bd95-d8ff7720bab4)

### OUTLIERS:
![05](https://github.com/arshatha-palanivel/Ex-1-NN/assets/118682484/7c97874e-5c9c-4e59-9e24-086c0e2b152c)

### CHECKING DATASET AFTER DROPPING STRING VALUES DATA FROM DATASET:
![06](https://github.com/arshatha-palanivel/Ex-1-NN/assets/118682484/1e7ed9f5-9bae-4e7b-8a7b-e496c89b7209)

### NORMALIZE THE DATASET:
![07](https://github.com/arshatha-palanivel/Ex-1-NN/assets/118682484/26cd8369-d209-4ceb-bd4c-ee169fda165b)

### SPLIT THE DATASET:
![08](https://github.com/arshatha-palanivel/Ex-1-NN/assets/118682484/13b4b35d-6267-4833-bef4-ccea5150d18a)

### TRAINING AND TESTING MODEL:
![0901](https://github.com/arshatha-palanivel/Ex-1-NN/assets/118682484/aa0e8539-3509-4a85-96c3-d209b2da9027)
![0902](https://github.com/arshatha-palanivel/Ex-1-NN/assets/118682484/925b72d8-a880-4ae3-a5a6-fc919b304c5a)
![0903](https://github.com/arshatha-palanivel/Ex-1-NN/assets/118682484/f215f7ae-4530-4d32-bc9c-d543549bbf37)
![0904](https://github.com/arshatha-palanivel/Ex-1-NN/assets/118682484/be128f2d-fb10-4060-bf09-403763ff8015)







## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


