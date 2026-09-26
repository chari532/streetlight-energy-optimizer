\# 💡 Streetlight Energy Optimizer



An end-to-end Machine Learning project that predicts streetlight energy consumption and identifies periods where lighting could potentially be reduced or dimmed to improve energy efficiency.



\## 📌 Project Overview



Streetlights consume significant amounts of electricity, and their energy usage varies depending on factors such as time of day, day of the month, month, weekday/weekend patterns, and individual streetlight devices.



This project uses historical streetlight energy-meter data to:



\* Clean and preprocess energy-meter readings

\* Calculate hourly energy consumption

\* Perform feature engineering

\* Train and compare Machine Learning models

\* Predict streetlight energy consumption

\* Identify high-consumption periods

\* Generate Reduce/DIM lighting recommendations

\* Estimate potential energy savings under a defined reduction scenario

\* Deploy the Machine Learning model through a Streamlit web application



\---



\## 🎯 Project Objective



The main objective is to build a Machine Learning system that can help identify periods of higher streetlight energy consumption and provide an automated recommendation for potential lighting optimization.



\### Workflow



```text

Streetlight Energy Data

&#x20;       ↓

Data Cleaning

&#x20;       ↓

Energy Consumption Calculation

&#x20;       ↓

Feature Engineering

&#x20;       ↓

Exploratory Data Analysis

&#x20;       ↓

Machine Learning

&#x20;       ↓

Model Evaluation

&#x20;       ↓

Energy Optimization

&#x20;       ↓

Streamlit Application

```



\---



\## 📊 Dataset



The project uses the \*\*SmartLivingEPC Public Street Lighting dataset\*\* containing energy-meter measurements from multiple streetlight devices.



Dataset source:



\[Zenodo — SmartLivingEPC Public Street Lighting Dataset](https://zenodo.org/records/15781077?utm\_source=chatgpt.com)



\### Dataset characteristics



\* \*\*4 streetlight devices\*\*

\* \*\*4,958 raw records\*\*

\* Energy measurements in \*\*kWh\*\*

\* Timestamp-based measurements

\* Device-specific energy-meter readings



\### Main columns



| Column      | Description                          |

| ----------- | ------------------------------------ |

| `device\_id` | Unique streetlight device identifier |

| `timestamp` | Date and time of measurement         |

| `meas\_type` | Measurement type                     |

| `value`     | Energy-meter reading                 |

| `unit`      | Measurement unit                     |



\---



\## 🧹 Data Preprocessing



The raw energy-meter readings were processed before Machine Learning.



\### Steps performed



1\. Loaded multiple CSV files

2\. Combined the individual device datasets

3\. Converted timestamps into datetime format

4\. Sorted data by device and timestamp

5\. Calculated changes between consecutive meter readings

6\. Identified negative meter changes caused by resets/corrections

7\. Removed invalid/missing consumption values

8\. Detected abnormal energy-consumption values using the IQR method

9\. Created a cleaned dataset



After cleaning:



```text

Cleaned records: 4,814

```



\---



\## ⚙️ Feature Engineering



The following features were created:



\### Time-based features



\* `hour`

\* `day`

\* `month`

\* `day\_of\_week`

\* `is\_weekend`



\### Cyclic time features



To represent the cyclical nature of hours and months:



\* `hour\_sin`

\* `hour\_cos`

\* `month\_sin`

\* `month\_cos`



\### Target variable



```text

energy\_consumption

```



The target was calculated from the change in consecutive energy-meter readings.



\---



\## 🤖 Machine Learning Models



Three regression approaches were evaluated.



\### 1. Linear Regression



Used as a baseline model.



\### 2. Random Forest Regression



Used to capture nonlinear relationships between time/device features and energy consumption.



\### 3. Improved Random Forest Regression



The model was enhanced with cyclic time features such as `hour\_sin`, `hour\_cos`, `month\_sin`, and `month\_cos`.



\---



\## 📈 Model Results



The models were evaluated using a time-based train/test split.



| Model                  |   MAE |  RMSE |        R² |

| ---------------------- | ----: | ----: | --------: |

| Linear Regression      | 1.693 | 1.886 |     0.139 |

| Random Forest          | 0.516 | 1.186 |     0.660 |

| Improved Random Forest | 0.547 | 1.138 | \*\*0.686\*\* |



The improved Random Forest achieved an \*\*R² of 0.686\*\* on the test set.



\### Evaluation metrics



\*\*MAE — Mean Absolute Error\*\*



Measures the average absolute difference between actual and predicted energy consumption.



\*\*RMSE — Root Mean Squared Error\*\*



Penalizes larger prediction errors more strongly.



\*\*R² — R-squared\*\*



Measures how much of the variation in the target is explained by the model.



\---



\## 💡 Energy Optimization



The predicted energy consumption was used to classify test periods into:



\* \*\*Normal lighting\*\*

\* \*\*Reduce/DIM lighting\*\*



\### Test-set classification



| Recommendation      | Periods |

| ------------------- | ------: |

| Normal lighting     |     700 |

| Reduce/DIM lighting |     263 |

| Total               |     963 |



Approximately \*\*27.3% of the test periods\*\* received a Reduce/DIM recommendation.



\### Potential energy-saving scenario



A hypothetical scenario was created where Reduce/DIM periods consume \*\*20% less energy\*\*.



Under this assumption:



```text

Estimated potential saving = 264.66 kWh

```



> \*\*Note:\*\* This is a scenario-based estimate, not measured real-world savings. Actual savings would depend on the lighting-control system, dimming level, operating conditions, and hardware capabilities.



\---



\## 🖥️ Streamlit Application



The trained Random Forest model was saved and integrated into a Streamlit application.



The application allows a user to enter:



\* Hour

\* Day

\* Month

\* Day of week

\* Device ID



The application then:



```text

User Input

&#x20;   ↓

Feature Generation

&#x20;   ↓

Saved Random Forest Model

&#x20;   ↓

Energy Prediction

&#x20;   ↓

Lighting Recommendation

```



\### Example prediction



```text

Predicted Energy Consumption: 3.83 kWh



Recommendation:

High energy consumption — Consider Reduce/DIM lighting.

```



\---



\## 📸 Screenshots



\### Streamlit Application



Add your Streamlit screenshot here:



```text

screenshots/streamlit\_prediction.png

```



!\[Streamlit Prediction](screenshots/streamlit\_prediction.png)



\### Actual vs Predicted Energy



Add your model visualization here:



```text

visualizations/actual\_vs\_predicted.png

```



!\[Actual vs Predicted](visualizations/actual\_vs\_predicted.png)



> Create the `screenshots` and `visualizations` folders and place your screenshots/graphs inside them before pushing the final README update.



\---



\## 🛠️ Technologies Used



\### Programming



\* Python



\### Data Analysis



\* Pandas

\* NumPy



\### Visualization



\* Matplotlib



\### Machine Learning



\* Scikit-learn

\* Linear Regression

\* Random Forest Regression



\### Deployment



\* Streamlit

\* Joblib



\### Development Environment



\* Google Colab

\* Visual Studio Code / PowerShell

\* Git

\* GitHub



\---



\## 📁 Project Structure



```text

streetlight-energy-optimizer/

│

├── data/

│   └── Public lighting/

│       ├── ENERGYMETER.csv

│       ├── ENERGYMETER.csv

│       ├── ENERGYMETER.csv

│       └── ENERGYMETER.csv

│

├── notebooks/

│   └── streetlight\_energy\_optimizer.ipynb

│

├── screenshots/

│   └── streamlit\_prediction.png

│

├── visualizations/

│   └── actual\_vs\_predicted.png

│

├── app.py

├── model\_features.pkl

├── streetlight\_energy\_model.pkl

├── requirements.txt

├── .gitignore

└── README.md

```



> If the `.pkl` model files are excluded through `.gitignore`, they must be provided separately when deploying the application.



\---



\## 🚀 How to Run the Project



\### 1. Clone the repository



```bash

git clone https://github.com/chari532/streetlight-energy-optimizer.git

```



\### 2. Open the project



```bash

cd streetlight-energy-optimizer

```



\### 3. Install dependencies



```bash

pip install -r requirements.txt

```



\### 4. Make sure the model files are available



The Streamlit application requires:



```text

streetlight\_energy\_model.pkl

model\_features.pkl

```



Place these files in the project root directory.



\### 5. Run the Streamlit application



```bash

streamlit run app.py

```



\### 6. Open the application



Streamlit will provide a local address similar to:



```text

http://localhost:8501

```



Open it in your browser.



\---



\## 🔮 Future Improvements



Possible future improvements include:



\* Add real-time streetlight sensor data

\* Include weather information

\* Include traffic/vehicle activity

\* Include ambient light or sunlight information

\* Add automatic dimming-control integration

\* Add device-level monitoring

\* Add daily/monthly energy dashboards

\* Add electricity-cost estimation

\* Compare multiple Machine Learning algorithms

\* Deploy the application to a cloud platform

\* Add automated model retraining



\---



\## 🎓 Skills Demonstrated



This project demonstrates practical experience in:



\* Python programming

\* Data preprocessing

\* Exploratory Data Analysis

\* Time-based feature engineering

\* Feature encoding

\* Regression Machine Learning

\* Random Forest

\* Model evaluation

\* Data visualization

\* Energy optimization logic

\* Model serialization

\* Streamlit application development

\* Git and GitHub



\---



\## 👨‍💻 Author



\*\*R Siva\*\*



B.Tech — Electrical \& Electronics Engineering



Interested in:



\* Data Science

\* Machine Learning

\* Artificial Intelligence

\* Data Analytics



\---



\## ⭐ Project Highlights



```text

✔ Real-world public energy dataset

✔ 4,958 raw records

✔ 4 streetlight devices

✔ 4,814 records after cleaning

✔ Multiple ML models compared

✔ Improved Random Forest R² = 0.686

✔ 263 high-consumption test periods identified

✔ Streamlit prediction application

✔ GitHub-ready Machine Learning project

```



