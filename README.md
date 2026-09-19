# 🚘 AutoMile-AI

A machine learning project that predicts **vehicle fuel efficiency (mileage)** based on vehicle specifications and usage-related features. The project covers the complete ML workflow, from data preprocessing and exploratory analysis to model training, evaluation, and deployment using Streamlit.

## 📌 Project Overview

Vehicle mileage depends on several factors such as engine capacity, fuel type, transmission, vehicle age, and distance driven. This project uses machine learning regression techniques to estimate the expected mileage of a vehicle from these attributes.

The trained model is integrated into an interactive **Streamlit web application**, where users can enter vehicle specifications and receive an estimated mileage instantly.

## 🎯 Objectives

* Predict vehicle mileage using machine learning regression.
* Analyze the relationship between vehicle specifications and fuel efficiency.
* Compare multiple regression algorithms and identify the best-performing model.
* Build an interactive web application for real-time predictions.
* Implement a prediction pipeline consistent with the features used during model training.

## 🛠️ Tech Stack

| Category            | Technologies             |
| ------------------- | ------------------------ |
| Programming         | Python                   |
| Data Manipulation   | Pandas, NumPy            |
| Data Visualization  | Matplotlib, Seaborn      |
| Machine Learning    | Scikit-learn             |
| Model               | Random Forest Regressor  |
| Deployment          | Streamlit                |
| Model Serialization | Pickle                   |
| Development         | Google Colab, VS Code    |
| Dataset             | CarDekho Vehicle Dataset |

## 📊 Dataset

The dataset contains vehicle-related attributes used to predict mileage.

### Key Features

* Brand
* Model
* Engine Capacity
* Fuel Type
* Transmission Type
* Vehicle Age
* Kilometers Driven
* Maximum Power
* Number of Seats
* Selling Price
* Seller Type

### Target

**Mileage** — vehicle fuel efficiency measured in `km/l` or `km/kg`, depending on the fuel type.

## 🔄 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Selection
     ↓
Categorical Encoding
     ↓
Train-Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Best Model Selection
     ↓
Model Serialization
     ↓
Streamlit Deployment
```

## 🤖 Models Compared

Multiple regression models were trained and evaluated:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* XGBoost Regressor
* Support Vector Regression (SVR)

### Model Performance

| Model             |    Test R² |
| ----------------- | ---------: |
| Linear Regression |     0.8686 |
| Decision Tree     |     0.9592 |
| Random Forest     | **0.9759** |
| XGBoost           |     0.9728 |
| SVR               |     0.9210 |

**Random Forest Regressor achieved the highest test R² of approximately 0.976 among the evaluated models.**

## 🧠 Feature Engineering & Preprocessing

The project includes:

* Removal of irrelevant columns.
* Duplicate and missing-value analysis.
* Numerical feature handling.
* Categorical feature encoding using **One-Hot Encoding**.
* Train-test splitting for model evaluation.
* Feature alignment during inference to ensure the deployed model receives the same feature structure used during training.

The final model uses **163 input features** after categorical encoding.

## 🌐 Streamlit Application

The trained model is deployed through an interactive Streamlit application called:

### **AutoMile AI | Fuel Efficiency Platform**

Users can enter:

* Brand
* Model
* Engine Capacity
* Fuel Type
* Transmission
* Vehicle Age
* Kilometers Driven

The application then generates an estimated vehicle mileage.

### Application Features

* Interactive vehicle selection
* Dynamic model selection based on brand
* Automatic retrieval of supporting vehicle attributes
* Real-time ML inference
* Mileage displayed in the appropriate unit
* Estimated city and highway mileage indicators
* Clean responsive dashboard interface

## 📸 Application Preview

*Add your Streamlit application screenshot here.*

```text
![AutoMile AI Screenshot](screenshots/app.png)
```

## 📁 Project Structure

```text
AutoMile-AI/
│
├── cardekho_dataset.csv
├── car.ipynb
├── model.pkl
├── app.py
├── requirements.txt
├── README.md
│
└── screenshots/
    └── app.png
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/archana-o-p/AutoMile-AI.git
cd AutoMile-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📈 Key Result

The **Random Forest Regressor** achieved a test **R² score of approximately 0.976**, demonstrating strong predictive performance on the evaluation dataset.

The project also demonstrates the complete transition from:

**Machine Learning Model → Saved Model → Interactive Web Application**

## 💡 Key Learning Outcomes

Through this project, I worked on:

* End-to-end machine learning workflow
* Regression model development
* Exploratory Data Analysis
* Feature engineering
* One-hot encoding
* Model comparison and evaluation
* Model serialization using Pickle
* Building ML inference pipelines
* Streamlit application development
* Deploying a machine learning model as an interactive application

## 👩‍💻 Author

**Archana O.P.**

* GitHub: https://github.com/archana-o-p
* LinkedIn: https://www.linkedin.com/in/archana-op/

## ⭐ Future Improvements

* Improve handling of unseen vehicle categories.
* Add additional real-world vehicle data.
* Deploy the application using Streamlit Community Cloud or another cloud platform.
