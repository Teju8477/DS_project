# Tourism Expenditure Prediction

## Project Overview
Tourism Expenditure Prediction is a data science and machine learning project that estimates tourism-related expenditure based on selected travel and destination details.

The project uses tourism expenditure data from the Domestic Tourism Expenditure Survey (DTES) and provides a simple, interactive web application built using Streamlit.

## Objectives
- Analyze tourism expenditure data.
- Explore patterns in tourism spending across Indian states and districts.
- Build a machine learning model to predict tourism expenditure.
- Provide an interactive interface for users to explore estimated travel costs.

## Technologies Used
- **Python** — programming language
- **Pandas and NumPy** — data processing
- **Matplotlib and Seaborn** — exploratory data analysis and visualization
- **Scikit-learn** — machine learning
- **Streamlit** — interactive web application
- **Git and GitHub** — version control and project hosting

## Dataset
The project uses data and documentation from the Domestic Tourism Expenditure Survey (DTES).

The dataset includes tourism expenditure information associated with geographical categories such as states and districts.

**Note:** The dataset's original fields and survey definitions determine which features can be used by the prediction model. Additional travel inputs in the web application should be treated as user-interface features unless they are incorporated into and supported by the trained model.

## Application Features
- Interactive tourism expenditure prediction form.
- State and district selection, where supported by the dataset.
- Estimated expenditure displayed to the user.
- Additional travel-planning questions to make the interface more interactive.
- Simple web interface powered by Streamlit.

## Machine Learning
The project uses a regression-based machine learning approach to estimate tourism expenditure.

Model performance is evaluated using:
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

Model accuracy depends on the available features, data quality, and the relationship between the input variables and tourism expenditure.

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/Teju8477/DS_project.git
cd DS_project
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn openpyxl
```

### 3. Run the application
```bash
streamlit run app.py
```

If your Streamlit entry-point file has a different name, replace `app.py` with the actual filename.

## Future Enhancements
- Improve prediction accuracy through feature engineering and model comparison.
- Add more destination and travel-related inputs where suitable data is available.
- Improve visualizations and the user interface.
- Deploy the application online.
- Add model evaluation results and example predictions.

## Project Purpose
This project was developed as an academic data science project to demonstrate data analysis, machine learning, and interactive application development.

## Author
Developed as part of an academic data science project.
