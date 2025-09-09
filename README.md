# Student Score Prediction

A machine learning application that predicts student exam scores based on various academic, personal, and environmental factors.

## Features

- **Linear Regression Model**: Basic prediction model
- **Polynomial Regression Model**: Enhanced model with polynomial features
- **Interactive Web Interface**: Clean Streamlit GUI for user input
- **Comprehensive Analysis**: 13 different factors affecting student performance
- **Visual Analytics**: Radar charts and factor contribution analysis
- **Performance Comparison**: Side-by-side model comparison

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Jupyter Notebook (Optional)
Open and run `Student Score prediction.ipynb` to see the complete analysis.

### 3. Launch Streamlit App
```bash
streamlit run app.py
```

## Input Parameters

### Academic Factors
- Hours studied per day (1-12 hours)
- Attendance percentage (50-100%)
- Previous scores (40-100%)

### Family Background
- Parental education level (1-5 scale)
- Family income level (1-5 scale)

### Lifestyle Factors
- Sleep hours per night (4-12 hours)
- Extracurricular activities (Yes/No)

### Resources & Environment
- Access to resources (1-5 scale)
- Motivation level (1-5 scale)
- Internet access (Yes/No)

### Personal Information
- Distance from home (1-50 km)
- Gender (Male/Female)
- Age (15-18 years)

## Model Performance

- **Linear Regression R²**: ~0.25
- **Polynomial Regression R²**: ~0.30
- Models trained on 1000 synthetic student records

## File Structure

```
Student Score Prediction/
├── Student Score prediction.ipynb    # Complete analysis notebook
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies                 # Model performance metrics
└── README.md                        # This file
```

## Usage

1. Run the Streamlit app: `streamlit run app.py`
2. Adjust student parameters using the sidebar controls
3. View predictions from both linear and polynomial models
4. Analyze factor contributions and student profile
5. Review personalized recommendations

## Technology Stack

- **Backend**: Python, scikit-learn
- **Frontend**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy

## Model Details

The application uses synthetic data modeling realistic student performance factors. Both linear and polynomial regression models are trained to predict exam scores (0-100 scale) based on the input parameters.

### Key Findings
- Study hours have the highest impact on performance
- Internet access and motivation are significant factors
- Polynomial features improve prediction accuracy by ~20%

## Created by:

Aatiqa Sadiq for Elevvo Pathways
