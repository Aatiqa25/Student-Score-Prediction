import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Student Score Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2a5298;
        margin: 0.5rem 0;
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .info-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    
    .creator-banner {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 25px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        font-size: 0.9rem;
        font-weight: 500;
        z-index: 1000;
        border: 2px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .creator-banner:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        linear_model = joblib.load('linear_model.pkl')
        poly_model = joblib.load('poly_model.pkl')
        poly_features = joblib.load('poly_features.pkl')
        feature_names = joblib.load('feature_names.pkl')
        model_info = joblib.load('model_info.pkl')
        encoders = joblib.load('encoders.pkl')
        return linear_model, poly_model, poly_features, feature_names, model_info, encoders
    except FileNotFoundError:
        st.error("model files not found. please run the jupyter notebook first.")
        st.stop()

st.markdown('<h1 class="main-header">Student Score Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">predict exam performance using machine learning for Elevvo Pathways</p>', unsafe_allow_html=True)

linear_model, poly_model, poly_features, feature_names, model_info, encoders = load_models()

st.sidebar.markdown("### Student information")
st.sidebar.markdown("---")

st.sidebar.markdown("#### Academic factors")
hours_studied = st.sidebar.slider("Hours studied", 1, 30, 20)
attendance = st.sidebar.slider("Attendance percentage", 50, 100, 85)
previous_scores = st.sidebar.slider("Previous scores", 40, 100, 75)
tutoring_sessions = st.sidebar.slider("Tutoring sessions", 0, 8, 2)

st.sidebar.markdown("#### Family & Social factors")
parental_involvement = st.sidebar.selectbox("Parental involvement", 
                                           options=['Low', 'Medium', 'High'], 
                                           index=1)

family_income = st.sidebar.selectbox("Family income", 
                                    options=['Low', 'Medium', 'High'], 
                                    index=1)

parental_education_level = st.sidebar.selectbox("Parental education level", 
                                               options=['High School', 'College', 'Postgraduate'], 
                                               index=1)

peer_influence = st.sidebar.selectbox("Peer influence", 
                                     options=['Negative', 'Neutral', 'Positive'], 
                                     index=1)

st.sidebar.markdown("#### Lifestyle factors")
sleep_hours = st.sidebar.slider("Sleep hours per night", 4, 12, 7)
extracurricular_activities = st.sidebar.selectbox("Extracurricular activities", 
                                                 options=['No', 'Yes'], 
                                                 index=1)
physical_activity = st.sidebar.slider("Physical activity hours per week", 0, 10, 3)

st.sidebar.markdown("#### Resources & Environment")
access_to_resources = st.sidebar.selectbox("Access to resources", 
                                          options=['Low', 'Medium', 'High'], 
                                          index=1)

motivation_level = st.sidebar.selectbox("Motivation level", 
                                       options=['Low', 'Medium', 'High'], 
                                       index=1)

internet_access = st.sidebar.selectbox("Internet access", 
                                      options=['No', 'Yes'], 
                                      index=1)

teacher_quality = st.sidebar.selectbox("Teacher quality", 
                                      options=['Low', 'Medium', 'High'], 
                                      index=1)

school_type = st.sidebar.selectbox("School type", 
                                  options=['Public', 'Private'], 
                                  index=0)

st.sidebar.markdown("#### Personal information")
distance_from_home = st.sidebar.selectbox("distance from home", 
                                         options=['Near', 'Moderate', 'Far'], 
                                         index=1)

gender = st.sidebar.selectbox("Gender", 
                             options=['Male', 'Female'], 
                             index=0)

learning_disabilities = st.sidebar.selectbox("Learning disabilities", 
                                            options=['No', 'Yes'], 
                                            index=0)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### prediction results")
    
    input_data = {
        'Hours_Studied': hours_studied,
        'Attendance': attendance,
        'Parental_Involvement_encoded': encoders['Parental_Involvement'].transform([parental_involvement])[0],
        'Access_to_Resources_encoded': encoders['Access_to_Resources'].transform([access_to_resources])[0],
        'Extracurricular_Activities_encoded': encoders['Extracurricular_Activities'].transform([extracurricular_activities])[0],
        'Sleep_Hours': sleep_hours,
        'Previous_Scores': previous_scores,
        'Motivation_Level_encoded': encoders['Motivation_Level'].transform([motivation_level])[0],
        'Internet_Access_encoded': encoders['Internet_Access'].transform([internet_access])[0],
        'Tutoring_Sessions': tutoring_sessions,
        'Family_Income_encoded': encoders['Family_Income'].transform([family_income])[0],
        'Teacher_Quality_encoded': encoders['Teacher_Quality'].transform([teacher_quality])[0],
        'School_Type_encoded': encoders['School_Type'].transform([school_type])[0],
        'Peer_Influence_encoded': encoders['Peer_Influence'].transform([peer_influence])[0],
        'Physical_Activity': physical_activity,
        'Learning_Disabilities_encoded': encoders['Learning_Disabilities'].transform([learning_disabilities])[0],
        'Parental_Education_Level_encoded': encoders['Parental_Education_Level'].transform([parental_education_level])[0],
        'Distance_from_Home_encoded': encoders['Distance_from_Home'].transform([distance_from_home])[0],
        'Gender_encoded': encoders['Gender'].transform([gender])[0]
    }
    
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]
    
    linear_pred = linear_model.predict(input_df)[0]
    
    input_poly = poly_features.transform(input_df)
    poly_pred = poly_model.predict(input_poly)[0]
    
    col1_1, col1_2 = st.columns(2)
    
    with col1_1:
        st.markdown(f"""
        <div class="prediction-container">
            <h3>linear regression</h3>
            <h1 style="font-size: 2.5rem; margin: 0;">{linear_pred:.1f}</h1>
            <p>predicted score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col1_2:
        st.markdown(f"""
        <div class="prediction-container">
            <h3>polynomial regression</h3>
            <h1 style="font-size: 2.5rem; margin: 0;">{poly_pred:.1f}</h1>
            <p>predicted score</p>
        </div>
        """, unsafe_allow_html=True)
    
    def get_performance_level(score):
        if score >= 90: return "excellent", "#28a745"
        elif score >= 80: return "very good", "#20c997"
        elif score >= 70: return "good", "#ffc107"
        elif score >= 60: return "satisfactory", "#fd7e14"
        else: return "needs improvement", "#dc3545"
    
    linear_level, linear_color = get_performance_level(linear_pred)
    poly_level, poly_color = get_performance_level(poly_pred)
    
    st.markdown(f"""
    <div class="info-box">
        <h4>performance assessment</h4>
        <p><strong>linear model:</strong> <span style="color: {linear_color};">{linear_level}</span></p>
        <p><strong>polynomial model:</strong> <span style="color: {poly_color};">{poly_level}</span></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### input summary")
    
    summary_data = {
        'factor': [
            'study hours',
            'attendance',
            'previous scores',
            'sleep hours',
            'motivation',
            'resources',
            'tutoring',
            'family income'
        ],
        'value': [
            f"{hours_studied} hrs",
            f"{attendance}%",
            f"{previous_scores}%",
            f"{sleep_hours} hrs",
            motivation_level,
            access_to_resources,
            f"{tutoring_sessions} sessions",
            family_income
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("### model performance")
    st.markdown(f"""
    <div class="metric-box">
        <p><strong>linear regression r²:</strong> {model_info['linear_r2']:.4f}</p>
        <p><strong>polynomial regression r²:</strong> {model_info['poly_r2']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### factor analysis")

col1, col2 = st.columns(2)

with col1:
    categories = ['academic', 'family', 'lifestyle', 'resources', 'environment']
    
    academic_score = (hours_studied/30 + attendance/100 + previous_scores/100 + tutoring_sessions/8) / 4 * 100
    family_score = ({'High School': 1, 'College': 2, 'Postgraduate': 3}[parental_education_level]/3 + 
                   {'Low': 1, 'Medium': 2, 'High': 3}[family_income]/3 + 
                   {'Low': 1, 'Medium': 2, 'High': 3}[parental_involvement]/3) / 3 * 100
    lifestyle_score = (sleep_hours/12 + (1 if extracurricular_activities == 'Yes' else 0) + physical_activity/10) / 3 * 100
    resources_score = ({'Low': 1, 'Medium': 2, 'High': 3}[access_to_resources]/3 + 
                      {'Low': 1, 'Medium': 2, 'High': 3}[motivation_level]/3 + 
                      (1 if internet_access == 'Yes' else 0) + 
                      {'Low': 1, 'Medium': 2, 'High': 3}[teacher_quality]/3) / 4 * 100
    environment_score = ((1 if school_type == 'Private' else 0.5) + 
                        {'Negative': 0, 'Neutral': 0.5, 'Positive': 1}[peer_influence] + 
                        {'Far': 0.3, 'Moderate': 0.6, 'Near': 1}[distance_from_home]) / 3 * 100
    
    values = [academic_score, family_score, lifestyle_score, resources_score, environment_score]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='student profile',
        line=dict(color='#667eea'),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title="student profile radar",
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    feature_impact = {
        'hours studied': hours_studied,
        'attendance': attendance,
        'previous scores': previous_scores,
        'tutoring sessions': tutoring_sessions * 5,
        'motivation': {'Low': 20, 'Medium': 60, 'High': 100}[motivation_level],
        'resources': {'Low': 20, 'Medium': 60, 'High': 100}[access_to_resources],
        'teacher quality': {'Low': 20, 'Medium': 60, 'High': 100}[teacher_quality],
        'family income': {'Low': 20, 'Medium': 60, 'High': 100}[family_income]
    }
    
    impact_df = pd.DataFrame(list(feature_impact.items()), columns=['factor', 'impact'])
    impact_df = impact_df.sort_values('impact', ascending=True)
    
    fig = px.bar(impact_df, x='impact', y='factor', orientation='h',
                title='factor contribution to score',
                color='impact', color_continuous_scale='viridis')
    
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### recommendations")

recommendations = []

if hours_studied < 15:
    recommendations.append("increase daily study time to improve performance")
if attendance < 80:
    recommendations.append("focus on improving class attendance")
if sleep_hours < 6:
    recommendations.append("ensure adequate sleep for better cognitive function")
if motivation_level == 'Low':
    recommendations.append("work on building motivation and study habits")
if access_to_resources == 'Low':
    recommendations.append("seek additional educational resources")
if tutoring_sessions < 2:
    recommendations.append("consider additional tutoring sessions")
if teacher_quality == 'Low':
    recommendations.append("seek additional academic support or resources")
if physical_activity < 2:
    recommendations.append("include more physical activity for better overall health")

if not recommendations:
    recommendations.append("excellent profile! maintain current study patterns")

for i, rec in enumerate(recommendations, 1):
    st.markdown(f"""
    <div class="info-box">
        <p><strong>{i}.</strong> {rec}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>developed using machine learning for educational insights</p>
</div>
""", unsafe_allow_html=True)

# Floating creator banner
st.markdown("""
<div class="creator-banner">
    Created by Aatiqa Sadiq
</div>
""", unsafe_allow_html=True)