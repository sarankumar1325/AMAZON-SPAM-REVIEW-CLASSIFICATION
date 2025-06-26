import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import re
from datetime import datetime
import os

# Set page configuration
st.set_page_config(
    page_title="Amazon Review Classification Dashboard",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .spam-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    .not-spam-alert {
        background-color: #00cc66;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'text_analysis' not in st.session_state:
    st.session_state.text_analysis = {}

@st.cache_resource
def load_models():
    """Load the trained model and vectorizer"""
    try:
        # Try absolute paths first
        model_path = r"c:\Users\GANES\OneDrive\Desktop\ML\NLP\EMAIL SPAM CLASSIFICATION\Logistic_Regression_best.joblib"
        vectorizer_path = r"c:\Users\GANES\OneDrive\Desktop\ML\NLP\EMAIL SPAM CLASSIFICATION\tfidf_vectorizer.joblib"
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
        else:
            # Fallback to relative paths
            model = joblib.load('Logistic_Regression_best.joblib')
            vectorizer = joblib.load('tfidf_vectorizer.joblib')
        
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def analyze_text_features(text):
    """Analyze various features of the input text"""
    features = {
        'character_count': len(text),
        'word_count': len(text.split()),
        'sentence_count': len(re.split(r'[.!?]+', text)),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_words': len([word for word in text.split() if word.isupper() and len(word) > 1]),
        'digit_count': sum(c.isdigit() for c in text),
        'special_chars': len(re.findall(r'[^a-zA-Z0-9\s]', text)),
        'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0
    }
    return features

def get_prediction_confidence(model, X_transformed):
    """Get prediction probabilities for confidence scoring"""
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_transformed)
            return probabilities[0]
        else:
            # Fallback for models without predict_proba
            decision = model.decision_function(X_transformed)
            # Convert decision function to probability-like scores
            prob_spam = 1 / (1 + np.exp(-decision[0]))
            return [1 - prob_spam, prob_spam]
    except:
        return [0.5, 0.5]

def create_feature_radar_chart(features):
    """Create a radar chart for text features"""
    # Normalize features for better visualization
    max_values = {
        'character_count': 500,
        'word_count': 100,
        'exclamation_count': 10,
        'question_count': 5,
        'uppercase_words': 20,
        'digit_count': 50,
        'special_chars': 30
    }
    
    categories = []
    values = []
    
    for feature, value in features.items():
        if feature in max_values:
            categories.append(feature.replace('_', ' ').title())
            normalized_value = min(value / max_values[feature], 1.0) * 100
            values.append(normalized_value)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Text Features',
        line_color='rgb(255, 107, 107)',
        fillcolor='rgba(255, 107, 107, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Text Feature Analysis",
        height=400
    )
    
    return fig

def create_confidence_gauge(confidence_score, prediction):
    """Create a gauge chart for prediction confidence"""
    color = 'red' if prediction == 0 else 'green'
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Prediction Confidence"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_history_charts():
    """Create charts for prediction history"""
    if not st.session_state.prediction_history:
        return None, None
    
    df = pd.DataFrame(st.session_state.prediction_history)
    
    # Prediction distribution pie chart
    prediction_counts = df['prediction'].value_counts()
    labels = ['Genuine Review', 'Fake Review']
    values = [prediction_counts.get(1, 0), prediction_counts.get(0, 0)]
    colors = ['#00cc66', '#ff4b4b']
    
    pie_fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=.3,
        marker_colors=colors
    )])
    pie_fig.update_layout(
        title="Prediction Distribution",
        height=300
    )
    
    # Timeline chart
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    timeline_fig = go.Figure()
    
    fake_data = df[df['prediction'] == 0]
    genuine_data = df[df['prediction'] == 1]
    
    if not fake_data.empty:
        timeline_fig.add_trace(go.Scatter(
            x=fake_data['timestamp'],
            y=fake_data['confidence'],
            mode='markers',
            name='Fake Review',
            marker=dict(color='red', size=10),
            text=fake_data['text_preview'],
            hovertemplate='<b>Fake Review</b><br>Confidence: %{y:.2f}<br>Text: %{text}<extra></extra>'
        ))
    
    if not genuine_data.empty:
        timeline_fig.add_trace(go.Scatter(
            x=genuine_data['timestamp'],
            y=genuine_data['confidence'],
            mode='markers',
            name='Genuine Review',
            marker=dict(color='green', size=10),
            text=genuine_data['text_preview'],
            hovertemplate='<b>Genuine Review</b><br>Confidence: %{y:.2f}<br>Text: %{text}<extra></extra>'
        ))
    
    timeline_fig.update_layout(
        title="Prediction Timeline",
        xaxis_title="Time",
        yaxis_title="Confidence Score",
        height=400
    )
    
    return pie_fig, timeline_fig

def main():
    # Header
    st.markdown('<div class="main-header">� Amazon Review Classification Dashboard</div>', unsafe_allow_html=True)
    
    # Load models
    model, vectorizer = load_models()
    
    if model is None or vectorizer is None:
        st.error("❌ Failed to load models. Please check if the model files exist in the correct directory.")
        st.info("Expected files: Logistic_Regression_best.joblib and tfidf_vectorizer.joblib")
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.write("**Model Type:** Logistic Regression")
        st.write("**Vectorizer:** TF-IDF")
        st.write("**Classes:** Fake Review (0) | Genuine Review (1)")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.header("🎛️ Settings")
        show_advanced = st.checkbox("Show Advanced Analytics", value=True)
        auto_analyze = st.checkbox("Auto-analyze text features", value=True)
        
        st.header("📈 Quick Stats")
        total_predictions = len(st.session_state.prediction_history)
        fake_count = sum(1 for p in st.session_state.prediction_history if p['prediction'] == 0)
        genuine_count = total_predictions - fake_count
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Predictions", total_predictions)
        with col2:
            st.metric("Fake Reviews Detected", fake_count)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Review Analysis", "📊 Batch Analysis", "📈 Analytics Dashboard", "📚 Model Insights"])
    
    with tab1:
        st.header("Single Review Classification")
        
        # Input methods
        input_method = st.radio("Choose input method:", ["Text Area", "File Upload", "Sample Reviews"])
        
        user_text = ""
        
        if input_method == "Text Area":
            user_text = st.text_area(
                "Enter Amazon review to classify:",
                placeholder="Type or paste your Amazon review here...",
                height=150
            )
        
        elif input_method == "File Upload":
            uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
            if uploaded_file is not None:
                user_text = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", value=user_text, height=150)
        
        elif input_method == "Sample Reviews":
            sample_texts = [
                "This product is absolutely amazing! I love everything about it and would definitely recommend to everyone!",
                "The product arrived quickly and works as expected. Good quality for the price.",
                "BEST PRODUCT EVER!!! 5 STARS!!! HIGHLY RECOMMEND!!! AMAZING QUALITY!!!",
                "Decent product, does what it's supposed to do. Nothing spectacular but gets the job done.",
                "Perfect! Amazing! Incredible! This changed my life! Everyone should buy this now!",
                "Good value for money. Packaging could be better but the product itself is solid."
            ]
            selected_sample = st.selectbox("Choose a sample review:", [""] + sample_texts)
            if selected_sample:
                user_text = selected_sample
                st.text_area("Selected review:", value=user_text, height=100)
        
        if st.button("🔍 Classify Review", type="primary", use_container_width=True):
            if user_text.strip():
                with st.spinner("Analyzing review..."):
                    # Transform and predict
                    X_transformed = vectorizer.transform([user_text])
                    prediction = model.predict(X_transformed)[0]
                    confidence_scores = get_prediction_confidence(model, X_transformed)
                    confidence = max(confidence_scores)
                    
                    # Analyze text features
                    features = analyze_text_features(user_text)
                    
                    # Store in history
                    st.session_state.prediction_history.append({
                        'text': user_text,
                        'text_preview': user_text[:50] + "..." if len(user_text) > 50 else user_text,
                        'prediction': prediction,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat(),
                        'features': features
                    })
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if prediction == 0:
                        st.markdown(f'<div class="spam-alert">🚨 FAKE REVIEW DETECTED! 🚨<br>Confidence: {confidence:.2%}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="not-spam-alert">✅ GENUINE REVIEW<br>Confidence: {confidence:.2%}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.plotly_chart(create_confidence_gauge(confidence, prediction), use_container_width=True)
                
                if show_advanced:
                    st.subheader("📊 Detailed Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Text Statistics")
                        stats_df = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
                        st.dataframe(stats_df, use_container_width=True)
                    
                    with col2:
                        st.subheader("Feature Visualization")
                        radar_chart = create_feature_radar_chart(features)
                        st.plotly_chart(radar_chart, use_container_width=True)
                    
                    # Probability breakdown
                    st.subheader("Prediction Probabilities")
                    prob_df = pd.DataFrame({
                        'Class': ['Fake Review', 'Genuine Review'],
                        'Probability': confidence_scores
                    })
                    
                    fig_bar = px.bar(
                        prob_df, 
                        x='Class', 
                        y='Probability',
                        color='Class',
                        color_discrete_map={'Fake Review': '#ff4b4b', 'Genuine Review': '#00cc66'},
                        title="Classification Probabilities"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("⚠️ Please enter a review to classify.")
    
    with tab2:
        st.header("Batch Review Analysis")
        
        batch_input = st.text_area(
            "Enter multiple reviews (one per line):",
            placeholder="Enter multiple Amazon reviews, each on a new line...",
            height=200
        )
        
        if st.button("🔍 Analyze Batch", type="primary"):
            if batch_input.strip():
                texts = [text.strip() for text in batch_input.split('\n') if text.strip()]
                
                if texts:
                    with st.spinner(f"Analyzing {len(texts)} reviews..."):
                        results = []
                        
                        for i, text in enumerate(texts):
                            X_transformed = vectorizer.transform([text])
                            prediction = model.predict(X_transformed)[0]
                            confidence_scores = get_prediction_confidence(model, X_transformed)
                            confidence = max(confidence_scores)
                            
                            results.append({
                                'Index': i + 1,
                                'Review Preview': text[:50] + "..." if len(text) > 50 else text,
                                'Full Review': text,
                                'Prediction': 'Fake Review' if prediction == 0 else 'Genuine Review',
                                'Confidence': f"{confidence:.2%}",
                                'Confidence_Score': confidence,
                                'Class': prediction
                            })
                    
                    results_df = pd.DataFrame(results)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Reviews", len(results))
                    with col2:
                        fake_count = len([r for r in results if r['Class'] == 0])
                        st.metric("Fake Reviews", fake_count)
                    with col3:
                        genuine_count = len(results) - fake_count
                        st.metric("Genuine Reviews", genuine_count)
                    with col4:
                        avg_confidence = np.mean([r['Confidence_Score'] for r in results])
                        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    display_df = results_df[['Index', 'Review Preview', 'Prediction', 'Confidence']].copy()
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribution chart
                        fig_dist = px.pie(
                            values=[fake_count, genuine_count],
                            names=['Fake Review', 'Genuine Review'],
                            title="Classification Distribution",
                            color_discrete_map={'Fake Review': '#ff4b4b', 'Genuine Review': '#00cc66'}
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with col2:
                        # Confidence distribution
                        fig_conf = px.histogram(
                            results_df,
                            x='Confidence_Score',
                            color='Prediction',
                            title="Confidence Score Distribution",
                            color_discrete_map={'Fake Review': '#ff4b4b', 'Genuine Review': '#00cc66'}
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"review_classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ Please enter at least one review to analyze.")
            else:
                st.warning("⚠️ Please enter some reviews to analyze.")
    
    with tab3:
        st.header("Analytics Dashboard")
        
        if not st.session_state.prediction_history:
            st.info("📊 No predictions yet. Analyze some reviews to see analytics!")
        else:
            # Create charts
            pie_fig, timeline_fig = create_history_charts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if pie_fig:
                    st.plotly_chart(pie_fig, use_container_width=True)
            
            with col2:
                # Feature analysis across all predictions
                if st.session_state.prediction_history:
                    all_features = [p['features'] for p in st.session_state.prediction_history if 'features' in p]
                    if all_features:
                        feature_names = list(all_features[0].keys())
                        feature_avgs = {name: np.mean([f[name] for f in all_features]) for name in feature_names}
                        
                        fig_features = px.bar(
                            x=list(feature_avgs.keys()),
                            y=list(feature_avgs.values()),
                            title="Average Text Features",
                            labels={'x': 'Features', 'y': 'Average Value'}
                        )
                        fig_features.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_features, use_container_width=True)
            
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Detailed history table
            st.subheader("📝 Prediction History")
            history_df = pd.DataFrame(st.session_state.prediction_history)
            if not history_df.empty:
                display_history = history_df[['text_preview', 'prediction', 'confidence', 'timestamp']].copy()
                display_history['prediction'] = display_history['prediction'].map({0: 'Fake Review', 1: 'Genuine Review'})
                display_history['confidence'] = display_history['confidence'].apply(lambda x: f"{x:.2%}")
                display_history.columns = ['Review Preview', 'Prediction', 'Confidence', 'Timestamp']
                st.dataframe(display_history, use_container_width=True)
                
                # Clear history button
                if st.button("🗑️ Clear History"):
                    st.session_state.prediction_history = []
                    st.rerun()
    
    with tab4:
        st.header("Model Insights & Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Model Details")
            st.write("**Algorithm:** Logistic Regression")
            st.write("**Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)")
            st.write("**Output Classes:**")
            st.write("- 0: Fake Review")
            st.write("- 1: Genuine Review")
            
            st.subheader("📊 How It Works")
            st.write("""
            1. **Text Preprocessing:** The input review is cleaned and tokenized
            2. **TF-IDF Vectorization:** Converts review text to numerical features
            3. **Classification:** Logistic regression determines authenticity probability
            4. **Confidence Score:** Based on prediction probabilities
            """)
        
        with col2:
            st.subheader("🎯 Usage Tips")
            st.write("""
            **For Best Results:**
            - Include complete review sentences when possible
            - Consider context and product type
            - Review confidence scores for uncertain predictions
            
            **Common Fake Review Indicators:**
            - Excessive use of capital letters and exclamation marks
            - Overly enthusiastic language without specific details
            - Generic praise without product specifics
            - Repetitive phrases and unnatural language patterns
            - Extreme ratings without balanced feedback
            """)
            
            st.subheader("⚠️ Limitations")
            st.write("""
            - Model performance depends on training data quality
            - May struggle with very short reviews
            - Context-dependent classifications might vary
            - Regular retraining recommended for optimal performance
            - Cultural and language nuances may affect accuracy
            """)
        
        # Model file info
        st.subheader("📁 Model Files")
        try:
            model_path = r"c:\Users\GANES\OneDrive\Desktop\ML\NLP\EMAIL SPAM CLASSIFICATION\Logistic_Regression_best.joblib"
            vectorizer_path = r"c:\Users\GANES\OneDrive\Desktop\ML\NLP\EMAIL SPAM CLASSIFICATION\tfidf_vectorizer.joblib"
            
            if os.path.exists(model_path):
                model_size = os.path.getsize(model_path) / 1024  # KB
                st.write(f"**Model File:** {model_size:.2f} KB")
            
            if os.path.exists(vectorizer_path):
                vectorizer_size = os.path.getsize(vectorizer_path) / 1024  # KB
                st.write(f"**Vectorizer File:** {vectorizer_size:.2f} KB")
        except:
            st.write("File size information not available")

if __name__ == "__main__":
    main()
