# � Amazon Review Classification Dashboard

A comprehensive and interactive Streamlit application for Amazon review authenticity classification using machine learning models with advanced analytics and visualizations.

## 🚀 Features

### Core Functionality
- **Single Review Classification**: Classify individual Amazon reviews with confidence scores
- **Batch Analysis**: Process multiple reviews simultaneously
- **Real-time Analytics**: Interactive dashboard with prediction history
- **Model Insights**: Detailed information about the classification model

### Advanced Analytics
- **Confidence Scoring**: Visual gauge showing prediction confidence
- **Review Feature Analysis**: Radar charts and statistical breakdowns
- **Prediction Timeline**: Track classification history over time
- **Feature Visualization**: Interactive charts using Plotly

### User Experience
- **Multiple Input Methods**: Text area, file upload, or sample reviews
- **Responsive Design**: Beautiful UI with custom CSS styling
- **Download Results**: Export batch analysis results as CSV
- **Real-time Updates**: Live dashboard updates with new predictions

## 📋 Requirements

- Python 3.7 or higher
- Required packages (see requirements.txt):
  - streamlit==1.28.0
  - joblib==1.3.2
  - pandas==2.0.3
  - numpy==1.24.3
  - plotly==5.17.0
  - scikit-learn==1.3.0

## 🛠️ Installation

### Method 1: Automatic Setup (Recommended)
1. Double-click `setup.bat` to automatically install all dependencies
2. Double-click `run_app.bat` to start the application

### Method 2: Manual Setup
1. Open Command Prompt or PowerShell in the project directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📁 File Structure

```
EMAIL SPAM CLASSIFICATION/
├── app.py                          # Main Streamlit application
├── Logistic_Regression_best.joblib # Trained classification model
├── tfidf_vectorizer.joblib         # TF-IDF vectorizer
├── requirements.txt                # Python dependencies
├── setup.bat                       # Windows setup script
├── run_app.bat                     # Windows run script
└── README.md                       # This file
```

## 🎯 How to Use

### 1. Single Review Classification
- Navigate to the "🔍 Single Review Analysis" tab
- Choose input method:
  - **Text Area**: Type or paste Amazon review directly
  - **File Upload**: Upload a .txt file containing reviews
  - **Sample Reviews**: Select from predefined Amazon review examples
- Click "🔍 Classify Review" to get results
- View confidence scores and detailed analysis

### 2. Batch Analysis
- Go to "📊 Batch Analysis" tab
- Enter multiple Amazon reviews (one per line)
- Click "🔍 Analyze Batch" for bulk processing
- Download results as CSV file

### 3. Analytics Dashboard
- Visit "📈 Analytics Dashboard" tab
- View prediction distribution charts
- Analyze timeline of classifications
- Review prediction history table

### 4. Model Information
- Check "📚 Model Insights" tab for:
  - Algorithm details
  - Usage tips
  - Model limitations
  - File information

## 🔧 Model Details

- **Algorithm**: Logistic Regression
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classes**: 
  - 0: Fake Review
  - 1: Genuine Review
- **Features Analyzed**:
  - Character count
  - Word count
  - Sentence count
  - Exclamation marks
  - Question marks
  - Uppercase words
  - Special characters
  - Average word length

## 📊 Visualizations

The app includes various interactive charts:
- **Confidence Gauge**: Real-time confidence scoring
- **Radar Charts**: Review feature analysis
- **Pie Charts**: Prediction distribution
- **Bar Charts**: Classification probabilities
- **Timeline Charts**: Historical prediction tracking
- **Histograms**: Confidence score distributions

## 🎨 UI Features

- **Responsive Design**: Works on desktop and tablet devices
- **Custom Styling**: Modern gradient backgrounds and animations
- **Interactive Elements**: Hover effects and dynamic updates
- **Color Coding**: Red for fake reviews, green for genuine reviews
- **Progress Indicators**: Loading spinners for better UX

## ⚡ Performance Tips

- For best results, use complete review sentences
- Consider product context when interpreting results
- Review confidence scores for uncertain predictions
- Regular model retraining recommended for optimal performance

## 🔍 Common Fake Review Indicators

The model looks for these patterns:
- Excessive capital letters and exclamation marks
- Overly enthusiastic language without specific details
- Generic praise without product specifics
- Repetitive phrases and unnatural language patterns
- Extreme ratings without balanced feedback
- Lack of specific product features or experiences

## 🐛 Troubleshooting

### Model Loading Issues
- Ensure both `.joblib` files are in the correct directory
- Check file permissions
- Verify Python and package versions

### Performance Issues
- Clear prediction history if it becomes too large
- Reduce batch size for large review collections
- Check system memory for very long reviews

### Display Issues
- Refresh the browser page
- Clear browser cache
- Ensure JavaScript is enabled

## 📞 Support

If you encounter any issues:
1. Check the requirements.txt file matches your installed packages
2. Verify Python version compatibility
3. Ensure model files are not corrupted
4. Check console for detailed error messages

## 🔄 Updates

To update the application:
1. Replace the model files with newer versions
2. Update requirements.txt if needed
3. Restart the application

---

**Note**: This application is designed for educational and demonstration purposes. For production use, consider additional security measures and model validation.
