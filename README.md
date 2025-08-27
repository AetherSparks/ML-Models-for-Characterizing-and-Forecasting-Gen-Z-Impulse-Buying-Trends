# Machine Learning Models for Characterizing and Forecasting Impulse Buying Trends in Generation Z

## 📋 Project Overview

This research project investigates how machine learning models can effectively characterize and forecast impulse buying trends in Generation Z using diverse data sources. The study combines e-commerce product data, psychological survey responses, and social media sentiment analysis to understand consumer behavior patterns.

## 🎯 Research Objectives

- **Baseline Characterization**: Build interpretable ML baselines to classify and predict impulse buying signals
- **Cross-Dataset Validation**: Demonstrate that different data sources provide complementary insights
- **Performance Demonstration**: Show that advanced models achieve higher accuracy than traditional baselines
- **Forecasting Support**: Illustrate model extensibility for consumer behavior trend forecasting

## 🔬 Research Question

*How can machine learning models effectively characterize and forecast impulse buying trends in Generation Z, using diverse data sources (e-commerce, social media, and behavioral survey responses)?*

## 📊 Hypotheses

- **H1 (Textual Signals)**: Consumer-facing text data contain latent sentiment cues correlating with impulse buying tendencies
- **H2 (Survey Behavioral)**: Psychological features from surveys can be encoded to capture impulse buying propensities
- **H3 (Model Performance)**: Advanced models outperform classical baselines in identifying impulse buying signals
- **H4 (Cross-Domain)**: Multiple data sources converge into shared predictive patterns across domains

## 🗂️ Dataset Information

### 1. E-commerce Product Dataset
- **Files**: `X_train_update.csv`, `Y_train_CVw08PX.csv`, `X_test_update.csv`
- **Size**: 84,916 training samples, 13,812 test samples
- **Features**: Product designations, descriptions, product IDs, image IDs
- **Target**: Product type codes (27 categories)
- **Source**: E-commerce platform product catalog data

### 2. Psychological Survey Dataset  
- **File**: `Raw data_Impulse buying behavior.xlsx`
- **Size**: 361 responses, 32 features
- **Features**: Likert-scale responses (SC1-SC4, SI1-SI5, TR1, etc.)
- **Target**: Impulse buying tendency (Low/Medium/High)
- **Source**: Gen Z consumer psychology questionnaire

### 3. Social Media Dataset
- **Status**: Pending upload - framework ready for integration
- **Expected Features**: Tweet text, sentiment labels, engagement metrics
- **Purpose**: Social media sentiment analysis for impulse buying signals

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost
```

### Quick Start

1. **Clone/Download** the project files to your local directory
2. **Ensure datasets** are in the same folder as the scripts
3. **Run the optimized pipeline**:

```bash
python run_research_optimized.py
```

## 📁 Project Structure

```
idk what this is/
├── README.md                              # This file
├── run_research_optimized.py             # Main optimized pipeline (RECOMMENDED)
├── run_research_pipeline_real_data.py    # Full-featured pipeline
├── Research_paper.ipynb                  # Jupyter notebook version
├── X_train_update.csv                    # E-commerce training features
├── Y_train_CVw08PX.csv                   # E-commerce training labels  
├── X_test_update.csv                     # E-commerce test features
├── Raw data_Impulse buying behavior.xlsx # Survey responses
├── Questionnaire_Impulse buying behavior.pdf # Survey documentation
├── confusion_matrices/                   # Individual confusion matrix images
│   ├── confusion_matrix_ecommerce_XGBoost_[timestamp].png
│   ├── confusion_matrix_ecommerce_LightGBM_[timestamp].png
│   ├── confusion_matrix_ecommerce_CatBoost_[timestamp].png
│   ├── confusion_matrix_ecommerce_Random_Forest_[timestamp].png
│   ├── confusion_matrix_ecommerce_Logistic_Regression_[timestamp].png
│   └── confusion_matrix_ecommerce_Voting_Ensemble_[timestamp].png
├── visualizations/                       # Performance comparison graphs
│   ├── accuracy_comparison_[timestamp].png
│   ├── f1_score_comparison_[timestamp].png
│   └── performance_heatmap_[timestamp].png
└── Gen_Z_Research_Report_[timestamp].txt # Comprehensive research report
```

## 🛠️ Available Scripts

### 1. `run_research_optimized.py` ⭐ **RECOMMENDED**
- **Purpose**: Fast, efficient pipeline for quick results
- **Dataset Size**: 25,000 samples (optimized for speed vs. accuracy)
- **Features**: 2,000 TF-IDF features with bigrams
- **Runtime**: ~5-10 minutes
- **Output**: Individual confusion matrices, comparison graphs, research report

### 2. `run_research_pipeline_real_data.py`
- **Purpose**: Full-scale research pipeline with all features
- **Dataset Size**: Full dataset (84,916 samples)
- **Features**: Comprehensive analysis with all available data
- **Runtime**: ~30-60 minutes
- **Output**: Complete research artifacts and detailed analysis

### 3. `Research_paper.ipynb`
- **Purpose**: Interactive Jupyter notebook for step-by-step analysis
- **Usage**: For detailed exploration and customization
- **Features**: Cell-by-cell execution with intermediate results

## 📈 Model Performance Results

Based on the latest run with 25,000 samples:

### E-commerce Product Classification
- **Logistic Regression**: 59.5% accuracy
- **Voting Ensemble**: 59.4% accuracy  
- **LightGBM**: 59.0% accuracy
- **Random Forest**: 57.9% accuracy
- **XGBoost**: 58.4% accuracy
- **CatBoost**: 53.4% accuracy

### Survey Analysis
- **Logistic Regression**: 91.8% accuracy
- **Random Forest**: 83.6% accuracy

## 🖼️ Generated Visualizations

### Individual Confusion Matrices
Each model generates a separate confusion matrix saved as high-resolution PNG:
- Clear visualization of classification performance per product category
- Individual files for easy integration into presentations/papers
- Accuracy scores displayed on each matrix

### Performance Comparison Charts
- **Accuracy Comparison**: Bar chart comparing all models
- **F1-Score Comparison**: Model performance across precision/recall
- **Performance Heatmap**: Multi-metric comparison matrix

## 📝 Research Report

Each run generates a comprehensive research report (`Gen_Z_Research_Report_[timestamp].txt`) containing:
- Research objectives and methodology
- Dataset analysis and preprocessing steps
- Model performance results and comparisons
- Key findings and insights
- Generated file listings
- Future research directions

## 🔍 Key Findings

1. **Text-based Classification**: Successfully classified e-commerce products using product descriptions
2. **Survey Analysis**: High accuracy in predicting impulse buying behavior from psychological surveys
3. **Ensemble Benefits**: Voting ensembles provide robust predictions across different models
4. **Multi-modal Potential**: Framework ready for social media data integration

## 🚀 Future Enhancements

- **Social Media Integration**: Add Twitter/social media sentiment analysis
- **Transformer Models**: Implement DistilBERT for enhanced text analysis
- **Real-time Prediction**: Deploy models for live impulse buying detection
- **Longitudinal Study**: Track Gen Z buying trends over time

## 📊 Research Impact

This project demonstrates the practical application of machine learning in consumer psychology research, providing:
- **Theoretical Contributions**: Understanding of Gen Z impulse buying patterns
- **Practical Applications**: Tools for retail and marketing forecasting
- **Methodological Advances**: Multi-modal ML approach to consumer behavior analysis

## 🤝 Contributing

To extend this research:
1. Add social media datasets to the project folder
2. The pipeline will automatically detect and integrate new data sources
3. Modify model parameters in the script for different performance trade-offs
4. Customize visualizations for specific research needs

## 📞 Contact & Citation

When citing this work, please reference:
- Research Question: ML models for Gen Z impulse buying characterization and forecasting
- Methodology: Multi-modal analysis using e-commerce, survey, and social media data
- Key Innovation: Cross-domain validation of impulse buying prediction models

---

**Last Updated**: August 2025  
**Pipeline Version**: Optimized for 25K samples  
**Status**: Ready for academic research and practical applications