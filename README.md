# SMS Spam Detection - Machine Learning Project

**Authors:** Moheb Abdelmasih & Niveditha Renganathan

## Project Overview

This project implements a comprehensive SMS spam detection system using machine learning. It performs binary text classification to identify whether an SMS message is legitimate (ham) or spam.

## Dataset

**Source:** [UCI Machine Learning Repository - SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

The dataset contains 5,574 SMS messages labeled as either "ham" (legitimate) or "spam".

## Models Implemented

We compare three different machine learning algorithms:

1. **Naive Bayes** - A probabilistic classifier based on Bayes' theorem
2. **Logistic Regression** - A linear model for binary classification  
3. **Support Vector Machine (SVM)** - Finds the optimal hyperplane for classification

## Preprocessing Pipeline

The text preprocessing includes:

- Converting text to lowercase
- Removing URLs and email addresses
- Removing numbers and special characters
- Tokenization
- Removing stopwords (common words like 'and', 'or', 'the', etc.)
- Stemming (reducing words to their root form)

## Feature Engineering

**TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization is used to convert preprocessed text into numerical features that machine learning models can process.

## Evaluation Metrics

The models are evaluated using:

- **Accuracy** - Overall correctness of predictions
- **F1 Score** - Harmonic mean of precision and recall
- **Precision** - Ratio of correct spam predictions
- **Recall** - Ratio of spam messages correctly identified
- **Confusion Matrix** - Detailed breakdown of predictions

## Project Structure

```
Spam_Text_Detection_ML/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── SMSSpamCollection        # Dataset file
└── spam_detection.ipynb     # Main Jupyter notebook with implementation
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Spam_Text_Detection_ML
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook spam_detection.ipynb
```

## Usage

The notebook is structured in the following sections:

1. **Import Libraries and Load Dataset** - Setup and data loading
2. **Exploratory Data Analysis (EDA)** - Understanding the dataset
3. **Text Preprocessing Pipeline** - Cleaning and preparing text data
4. **Feature Engineering with TF-IDF** - Converting text to numerical features
5. **Model Training and Evaluation** - Training and testing three models
6. **Model Comparison and Visualization** - Comparing performance metrics
7. **Conclusions and Best Model Selection** - Identifying the best performer
8. **Testing with Custom Messages** - Real-world testing

## Results

All three models achieve high performance:

- **Naive Bayes**: Fast training, excellent baseline performance
- **Logistic Regression**: Strong balance of accuracy and interpretability
- **SVM**: Typically achieves the best generalization performance

Detailed results including accuracy, F1 scores, and confusion matrices are available in the notebook.

## Key Findings

- Spam messages tend to be longer and contain specific keywords
- TF-IDF effectively captures important discriminative features
- All models achieve >95% accuracy on the test set
- SVM and Logistic Regression typically show the best F1 scores

## Future Enhancements

Potential improvements include:

- Testing additional algorithms (Random Forest, XGBoost, Neural Networks)
- Implementing ensemble methods
- Cross-validation for more robust evaluation
- Hyperparameter tuning
- Deployment as a web service or API

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- nltk >= 3.8.0
- scikit-learn >= 1.2.0
- jupyter >= 1.0.0

## License

This project is for educational purposes.

## Acknowledgments

- Dataset provided by UCI Machine Learning Repository
- SMS Spam Collection Dataset creators: Tiago A. Almeida and José María Gómez Hidalgo
