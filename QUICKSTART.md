# Quick Start Guide

## SMS Spam Detection Project

**Authors:** Moheb Abdelmasih & Niveditha Renganathan

---

## Getting Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Setup (Optional but Recommended)

```bash
python test_setup.py
```

This will check that all packages are installed and the dataset is available.

### Step 3: Run the Notebook

```bash
jupyter notebook spam_detection.ipynb
```

Then click "Run All" in the Jupyter interface (Cell → Run All).

---

## What to Expect

The notebook will:

1. **Load and explore the data** (~5,574 SMS messages)
2. **Preprocess the text** (remove special chars, stopwords, apply stemming)
3. **Create TF-IDF features** (convert text to numbers)
4. **Train 3 models**:
   - Naive Bayes
   - Logistic Regression
   - Support Vector Machine (SVM)
5. **Compare performance** with visualizations
6. **Test on custom messages** (you can add your own!)

### Expected Runtime

- Full notebook execution: **2-5 minutes** on a standard laptop
- Dataset size: ~500KB
- Models: Fast training due to dataset size

---

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:** Install missing packages
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn
```

### Issue: NLTK Data Not Found

**Solution:** The notebook automatically downloads required NLTK data, but if it fails:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Issue: Dataset Not Found

**Solution:** Ensure `SMSSpamCollection` file is in the same directory as the notebook.

---

## Key Metrics to Watch

- **Accuracy**: Overall correctness (target: >95%)
- **F1 Score**: Balance of precision and recall (target: >0.90)
- **Confusion Matrix**: See where models make mistakes

---

## Customization Ideas

### Test Your Own Messages

At the end of the notebook, you'll find a section to test custom messages. Add your own examples:

```python
test_messages = [
    "Your custom message here",
    "Another test message",
]
```

### Adjust TF-IDF Parameters

In the feature engineering section, try different values:

```python
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,  # Try 5000 instead of 3000
    min_df=3,           # Try 3 instead of 2
    max_df=0.7          # Try 0.7 instead of 0.8
)
```

### Try Different Model Parameters

```python
# Naive Bayes with different alpha
nb_model = MultinomialNB(alpha=0.5)

# Logistic Regression with different C
lr_model = LogisticRegression(C=0.5, max_iter=1000)

# SVM with RBF kernel
svm_model = SVC(kernel='rbf', C=1.0)
```

---

## Project Structure

```
Spam_Text_Detection_ML/
├── README.md                 # Full documentation
├── QUICKSTART.md            # This file
├── requirements.txt          # Dependencies
├── test_setup.py            # Setup verification script
├── SMSSpamCollection        # Dataset
└── spam_detection.ipynb     # Main notebook
```

---

## Need Help?

1. **Check the README.md** for detailed documentation
2. **Run test_setup.py** to diagnose setup issues
3. **Review the notebook comments** for code explanations

---

## Next Steps After Completion

- ✅ Compare which model performs best for your use case
- ✅ Test with your own SMS messages
- ✅ Experiment with different preprocessing techniques
- ✅ Try hyperparameter tuning
- ✅ Consider deploying the model as an API

---

**Happy Spam Detecting! 🎯📱**

