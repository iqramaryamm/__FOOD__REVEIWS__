# Sentiment Analysis  Pipeline Documentation

This documentation describes a modular Python pipeline for performing sentiment analysis on text data, specifically designed for processing a dataset like `Reviews.csv`. The pipeline consists of four main modules: `data_loader.py`, `data_analyser.py`, `model_trainer.py`, and `model_evaluator.py`. Each module is designed to handle a specific aspect of the sentiment analysis workflow, ensuring modularity, reusability, and maintainability.

## Overview

The pipeline performs the following tasks:
1. **Data Loading and Preprocessing** (`data_loader.py`): Loads a dataset, preprocesses text data, and creates sentiment labels.
2. **Data Analysis** (`data_analyser.py`): Provides exploratory data analysis (EDA) through descriptive statistics, class balance, score distribution, and word clouds.
3. **Model Training** (`model_trainer.py`): Trains multiple machine learning models for sentiment classification using a pipeline that includes text vectorization and classification.
4. **Model Evaluation** (`model_evaluator.py`): Evaluates trained models using metrics like accuracy, precision, recall, and F1-score, and visualizes results with confusion matrices and bar plots.

## Module Details

### 1. `data_loader.py`

**Purpose**: Loads and preprocesses a dataset (e.g., `Reviews.csv`) for sentiment analysis tasks.

**Key Features**:
- **Initialization**: Configurable parameters for text preprocessing (e.g., lowercase, remove punctuation, remove stopwords, lemmatize) and label creation (score-based or VADER-based).
- **Text Preprocessing**: Applies cleaning steps such as lowercasing, punctuation removal, stopword removal, and lemmatization using NLTK and TextBlob.
- **Label Creation**:
  - **Score-based**: Maps scores (1, 2 → Negative; 4, 5 → Positive; drops 3).
  - **VADER-based**: Uses VADER sentiment analyzer to assign Positive (compound ≥ 0) or Negative labels.
- **Sampling**: Supports optional sampling for faster experimentation.
- **NLTK Asset Management**: Automatically downloads required NLTK resources (stopwords, WordNet, OMW).

**Key Methods**:
- `load(path)`: Loads and optionally samples the dataset, keeping relevant columns.
- `preprocess_text(df)`: Applies text cleaning steps to the specified text column.
- `_labels_from_score(df)`: Creates binary sentiment labels based on score.
- `_labels_from_vader(df)`: Creates binary sentiment labels using VADER.
- `build_dataset(path)`: Combines loading, preprocessing, and label creation into a single method.

**Dependencies**: `pandas`, `numpy`, `nltk`, `textblob`, `vaderSentiment`.

### 2. `data_analyser.py`

**Purpose**: Performs exploratory data analysis (EDA) on the dataset to understand its structure and content.

**Key Features**:
- **Descriptive Statistics**: Summarizes dataset statistics and data types.
- **Class Balance**: Analyzes the distribution of sentiment labels (normalized or raw counts).
- **Score Distribution**: Visualizes the distribution of scores (if available) using a count plot.
- **Word Clouds**: Generates word clouds for overall data and by sentiment class, using the `Summary` or specified column.

**Key Methods**:
- `basic_info(df)`: Returns descriptive statistics and data types.
- `class_balance(df, normalize=True)`: Returns sentiment label distribution.
- `score_distribution(df)`: Plots score distribution using Seaborn.
- `wordcloud(text, background_color, title)`: Generates a word cloud for given text.
- `wordcloud_overall(df, source_col)`: Creates a word cloud for all summaries.
- `wordcloud_by_label(df, source_col)`: Creates word clouds for Positive and Negative summaries.

**Dependencies**: `pandas`, `matplotlib`, `seaborn`, `wordcloud`.

### 3. `model_trainer.py`

**Purpose**: Trains multiple machine learning models for sentiment classification using text data.

**Key Features**:
- **Data Splitting**: Splits data into training and test sets with optional stratification.
- **Model Zoo**: Defines a dictionary of pipelines combining TF-IDF vectorization with classifiers (Logistic Regression, LinearSVC, Multinomial Naive Bayes).
- **Hyperparameter Tuning**: Supports optional grid search for hyperparameter optimization.
- **Pipelines**: Ensures consistent text vectorization and classification steps.

**Key Methods**:
- `split_data(df, text_col, label_col, test_size, random_state, stratify)`: Splits data into train/test sets.
- `model_zoo(random_state)`: Returns a dictionary of model pipelines.
- `default_param_grids()`: Defines hyperparameter grids for grid search.
- `train_models(X_train, y_train, use_grid_search, cv, n_jobs, random_state)`: Trains models with or without grid search.

**Dependencies**: `pandas`, `sklearn` (including `model_selection`, `pipeline`, `feature_extraction.text`, `linear_model`, `svm`, `naive_bayes`).

### 4. `model_evaluator.py`

**Purpose**: Evaluates trained models and visualizes their performance.

**Key Features**:
- **Metrics**: Computes accuracy, macro-averaged precision, recall, and F1-score.
- **Visualizations**: Generates confusion matrices and a bar plot comparing model accuracies.
- **Classification Report**: Provides detailed per-class metrics (precision, recall, F1-score).

**Key Methods**:
- `evaluate_models(models, X_test, y_test)`: Evaluates models, generates visualizations, and returns a DataFrame of metrics.

**Dependencies**: `pandas`, `matplotlib`, `seaborn`, `sklearn.metrics`.

## Usage Example

```python
from data_loader import DataLoader
from data_analyser import DataAnalyser
from model_trainer import split_data, train_models
from model_evaluator import evaluate_models

# Initialize
loader = DataLoader(label_source="score", sample_size=1000)
analyser = DataAnalyser()

# Load and preprocess data
df = loader.build_dataset("Reviews.csv")

# Analyze data
desc, dtypes = analyser.basic_info(df)
print("Dataset Info:\n", desc)
print("Data Types:\n", dtypes)
analyser.class_balance(df)
analyser.score_distribution(df)
analyser.wordcloud_overall(df)
analyser.wordcloud_by_label(df)

# Split data
X_train, X_test, y_train, y_test = split_data(df)

# Train models
models = train_models(X_train, y_train, use_grid_search=True)

# Evaluate models
results = evaluate_models(models, X_test, y_test)
print("Results:\n", results)
```

## Approach and Design Choices

### Why This Approach?

1. **Modularity**:
   - The pipeline is split into four distinct modules, each responsible for a specific task (loading, analysis, training, evaluation). This separation enhances maintainability, reusability, and testability, aligning with software engineering best practices.
   - Each module is encapsulated as a class or function, making it easy to extend or modify individual components without affecting others.

2. **Flexibility**:
   - **DataLoader**: Supports multiple label creation strategies (score-based or VADER-based) and configurable text preprocessing options, allowing adaptation to different datasets or requirements.
   - **ModelTrainer**: Includes a model zoo with multiple classifiers and optional grid search, enabling experimentation with different models and hyperparameters.
   - **DataAnalyser**: Provides a range of EDA tools (statistics, distributions, word clouds) to support dataset understanding.
   - **ModelEvaluator**: Offers comprehensive evaluation metrics and visualizations for model comparison.

3. **Robustness**:
   - Handles edge cases (e.g., missing columns, non-string text, NLTK resource downloads).
   - Uses type hints and docstrings for better code clarity and IDE support.
   - Incorporates error handling (e.g., `KeyError` for missing columns, validation for `label_source`).

4. **Visualization and Interpretability**:
   - Includes visualizations like confusion matrices, score distributions, and word clouds to aid in understanding data and model performance.
   - Uses Seaborn and Matplotlib for professional, clear plots.

5. **Efficiency**:
   - Supports sampling for faster experimentation.
   - Uses TF-IDF vectorization for efficient text feature extraction.
   - Leverages `n_jobs=-1` in grid search for parallel processing.

### Why Not Other Approaches?

1. **Deep Learning Models**:
   - While deep learning (e.g., transformers like BERT) could offer higher accuracy, this code uses traditional ML models (Logistic Regression, LinearSVC) due to their simplicity, speed, and effectiveness for smaller datasets or resource-constrained environments. Deep learning requires more computational resources and data, which may not be necessary for this dataset.

2. **Custom Text Preprocessing**:
   - Instead of a custom preprocessing pipeline, the code uses established libraries (NLTK, TextBlob, VADER) for reliability and community support. Writing custom preprocessing from scratch would be error-prone and time-consuming.

3. **Alternative Vectorization**:
   - TF-IDF was chosen over other methods (e.g., word embeddings) for its simplicity, interpretability, and effectiveness in traditional ML pipelines. Embeddings require more complex models and preprocessing, which may not be justified for the task.

4. **Other Visualization Libraries**:
   - Seaborn and Matplotlib were chosen for their maturity, flexibility, and integration with Pandas. Alternatives like Plotly or Bokeh are more suited for interactive dashboards, which are not required here.

5. **Hardcoded Parameters**:
   - The code avoids hardcoding critical parameters (e.g., text column names, random state) by making them configurable via class initialization or function arguments, ensuring flexibility across datasets.

