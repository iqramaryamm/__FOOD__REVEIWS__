# main.py
from data_loader import DataLoader
from data_analyser import DataAnalyser
from model_trainer import split_data, train_models
from model_evaluator import evaluate_models

DATA_PATH = "Reviews.csv"

# 1) Build dataset (data loading)
loader = DataLoader(
    text_col="Text",
    score_col="Score",
    label_source="score",  
    sample_size=None,      
)
df = loader.build_dataset(DATA_PATH)

# 2) Quick EDA
analyser = DataAnalyser(text_col="Text", label_col="Sentiment", score_col="Score")
desc, dtypes = analyser.basic_info(df)
print(desc.head())
print("\nClass balance:\n", analyser.class_balance(df))
analyser.score_distribution(df)
analyser.wordcloud_overall(df, source_col="Summary")   # if Summary exists
analyser.wordcloud_by_label(df, source_col="Summary")  # if Summary exists

# 3) Split
X_train, X_test, y_train, y_test = split_data(df, text_col="Text", label_col="Sentiment")

# 4) Train
models = train_models(X_train, y_train, use_grid_search=False, cv=3)

# 5) Evaluate
results = evaluate_models(models, X_test, y_test)
print("\nFinal results:\n", results)
