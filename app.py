import streamlit as st
import pandas as pd
from data_loader import DataLoader
from data_analyser import DataAnalyser
from model_trainer import split_data, train_models
from model_evaluator import evaluate_models

# ---- Config ----
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

DATA_PATH = "Reviews.csv"

def main():
    st.title("Sentiment Analysis on Reviews")

    # 1) Build dataset (data loading)
    st.subheader("Data Loading")
    loader = DataLoader(
        text_col="Text",
        score_col="Score",
        label_source="score",  
        sample_size=None,     # Always load full dataset
    )

    with st.spinner("Loading dataset..."):
        df = loader.build_dataset(DATA_PATH)

    st.write("### Sample Data")
    st.dataframe(df.head())

    # 2) Quick EDA
    st.subheader("2️⃣ Exploratory Data Analysis (EDA)")
    analyser = DataAnalyser(text_col="Text", label_col="Sentiment", score_col="Score")

    desc, dtypes = analyser.basic_info(df)
    st.write("**Dataset Info:**")
    st.write(desc)

    st.write("**Class Balance:**")
    st.bar_chart(analyser.class_balance(df))

    st.write("**Score Distribution:**")
    analyser.score_distribution(df)
    st.pyplot()

    if "Summary" in df.columns:
        st.write("**WordCloud (Overall):**")
        analyser.wordcloud_overall(df, source_col="Summary")
        st.pyplot()

        st.write("**WordCloud by Sentiment:**")
        analyser.wordcloud_by_label(df, source_col="Summary")
        st.pyplot()

    # 3) Split
    st.subheader("3️⃣ Train/Test Split")
    X_train, X_test, y_train, y_test = split_data(df, text_col="Text", label_col="Sentiment")
    st.success(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 4) Train Models
    st.subheader("4️⃣ Model Training")
    with st.spinner("Training models..."):
        models = train_models(X_train, y_train, use_grid_search=False, cv=3)
    st.success("Training completed")

    # 5) Evaluate Models
    st.subheader("5️Model Evaluation")
    with st.spinner("Evaluating models..."):
        results = evaluate_models(models, X_test, y_test)

    st.write("### Final Results")
    st.dataframe(pd.DataFrame(results).T)

if __name__ == "__main__":
    main()
