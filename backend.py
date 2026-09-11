import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Key word lexicons for rule-based sentiment calculation
POS_KEYWORDS = set(['love', 'great', 'excellent', 'perfect', 'good', 'happy', 'smooth', 
                    'recommend', 'fast', 'exceeded', 'quick', 'easy', 'resolved', 'best', 'wonderful'])
NEG_KEYWORDS = set(['crash', 'crashing', 'freeze', 'freezes', 'terrible', 'bad', 'nightmare', 
                    'broken', 'drain', 'drains', 'late', 'deducted', 'damaged', 'frustrating', 
                    'slow', 'poor', 'failed', 'issue', 'worst', 'complaint'])

def compute_rule_based_sentiment(row):
    """Computes sentiment compound score (-1.0 to +1.0) and label if missing in raw data."""
    text = str(row.get('review_text', '')).lower()
    rating = float(row.get('rating', 3)) if pd.notnull(row.get('rating')) else 3.0
    
    p_count = sum(1 for w in POS_KEYWORDS if w in text)
    n_count = sum(1 for w in NEG_KEYWORDS if w in text)
    
    # Rating baseline score (-1.0 to 1.0)
    rating_score = (rating - 3.0) / 2.0
    text_score = (p_count - n_count) * 0.35
    
    compound = max(-1.0, min(1.0, rating_score + text_score))
    
    if compound >= 0.1:
        label = 'positive'
    elif compound <= -0.1:
        label = 'negative'
    else:
        label = 'neutral'
        
    return compound, label

def extract_negative_themes(df, n_clusters=5):
    """Extracts thematic clusters from negative reviews using TF-IDF and KMeans clustering."""
    neg_mask = df['sentiment_label'] == 'negative'
    if neg_mask.sum() < 3:
        df['theme'] = df['sentiment_label'].apply(lambda x: 'General Feedback' if x == 'negative' else 'n/a (not negative)')
        return df

    neg_indices = df[neg_mask].index
    neg_texts = df.loc[neg_indices, 'review_text'].fillna('').astype(str)

    try:
        n_clusters = min(n_clusters, len(neg_texts))
        vectorizer = TfidfVectorizer(stop_words='english', max_features=150)
        tfidf_mat = vectorizer.fit_transform(neg_texts)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_mat)
        
        feature_names = vectorizer.get_feature_names_out()
        cluster_theme_map = {}
        for i in range(n_clusters):
            center = kmeans.cluster_centers_[i]
            top_words_idx = center.argsort()[-3:][::-1]
            words = [feature_names[idx] for idx in top_words_idx if center[idx] > 0]
            if not words:
                words = [f"Theme #{i+1}"]
            theme_name = " / ".join(words)
            cluster_theme_map[i] = theme_name
            
        df['theme'] = 'n/a (not negative)'
        df.loc[neg_indices, 'theme'] = [cluster_theme_map[c] for c in cluster_labels]
    except Exception as e:
        df['theme'] = df['sentiment_label'].apply(lambda x: 'Negative Feedback' if x == 'negative' else 'n/a (not negative)')
        
    return df

def load_review_data(data_option, uploaded_file=None):
    """Loads and standardizes review dataset from local CSVs or custom upload."""
    if data_option == "Processed Reviews (processed_reviews.csv)":
        path = "/workspace/knowledge/processed_reviews.csv"
        if not os.path.exists(path):
            path = "processed_reviews.csv"
        df = pd.read_csv(path)
    elif data_option == "Sample Dataset (sample_reviews.csv - 600 reviews)":
        path = "/workspace/knowledge/sample_reviews.csv"
        if not os.path.exists(path):
            path = "sample_reviews.csv"
        df = pd.read_csv(path)
    elif data_option == "Upload Custom CSV" and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        return pd.DataFrame(columns=['review_id', 'date', 'rating', 'review_text', 'sentiment_compound', 'sentiment_label', 'theme'])

    # Standardize column names
    col_map = {c.lower(): c for c in df.columns}
    
    if 'review_text' not in df.columns:
        text_cols = [c for c in df.columns if 'text' in c.lower() or 'review' in c.lower() or 'comment' in c.lower()]
        if text_cols:
            df.rename(columns={text_cols[0]: 'review_text'}, inplace=True)
            
    if 'date' not in df.columns:
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if date_cols:
            df.rename(columns={date_cols[0]: 'date'}, inplace=True)
        else:
            df['date'] = '2025-01-01'

    if 'rating' not in df.columns:
        rating_cols = [c for c in df.columns if 'rating' in c.lower() or 'score' in c.lower() or 'star' in c.lower()]
        if rating_cols:
            df.rename(columns={rating_cols[0]: 'rating'}, inplace=True)
        else:
            df['rating'] = 3

    if 'review_id' not in df.columns:
        df['review_id'] = range(1, len(df) + 1)

    # Convert date
    df['date'] = pd.to_datetime(df['date'], errors='coerce').fillna(pd.to_datetime('2025-01-01'))

    # Compute sentiment if missing
    if 'sentiment_compound' not in df.columns or 'sentiment_label' not in df.columns:
        results = df.apply(compute_rule_based_sentiment, axis=1)
        df['sentiment_compound'] = [r[0] for r in results]
        df['sentiment_label'] = [r[1] for r in results]

    # Compute themes if missing
    if 'theme' not in df.columns or df['theme'].isnull().all():
        df = extract_negative_themes(df)

    return df

def apply_filters(df, date_range=None, selected_ratings=None, selected_sentiments=None, search_query=""):
    """Filters review dataframe according to user sidebar constraints."""
    filtered_df = df.copy()

    if date_range and len(date_range) == 2:
        start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = filtered_df[(filtered_df['date'] >= start_d) & (filtered_df['date'] <= end_d)]

    if selected_ratings:
        filtered_df = filtered_df[filtered_df['rating'].isin(selected_ratings)]

    if selected_sentiments:
        filtered_df = filtered_df[filtered_df['sentiment_label'].isin(selected_sentiments)]

    if search_query:
        filtered_df = filtered_df[filtered_df['review_text'].astype(str).str.contains(search_query, case=False, na=False)]

    return filtered_df

def compute_executive_kpis(df):
    """Calculates top-line KPI metrics for executive overview cards."""
    total_reviews = len(df)
    avg_rating = float(df['rating'].mean()) if total_reviews > 0 else 0.0

    pos_count = int((df['sentiment_label'] == 'positive').sum())
    neu_count = int((df['sentiment_label'] == 'neutral').sum())
    neg_count = int((df['sentiment_label'] == 'negative').sum())

    pos_pct = (pos_count / total_reviews * 100.0) if total_reviews > 0 else 0.0
    neg_pct = (neg_count / total_reviews * 100.0) if total_reviews > 0 else 0.0
    nsi_score = pos_pct - neg_pct  # Net Sentiment Index (-100 to +100)

    return {
        'total_reviews': total_reviews,
        'avg_rating': avg_rating,
        'pos_count': pos_count,
        'neu_count': neu_count,
        'neg_count': neg_count,
        'pos_pct': pos_pct,
        'neg_pct': neg_pct,
        'nsi_score': nsi_score
    }

def get_monthly_trend_data(df):
    """Aggregates review volume and sentiment compound score by month."""
    if df.empty:
        return pd.DataFrame(columns=['YearMonth', 'Avg_Compound', 'Avg_Rating', 'Review_Count'])

    trend_df = df.copy()
    trend_df['YearMonth'] = trend_df['date'].dt.to_period('M').dt.to_timestamp()

    monthly_summary = trend_df.groupby('YearMonth').agg(
        Avg_Compound=('sentiment_compound', 'mean'),
        Avg_Rating=('rating', 'mean'),
        Review_Count=('review_id', 'count')
    ).reset_index()

    return monthly_summary

def get_negative_theme_summary(df):
    """Returns aggregated negative feedback theme counts and representative quotes."""
    neg_reviews = df[df['sentiment_label'] == 'negative']
    if neg_reviews.empty:
        return pd.DataFrame(columns=['Theme', 'Count', 'SampleQuote'])

    theme_counts = neg_reviews['theme'].value_counts().reset_index()
    theme_counts.columns = ['Theme', 'Count']

    sample_quotes = []
    for t_name in theme_counts['Theme']:
        quote = neg_reviews[neg_reviews['theme'] == t_name]['review_text'].iloc[0]
        sample_quotes.append(quote)

    theme_counts['SampleQuote'] = sample_quotes
    return theme_counts
