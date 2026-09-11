import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ------------------------------------------------------------------------------
# LEXICONS & KEYWORD MAPPINGS
# ------------------------------------------------------------------------------
POS_KEYWORDS = set(['love', 'great', 'excellent', 'perfect', 'good', 'happy', 'smooth', 
                    'recommend', 'fast', 'exceeded', 'quick', 'easy', 'resolved', 'best', 'wonderful',
                    'helpful', 'amazing', 'superb', 'awesome', 'reliable'])

NEG_KEYWORDS = set(['crash', 'crashing', 'freeze', 'freezes', 'terrible', 'bad', 'nightmare', 
                    'broken', 'drain', 'drains', 'late', 'deducted', 'damaged', 'frustrating', 
                    'slow', 'poor', 'failed', 'issue', 'worst', 'complaint', 'horrible', 'delay',
                    'unresponsive', 'error', 'scam', 'waste'])

CHANNELS = ['Google Reviews', 'App Store', 'Trustpilot', 'Customer Support Emails']
CATEGORIES = ['Mobile App', 'Billing & Refunds', 'Delivery & Packaging', 'Customer Support', 'Product Quality']


def compute_rule_based_sentiment(row):
    """Computes compound sentiment score (-1.0 to +1.0) and sentiment label."""
    text = str(row.get('review_text', '')).lower()
    rating = float(row.get('rating', 3)) if pd.notnull(row.get('rating')) else 3.0
    
    p_count = sum(1 for w in POS_KEYWORDS if w in text)
    n_count = sum(1 for w in NEG_KEYWORDS if w in text)
    
    # Rating baseline (-1.0 to 1.0)
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


def analyze_single_review(review_text, star_rating=3):
    """Real-time sentiment analyzer & auto-reply generator for business owners."""
    text_lower = review_text.lower()
    
    p_matches = [w for w in POS_KEYWORDS if w in text_lower]
    n_matches = [w for w in NEG_KEYWORDS if w in text_lower]
    
    rating_score = (star_rating - 3.0) / 2.0
    text_score = (len(p_matches) - len(n_matches)) * 0.35
    compound = max(-1.0, min(1.0, rating_score + text_score))
    
    if compound >= 0.1:
        sentiment = 'positive'
        urgency = 'Low'
        suggested_reply = (
            f"Dear Customer, thank you so much for the glowing {star_rating}-star review! "
            f"We are thrilled to hear that you had a great experience with our business. "
            f"We look forward to serving you again soon!"
        )
    elif compound <= -0.1:
        sentiment = 'negative'
        urgency = 'Critical (Urgent Fire)' if star_rating == 1 or len(n_matches) >= 2 else 'High'
        suggested_reply = (
            f"Dear Valued Customer, we are deeply sorry to hear about your experience regarding "
            f"'{review_text[:60]}...'. This does not meet our quality standards. "
            f"Please contact our management team directly at support@ourbusiness.com so we can make this right immediately."
        )
    else:
        sentiment = 'neutral'
        urgency = 'Medium'
        suggested_reply = (
            f"Hello, thank you for sharing your honest feedback. We are constantly striving to improve "
            f"and have noted your feedback. Please let us know if there is anything specific we can do to make your experience a 5-star one!"
        )
        
    return {
        'review_text': review_text,
        'star_rating': star_rating,
        'sentiment': sentiment,
        'compound_score': compound,
        'urgency': urgency,
        'pos_keywords': p_matches,
        'neg_keywords': n_matches,
        'suggested_reply': suggested_reply
    }


def assign_channel_and_category(df):
    """Categorizes reviews into channels and product areas for business owner channel benchmarking."""
    np.random.seed(42)
    
    channels = []
    categories = []
    
    for idx, row in df.iterrows():
        text = str(row.get('review_text', '')).lower()
        r_id = int(row.get('review_id', idx))
        
        # Determine channel based on ID hash for consistency
        channel = CHANNELS[r_id % len(CHANNELS)]
        channels.append(channel)
        
        # Determine category based on text keywords
        if any(w in text for w in ['refund', 'money', 'deducted', 'payment', 'failed']):
            cat = 'Billing & Refunds'
        elif any(w in text for w in ['crash', 'login', 'android', 'slow', 'freeze', 'app', 'update']):
            cat = 'Mobile App'
        elif any(w in text for w in ['support', 'service', 'emails', 'reply']):
            cat = 'Customer Support'
        elif any(w in text for w in ['delivery', 'late', 'packaging', 'arrived', 'broken', 'damaged']):
            cat = 'Delivery & Packaging'
        else:
            cat = 'Product Quality'
            
        categories.append(cat)
        
    df['channel'] = channels
    df['category'] = categories
    return df


def extract_negative_themes(df, n_clusters=5):
    """Extracts negative complaint themes using TF-IDF and KMeans clustering."""
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


def extract_positive_drivers(df):
    """Extracts top positive value drivers that customers praise most."""
    pos_df = df[df['sentiment_label'] == 'positive']
    if pos_df.empty:
        return pd.DataFrame(columns=['Driver', 'Count', 'SampleQuote'])

    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100, ngram_range=(1, 2))
        tfidf_mat = vectorizer.fit_transform(pos_df['review_text'].fillna(''))
        sums = tfidf_mat.sum(axis=0)

        keywords = [(word, sums[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        keywords = sorted(keywords, key=lambda x: x[1], reverse=True)[:5]

        drivers = []
        for word, score in keywords:
            sample_match = pos_df[pos_df['review_text'].str.contains(word, case=False, na=False)]
            sample_quote = sample_match['review_text'].iloc[0] if not sample_match.empty else "Great customer experience"
            cnt = len(sample_match)
            drivers.append({
                'Driver': word.capitalize(),
                'Count': cnt,
                'SampleQuote': sample_quote
            })

        return pd.DataFrame(drivers)
    except Exception:
        return pd.DataFrame(columns=['Driver', 'Count', 'SampleQuote'])


def load_review_data(data_option, uploaded_file=None, n_clusters=5):
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
        df = pd.DataFrame(columns=['review_id', 'date', 'rating', 'review_text', 'sentiment_compound', 'sentiment_label', 'theme', 'channel', 'category'])
        return df

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

    # Assign channel & category
    if 'channel' not in df.columns or 'category' not in df.columns:
        df = assign_channel_and_category(df)

    # Compute themes if missing or custom cluster count requested
    df = extract_negative_themes(df, n_clusters=n_clusters)

    return df


def apply_filters(df, date_range=None, selected_ratings=None, selected_sentiments=None, selected_channels=None, selected_categories=None, search_query=""):
    """Filters review dataframe according to user sidebar constraints."""
    filtered_df = df.copy()

    if date_range and len(date_range) == 2:
        start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = filtered_df[(filtered_df['date'] >= start_d) & (filtered_df['date'] <= end_d)]

    if selected_ratings:
        filtered_df = filtered_df[filtered_df['rating'].isin(selected_ratings)]

    if selected_sentiments:
        filtered_df = filtered_df[filtered_df['sentiment_label'].isin(selected_sentiments)]

    if selected_channels and 'channel' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['channel'].isin(selected_channels)]

    if selected_categories and 'category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]

    if search_query:
        filtered_df = filtered_df[filtered_df['review_text'].astype(str).str.contains(search_query, case=False, na=False)]

    return filtered_df


def compute_business_kpis(df, avg_customer_ltv=120.0):
    """Calculates top-line executive KPIs including Revenue at Risk for Business Owners."""
    total_reviews = len(df)
    avg_rating = float(df['rating'].mean()) if total_reviews > 0 else 0.0

    pos_count = int((df['sentiment_label'] == 'positive').sum())
    neu_count = int((df['sentiment_label'] == 'neutral').sum())
    neg_count = int((df['sentiment_label'] == 'negative').sum())

    pos_pct = (pos_count / total_reviews * 100.0) if total_reviews > 0 else 0.0
    neg_pct = (neg_count / total_reviews * 100.0) if total_reviews > 0 else 0.0
    nsi_score = pos_pct - neg_pct  # Net Sentiment Index (-100 to +100)

    # Business impact estimation:
    # 1-star / 2-star negative reviews carry estimated churn risk
    critical_negatives = len(df[(df['sentiment_label'] == 'negative') & (df['rating'] <= 2)])
    est_revenue_at_risk = critical_negatives * avg_customer_ltv
    est_churn_rate = (neg_count / total_reviews * 100.0 * 0.4) if total_reviews > 0 else 0.0  # 40% of negative reviews churn

    return {
        'total_reviews': total_reviews,
        'avg_rating': avg_rating,
        'pos_count': pos_count,
        'neu_count': neu_count,
        'neg_count': neg_count,
        'pos_pct': pos_pct,
        'neg_pct': neg_pct,
        'nsi_score': nsi_score,
        'critical_negatives': critical_negatives,
        'est_revenue_at_risk': est_revenue_at_risk,
        'est_churn_rate': est_churn_rate
    }


def get_monthly_trend_data(df):
    """Aggregates review volume, rating, and sentiment compound score by month."""
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
    """Returns aggregated negative feedback theme counts, category breakdown, and representative quotes."""
    neg_reviews = df[df['sentiment_label'] == 'negative']
    if neg_reviews.empty:
        return pd.DataFrame(columns=['Theme', 'Count', 'Category', 'SampleQuote'])

    theme_counts = neg_reviews['theme'].value_counts().reset_index()
    theme_counts.columns = ['Theme', 'Count']

    sample_quotes = []
    cats = []
    for t_name in theme_counts['Theme']:
        matching = neg_reviews[neg_reviews['theme'] == t_name]
        quote = matching['review_text'].iloc[0] if not matching.empty else "N/A"
        cat = matching['category'].mode()[0] if ('category' in matching.columns and not matching['category'].empty) else "General"
        sample_quotes.append(quote)
        cats.append(cat)

    theme_counts['Category'] = cats
    theme_counts['SampleQuote'] = sample_quotes
    return theme_counts


def generate_business_action_plan(df):
    """Generates prioritized business action items for business owners based on current review data."""
    neg_df = df[df['sentiment_label'] == 'negative']
    total = len(df)
    
    actions = []
    
    # 1. Android / Mobile Crashes
    crash_count = len(neg_df[neg_df['review_text'].str.contains('crash|login|android|freeze', case=False, na=False)])
    if crash_count > 0:
        actions.append({
            'id': 'ACT-01',
            'priority': '🔥 P0 URGENT',
            'title': 'Deploy Android App Login & Crash Patch',
            'area': 'Mobile Engineering',
            'affected_reviews': crash_count,
            'business_impact': f"High. Fixing login crashes protects ~{crash_count} frustrated users and prevents 1-star ratings.",
            'recommended_action': "Release v2.4 hotfix for Android login activity & memory leak management."
        })
        
    # 2. Refund & Payment Deductions
    refund_count = len(neg_df[neg_df['review_text'].str.contains('refund|money|deducted|payment', case=False, na=False)])
    if refund_count > 0:
        actions.append({
            'id': 'ACT-02',
            'priority': '🔥 P0 URGENT',
            'title': 'Automate Duplicate Deduction Refunds & Payment Gateway Fallback',
            'area': 'Billing & Finance',
            'affected_reviews': refund_count,
            'business_impact': f"Critical Revenue Risk. {refund_count} complaints mention charge issues, leading to chargeback penalties.",
            'recommended_action': "Implement automated webhook refunds for failed transactions within 24 hours."
        })

    # 3. Customer Support Response Delay
    support_count = len(neg_df[neg_df['review_text'].str.contains('support|emails|reply|service', case=False, na=False)])
    if support_count > 0:
        actions.append({
            'id': 'ACT-03',
            'priority': '⚡ P1 HIGH',
            'title': 'SLA Guarantee & AI Helpdesk Escalation Route',
            'area': 'Customer Support Operations',
            'affected_reviews': support_count,
            'business_impact': f"Brand Reputation. {support_count} customers reported unresponsive email support.",
            'recommended_action': "Set up automated auto-responder and route 1-star tickets directly to support supervisors."
        })

    # 4. Delivery & Packaging
    delivery_count = len(neg_df[neg_df['review_text'].str.contains('delivery|late|damaged|packaging', case=False, na=False)])
    if delivery_count > 0:
        actions.append({
            'id': 'ACT-04',
            'priority': '📌 P2 MEDIUM',
            'title': 'Logistics Carrier Audit & Reinforced Packaging',
            'area': 'Logistics & Shipping',
            'affected_reviews': delivery_count,
            'business_impact': f"Product Quality Perception. {delivery_count} items arrived damaged or significantly delayed.",
            'recommended_action': "Audit regional shipping partner performance and upgrade transit protective packaging."
        })

    if not actions:
        actions.append({
            'id': 'ACT-00',
            'priority': '✨ P3 LOW',
            'title': 'Maintain High Standard of Customer Experience',
            'area': 'General Operations',
            'affected_reviews': 0,
            'business_impact': "Positive customer sentiment maintained across all channels.",
            'recommended_action': "Continue collecting reviews and launch customer loyalty rewards program."
        })
        
    return pd.DataFrame(actions)
