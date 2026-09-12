import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# Review grouping system for easier readability
# -----------------------------------------------------------------------------

CATEGORY_RULES: Dict[str, Dict[str, List[str]]] = {
    "Product Reliability & Bugs": {
        "keywords": [
            "bug", "bugs", "error", "errors", "crash", "crashes", "crashing", "freeze",
            "freezes", "frozen", "lag", "laggy", "slow", "slower", "stuck", "broken",
            "malfunction", "fails", "failure", "not working", "won't work", "glitch",
            "glitches", "unresponsive", "instability", "hanging"
        ],
        "priority": 1,
    },
    "Billing, Payments & Refunds": {
        "keywords": [
            "charged", "charge", "charges", "refund", "refunded", "refunds", "payment",
            "payments", "deducted", "deduction", "billing", "invoice", "money", "cash",
            "overcharged", "double charged", "card issue", "failed payment"
        ],
        "priority": 1,
    },
    "Customer Support Experience": {
        "keywords": [
            "support", "customer service", "agent", "representative", "response time",
            "help", "unhelpful", "no reply", "ignored", "rude", "waited", "phone support",
            "chat support", "ticket", "escalation"
        ],
        "priority": 2,
    },
    "Delivery, Fulfillment & Logistics": {
        "keywords": [
            "late", "delay", "delayed", "shipping", "delivery", "arrived", "missing",
            "lost", "damaged", "damaged package", "order", "tracking", "warehouse",
            "never arrived", "arrived late", "wrong item"
        ],
        "priority": 2,
    },
    "Pricing & Value for Money": {
        "keywords": [
            "price", "pricing", "expensive", "overpriced", "value", "cheap", "worth it",
            "too costly", "not worth", "rip off", "reasonable price", "costly"
        ],
        "priority": 3,
    },
    "User Experience & Ease of Use": {
        "keywords": [
            "confusing", "difficult", "hard to use", "easy", "user friendly", "interface",
            "navigation", "complicated", "unclear", "frustrating", "poor design", "clunky",
            "learning curve", "simple", "intuitive"
        ],
        "priority": 3,
    },
    "Account, Login & Access": {
        "keywords": [
            "login", "log in", "password", "account", "access", "locked out", "auth",
            "verification", "sign in", "unable to log in", "reset password", "session"
        ],
        "priority": 1,
    },
    "Security, Privacy & Trust": {
        "keywords": [
            "security", "privacy", "unsafe", "scam", "fraud", "trust", "data leak",
            "personal info", "suspicious", "concerned", "breach", "hack"
        ],
        "priority": 1,
    },
    "Mobile App & Platform Experience": {
        "keywords": [
            "android", "ios", "app", "mobile", "phone", "tablet", "website", "browser",
            "app crash", "mobile app", "desktop", "web version", "platform"
        ],
        "priority": 2,
    },
    "General Feedback": {
        "keywords": [
            "good", "great", "love", "excellent", "happy", "satisfied", "okay", "fine",
            "average", "recommend", "smooth", "nice", "helpful"
        ],
        "priority": 4,
    },
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _keyword_hits(text: str, keywords: List[str]) -> List[str]:
    normalized = _normalize_text(text)
    found = []
    for keyword in keywords:
        pattern = re.escape(keyword.lower())
        if re.search(rf"\b{pattern}\b", normalized):
            found.append(keyword)
    return found


def _detect_sentiment(review_text: Optional[str], rating: Optional[float] = None) -> str:
    text = _normalize_text(review_text)
    if not text and rating is not None:
        if float(rating) >= 4:
            return "positive"
        if float(rating) <= 2:
            return "negative"
        return "neutral"

    positive_terms = [
        "love", "great", "excellent", "amazing", "fast", "easy", "smooth", "helpful",
        "recommend", "happy", "good", "perfect", "satisfied", "nice"
    ]
    negative_terms = [
        "bad", "terrible", "slow", "bug", "crash", "refund", "charged", "worse",
        "broken", "frustrating", "late", "issue", "problem", "hate", "awful"
    ]

    pos_score = sum(1 for term in positive_terms if term in text)
    neg_score = sum(1 for term in negative_terms if term in text)

    if rating is not None:
        rating_score = float(rating)
        if rating_score >= 4.5:
            pos_score += 2
        elif rating_score <= 2.0:
            neg_score += 2

    if pos_score > neg_score:
        return "positive"
    if neg_score > pos_score:
        return "negative"
    return "neutral"


def _severity_from_rating_and_sentiment(rating: Optional[float], sentiment: str) -> str:
    if rating is None:
        if sentiment == "negative":
            return "high"
        if sentiment == "positive":
            return "low"
        return "medium"

    rating_val = float(rating)
    if rating_val <= 2 or sentiment == "negative":
        return "high"
    if rating_val == 3 or sentiment == "neutral":
        return "medium"
    return "low"


def _priority_from_severity(severity: str) -> str:
    mapping = {
        "high": "P1",
        "medium": "P2",
        "low": "P3",
    }
    return mapping.get(severity, "P3")


# -----------------------------------------------------------------------------
# Core classification logic
# -----------------------------------------------------------------------------

def classify_review(review_text: Optional[str], rating: Optional[float] = None) -> Dict[str, object]:
    """
    Classifies a review into a major complaint section and returns the metadata
    needed to make reading and triage easier.
    """
    text = _normalize_text(review_text)
    sentiment = _detect_sentiment(text, rating)
    severity = _severity_from_rating_and_sentiment(rating, sentiment)
    priority = _priority_from_severity(severity)

    category_scores: Dict[str, int] = {}
    keyword_map: Dict[str, List[str]] = {}

    for category, config in CATEGORY_RULES.items():
        hits = _keyword_hits(text, config["keywords"])
        score = len(hits)

        # Prefer stronger weighted matches for specific operational categories
        if category == "General Feedback":
            score = 0 if hits else 1

        if score > 0:
            category_scores[category] = score
            keyword_map[category] = hits

    if not category_scores:
        major_section = "General Feedback"
        minor_category = "General Feedback"
        complaint_type = "Unclassified Feedback"
        matched_keywords = []
    else:
        major_section = max(category_scores.items(), key=lambda item: (item[1], -CATEGORY_RULES[item[0]]["priority"]))[0]
        matched_keywords = keyword_map.get(major_section, [])

        if major_section == "Product Reliability & Bugs":
            minor_category = "Platform Stability"
            complaint_type = "Bugs / Crashes / Freezes"
        elif major_section == "Billing, Payments & Refunds":
            minor_category = "Payment Integrity"
            complaint_type = "Charges / Refund / Billing"
        elif major_section == "Customer Support Experience":
            minor_category = "Support Response"
            complaint_type = "Staff / Service / Timeliness"
        elif major_section == "Delivery, Fulfillment & Logistics":
            minor_category = "Fulfillment"
            complaint_type = "Delivery / Shipping / Orders"
        elif major_section == "Pricing & Value for Money":
            minor_category = "Commercial Value"
            complaint_type = "Price / Value"
        elif major_section == "User Experience & Ease of Use":
            minor_category = "UX & Navigation"
            complaint_type = "Usability / Complexity"
        elif major_section == "Account, Login & Access":
            minor_category = "Identity & Access"
            complaint_type = "Login / Account Access"
        elif major_section == "Security, Privacy & Trust":
            minor_category = "Trust & Safety"
            complaint_type = "Security / Privacy Concern"
        elif major_section == "Mobile App & Platform Experience":
            minor_category = "App Experience"
            complaint_type = "App / Device / Platform"
        else:
            minor_category = "General Feedback"
            complaint_type = "Positive or General Comment"

    return {
        "major_section": major_section,
        "minor_category": minor_category,
        "complaint_type": complaint_type,
        "sentiment": sentiment,
        "severity": severity,
        "priority": priority,
        "keyword_hits": ", ".join(matched_keywords[:8]),
        "sort_bucket": f"{major_section}::{priority}",
    }


# -----------------------------------------------------------------------------
# DataFrame operations
# -----------------------------------------------------------------------------

def assign_review_sections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds enterprise complaint taxonomy columns to a review dataframe.
    """
    enriched = df.copy()

    if 'review_text' not in enriched.columns:
        if 'comment' in enriched.columns:
            enriched = enriched.rename(columns={'comment': 'review_text'})
        else:
            enriched['review_text'] = ''

    if 'rating' not in enriched.columns:
        enriched['rating'] = 3.0

    if 'sentiment_label' not in enriched.columns:
        enriched['sentiment_label'] = enriched.apply(
            lambda row: classify_review(row['review_text'], row['rating']).get('sentiment', 'neutral'),
            axis=1,
        )

    classification_cols = enriched.apply(
        lambda row: classify_review(row['review_text'], row['rating']),
        axis=1,
    )

    for key in [
        'major_section',
        'minor_category',
        'complaint_type',
        'sentiment',
        'severity',
        'priority',
        'keyword_hits',
        'sort_bucket',
    ]:
        enriched[key] = [item[key] for item in classification_cols]

    enriched['sentiment_label'] = enriched['sentiment']
    return enriched


def sort_reviews_for_reading(df: pd.DataFrame, category_col: str = 'major_section', priority_col: str = 'priority') -> pd.DataFrame:
    """
    Sorts reviews by enterprise priority (P1, P2, P3) and then by complaint group.
    """
    ordered = df.copy()
    ordered[priority_col] = pd.Categorical(
        ordered[priority_col],
        categories=['P1', 'P2', 'P3'],
        ordered=True,
    )
    ordered = ordered.sort_values(by=[priority_col, category_col, 'date'], ascending=[False, True, False], na_position='last')
    return ordered.reset_index(drop=True)


def group_reviews_by_section(df: pd.DataFrame, group_by: str = 'major_section') -> pd.DataFrame:
    """
    Summarises reviews by their major complaint section.
    """
    summary = (
        df.groupby(group_by, dropna=False)
        .agg(
            review_count=('review_text', 'count'),
            avg_rating=('rating', 'mean'),
            negative_reviews=('sentiment', lambda s: (s == 'negative').sum()),
            positive_reviews=('sentiment', lambda s: (s == 'positive').sum()),
            top_keywords=('keyword_hits', lambda s: ', '.join(s.dropna().astype(str).head(1))),
        )
        .reset_index()
    )
    if 'avg_rating' in summary.columns:
        summary['avg_rating'] = summary['avg_rating'].round(2)
    return summary.sort_values(['negative_reviews', 'review_count'], ascending=[False, False]).reset_index(drop=True)


def review_reading_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a clean operational ordering of grouped complaint sections.
    """
    summary = group_reviews_by_section(df)
    priority_order = {
        'P1': 1,
        'P2': 2,
        'P3': 3,
    }

    if 'priority' in df.columns:
        section_priority = (
            df.groupby('major_section')['priority']
            .agg(lambda x: priority_order.get(str(x.mode().iloc[0]), 99))
            .reset_index()
            .rename(columns={'priority': 'section_priority'})
        )
        summary = summary.merge(section_priority, on='major_section', how='left')
        summary = summary.sort_values(['section_priority', 'negative_reviews', 'review_count'], ascending=[True, False, False])
    return summary.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Example usage / demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    sample_reviews = [
        {
            'review_id': 1,
            'date': '2025-01-02',
            'rating': 1,
            'review_text': 'The app crashes every time I open my account and my refund was deducted without notice.',
        },
        {
            'review_id': 2,
            'date': '2025-01-03',
            'rating': 2,
            'review_text': 'Customer support was rude and took forever to reply about my order being delayed.',
        },
        {
            'review_id': 3,
            'date': '2025-01-04',
            'rating': 5,
            'review_text': 'Great support and the dashboard is very easy to use.',
        },
        {
            'review_id': 4,
            'date': '2025-01-05',
            'rating': 3,
            'review_text': 'The pricing feels a bit high for what you get, but the product works okay.',
        },
    ]

    df = pd.DataFrame(sample_reviews)
    categorized = assign_review_sections(df)
    sorted_df = sort_reviews_for_reading(categorized)

    print('\n==== Categorized reviews ====' )
    print(sorted_df[['review_id', 'major_section', 'minor_category', 'priority', 'severity', 'sentiment', 'review_text']].to_string(index=False))

    print('\n==== Summary by major section ====' )
    print(group_reviews_by_section(categorized).to_string(index=False))


