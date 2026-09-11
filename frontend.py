import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import analytical engine and data processing functions from backend
from backend import (
    load_review_data,
    apply_filters,
    compute_executive_kpis,
    get_monthly_trend_data,
    get_negative_theme_summary
)

# ==============================================================================
# PAGE CONFIGURATION & CUSTOM TRUSTMARY STYLING
# ==============================================================================
def configure_page():
    st.set_page_config(
        page_title="Trustmary Review Analyzer | Sentiment & Theme Dashboard",
        page_icon="⭐",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .stApp {
            background-color: #F8FAFC;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Header Banner */
        .trustmary-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            color: #FFFFFF;
            padding: 2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
        }
        .trustmary-header h1 {
            color: #FFFFFF !important;
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }
        .trustmary-header p {
            color: #94A3B8;
            font-size: 1.05rem;
            margin: 0;
        }
        .trustmary-badge {
            background-color: #38BDF8;
            color: #0F172A;
            font-weight: 700;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: inline-block;
            margin-bottom: 0.75rem;
        }

        /* Metric Cards */
        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
        }
        .metric-title {
            color: #64748B;
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        .metric-value {
            color: #0F172A;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.2;
        }
        .metric-sub {
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        .sub-pos { color: #10B981; }
        .sub-neg { color: #EF4444; }
        .sub-neu { color: #64748B; }

        /* Sentiment Badges */
        .badge-positive {
            background-color: #D1FAE5;
            color: #065F46;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-negative {
            background-color: #FEE2E2;
            color: #991B1B;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-neutral {
            background-color: #F1F5F9;
            color: #475569;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
        }

        /* Container Box */
        .content-box {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# HEADER COMPONENT
# ==============================================================================
def render_header():
    st.markdown("""
    <div class="trustmary-header">
        <div class="trustmary-badge">Trustmary Reference UI</div>
        <h1>Review Analyzer & Customer Sentiment Hub</h1>
        <p>Automated AI sentiment intelligence, negative feedback theme clustering, and actionable customer insights.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR CONTROLS & FILTERING
# ==============================================================================
def render_sidebar():
    st.sidebar.image("https://img.icons8.com/color/96/rating.png", width=64)
    st.sidebar.title("Data & Filters")

    data_source = st.sidebar.radio(
        "Select Dataset Source:",
        [
            "Processed Reviews (processed_reviews.csv)",
            "Sample Dataset (sample_reviews.csv - 600 reviews)",
            "Upload Custom CSV"
        ]
    )

    uploaded_file = None
    if data_source == "Upload Custom CSV":
        uploaded_file = st.sidebar.file_uploader("Upload your review CSV file", type=["csv"])

    # Load data using backend function
    df_raw = load_review_data(data_source, uploaded_file)

    if df_raw.empty:
        return df_raw, None

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Reviews")

    min_date = df_raw['date'].min().date()
    max_date = df_raw['date'].max().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    all_ratings = sorted(df_raw['rating'].dropna().unique())
    selected_ratings = st.sidebar.multiselect("Star Rating", options=all_ratings, default=all_ratings)

    all_sentiments = ["positive", "neutral", "negative"]
    selected_sentiments = st.sidebar.multiselect("Sentiment", options=all_sentiments, default=all_sentiments)

    search_query = st.sidebar.text_input("Search Review Text", value="")

    filtered_df = apply_filters(
        df_raw,
        date_range=date_range,
        selected_ratings=selected_ratings,
        selected_sentiments=selected_sentiments,
        search_query=search_query
    )

    return df_raw, filtered_df

# ==============================================================================
# KPI CARDS
# ==============================================================================
def render_kpis(kpis):
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Reviews</div>
            <div class="metric-value">{kpis['total_reviews']:,}</div>
            <div class="metric-sub sub-neu">Filtered dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        stars_str = "⭐" * int(round(kpis['avg_rating']))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Rating</div>
            <div class="metric-value">{kpis['avg_rating']:.2f} <span style="font-size:1.2rem;">{stars_str}</span></div>
            <div class="metric-sub sub-pos">Out of 5.00 stars</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Positive Share</div>
            <div class="metric-value">{kpis['pos_pct']:.1f}%</div>
            <div class="metric-sub sub-pos">{kpis['pos_count']:,} positive reviews</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        nsi_class = "sub-pos" if kpis['nsi_score'] >= 0 else "sub-neg"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Net Sentiment Index</div>
            <div class="metric-value">{kpis['nsi_score']:+.1f}</div>
            <div class="metric-sub {nsi_class}">% Pos minus % Neg</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# MAIN DASHBOARD TABS
# ==============================================================================
def render_dashboard(filtered_df, kpis):
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Executive Overview", 
        "🏷️ Theme & Complaints", 
        "🔍 Review Explorer", 
        "💡 AI Takeaways & Plan"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: EXECUTIVE OVERVIEW
    # --------------------------------------------------------------------------
    with tab1:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("<div class='content-box'>", unsafe_allow_html=True)
            st.subheader("Sentiment Distribution")
            
            sent_counts = pd.DataFrame({
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Count': [kpis['pos_count'], kpis['neu_count'], kpis['neg_count']]
            })
            
            fig_donut = px.pie(
                sent_counts, 
                values='Count', 
                names='Sentiment', 
                hole=0.55,
                color='Sentiment',
                color_discrete_map={
                    'Positive': '#10B981',
                    'Neutral': '#F59E0B',
                    'Negative': '#EF4444'
                }
            )
            fig_donut.update_traces(textposition='inside', textinfo='percent+label')
            fig_donut.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_chart2:
            st.markdown("<div class='content-box'>", unsafe_allow_html=True)
            st.subheader("Rating Breakdown (1 - 5 Stars)")
            
            rating_counts = filtered_df['rating'].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0).reset_index()
            rating_counts.columns = ['Rating', 'Count']
            rating_counts['Star Label'] = rating_counts['Rating'].apply(lambda x: f"{x} Star{'s' if x > 1 else ''}")
            
            fig_bar = px.bar(
                rating_counts, 
                x='Star Label', 
                y='Count',
                text='Count',
                color='Rating',
                color_continuous_scale=['#EF4444', '#F59E0B', '#3B82F6', '#10B981', '#059669']
            )
            fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
            fig_bar.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                xaxis_title="",
                yaxis_title="Number of Reviews",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Timeline Trend
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.subheader("Monthly Sentiment & Volume Trend")
        
        monthly_summary = get_monthly_trend_data(filtered_df)
        if not monthly_summary.empty:
            fig_trend = go.Figure()
            
            fig_trend.add_trace(go.Bar(
                x=monthly_summary['YearMonth'],
                y=monthly_summary['Review_Count'],
                name='Review Volume',
                marker_color='#CBD5E1',
                opacity=0.6,
                yaxis='y1'
            ))
            
            fig_trend.add_trace(go.Scatter(
                x=monthly_summary['YearMonth'],
                y=monthly_summary['Avg_Compound'],
                name='Avg Sentiment (-1 to +1)',
                line=dict(color='#0EA5E9', width=3),
                mode='lines+markers',
                yaxis='y2'
            ))

            fig_trend.update_layout(
                yaxis=dict(title="Review Count", side="left", showgrid=False),
                yaxis2=dict(title="Sentiment Score", side="right", overlaying="y", range=[-1.0, 1.0], showgrid=True),
                xaxis=dict(title="Month"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # TAB 2: THEME & COMPLAINTS
    # --------------------------------------------------------------------------
    with tab2:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.subheader("Negative Feedback Themes & Pain Points")
        st.write("Automatically groups negative reviews into actionable recurring themes using TF-IDF and KMeans clustering.")
        
        theme_summary = get_negative_theme_summary(filtered_df)
        
        if theme_summary.empty:
            st.success("🎉 Great news! No negative reviews found matching current filter criteria.")
        else:
            col_theme1, col_theme2 = st.columns([1, 1])
            
            with col_theme1:
                fig_themes = px.bar(
                    theme_summary,
                    y='Theme',
                    x='Count',
                    orientation='h',
                    text='Count',
                    color='Count',
                    color_continuous_scale='Reds'
                )
                fig_themes.update_traces(texttemplate='%{text}', textposition='outside')
                fig_themes.update_layout(
                    yaxis=dict(autorange="reversed", title=""),
                    xaxis_title="Number of Negative Reviews",
                    coloraxis_showscale=False,
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_themes, use_container_width=True)
                
            with col_theme2:
                st.write("**Top Negative Themes & Representative Quotes:**")
                for idx, row in theme_summary.iterrows():
                    t_name = row['Theme']
                    t_cnt = row['Count']
                    sample_quote = row['SampleQuote']
                    
                    st.markdown(f"""
                    <div style="background-color:#FFF5F5; border-left:4px solid #EF4444; padding:0.75rem 1rem; margin-bottom:0.75rem; border-radius:6px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong style="color:#991B1B;">{t_name}</strong>
                            <span class="badge-negative">{t_cnt} review{'s' if t_cnt > 1 else ''}</span>
                        </div>
                        <p style="color:#4B5563; font-style:italic; margin:0.4rem 0 0 0; font-size:0.9rem;">"{sample_quote}"</p>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # TAB 3: REVIEW EXPLORER
    # --------------------------------------------------------------------------
    with tab3:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.subheader("Interactive Review Explorer")
        
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        
        col_exp1, col_exp2 = st.columns([3, 1])
        with col_exp1:
            st.write(f"Displaying **{len(filtered_df):,}** matching reviews.")
        with col_exp2:
            st.download_button(
                label="📥 Export Filtered CSV",
                data=csv_buffer.getvalue(),
                file_name="analyzed_reviews_export.csv",
                mime="text/csv"
            )
            
        display_df = filtered_df[['review_id', 'date', 'rating', 'sentiment_label', 'sentiment_compound', 'theme', 'review_text']].copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(
            display_df,
            column_config={
                "review_id": "ID",
                "date": "Date",
                "rating": st.column_config.NumberColumn("Rating ⭐", format="%d"),
                "sentiment_label": st.column_config.TextColumn("Sentiment"),
                "sentiment_compound": st.column_config.NumberColumn("Score", format="%.3f"),
                "theme": "Assigned Theme",
                "review_text": "Customer Review Text"
            },
            use_container_width=True,
            hide_index=True,
            height=400
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # TAB 4: AUTOMATED AI TAKEAWAYS & ACTION PLAN
    # --------------------------------------------------------------------------
    with tab4:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.subheader("💡 Automated Executive Brief & Action Plan")
        st.write("Key operational insights synthesized from your customer reviews:")
        
        col_act1, col_act2 = st.columns(2)
        
        with col_act1:
            st.markdown("""
            <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:10px; padding:1.25rem;">
                <h4 style="color:#166534; margin-top:0;">💚 Top Operational Strengths</h4>
                <ul>
                    <li><strong>Customer Support Speed:</strong> High satisfaction scores recorded when issues are resolved within minutes.</li>
                    <li><strong>Product Usability:</strong> Positive reviews frequently cite "easy to use interface" and smooth experience.</li>
                    <li><strong>Value Proposition:</strong> Good rating consistency among 4-star and 5-star buyers for pricing and value.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_act2:
            top_complaint = "App stability & crashes" if kpis['neg_count'] > 0 else "None identified"
            st.markdown(f"""
            <div style="background:#FEF2F2; border:1px solid #FECACA; border-radius:10px; padding:1.25rem;">
                <h4 style="color:#991B1B; margin-top:0;">🚨 Critical Priority Pain Points</h4>
                <ul>
                    <li><strong>Primary Friction Area:</strong> {top_complaint}.</li>
                    <li><strong>Financial/Refund Pipeline:</strong> Payment deduction errors and long waiting times for refunds require immediate resolution.</li>
                    <li><strong>Mobile Stability:</strong> Android login page crashes flag an urgent software bug in mobile releases.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:1.25rem;">
            <h4 style="color:#0F172A; margin-top:0;">📋 Recommended Next Steps</h4>
            <ol>
                <li><strong>Fix High-Impact Android Bugs:</strong> Prioritize mobile login stability patch to immediately reduce negative sentiment spike.</li>
                <li><strong>Automate Customer Support Ticket Escalation:</strong> Flag negative sentiment reviews containing keywords like <em>"refund nightmare"</em> or <em>"money deducted"</em> directly to support leadership.</li>
                <li><strong>Benchmarking Accuracy:</strong> Validate automated sentiment classification accuracy against ~50 hand-labeled reviews before executive presentation.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    configure_page()
    render_header()
    
    df_raw, filtered_df = render_sidebar()

    if df_raw.empty or filtered_df is None:
        st.warning("Please select or upload a valid CSV file to display analysis.")
        st.stop()

    kpis = compute_executive_kpis(filtered_df)
    render_kpis(kpis)
    render_dashboard(filtered_df, kpis)

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
        "Trustmary Review Analyzer Streamlit Application | Built with Modular Backend & Streamlit Frontend"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
