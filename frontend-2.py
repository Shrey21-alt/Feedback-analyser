import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import backend module
import backend

# ==============================================================================
# PAGE CONFIGURATION & CUSTOM STYLING
# ==============================================================================
def configure_page():
    st.set_page_config(
        page_title="Business Owner Review Intelligence & Revenue Protection Hub",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        /* Main Background & Fonts */
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
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }
        .metric-value {
            color: #0F172A;
            font-size: 1.85rem;
            font-weight: 800;
            line-height: 1.2;
        }
        .metric-sub {
            font-size: 0.825rem;
            font-weight: 600;
            margin-top: 0.4rem;
        }
        .sub-pos { color: #10B981; }
        .sub-neg { color: #EF4444; }
        .sub-neu { color: #64748B; }

        /* Container Box */
        .content-box {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }

        /* Action Cards */
        .action-card {
            background: #FFFFFF;
            border-left: 5px solid #0EA5E9;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        }
        .action-card-p0 { border-left-color: #EF4444; background: #FEF2F2; }
        .action-card-p1 { border-left-color: #F59E0B; background: #FFFBEB; }
        .action-card-p2 { border-left-color: #3B82F6; background: #EFF6FF; }

        /* Badges */
        .badge-positive { background-color: #D1FAE5; color: #065F46; padding: 0.2rem 0.6rem; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
        .badge-negative { background-color: #FEE2E2; color: #991B1B; padding: 0.2rem 0.6rem; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
        .badge-neutral { background-color: #F1F5F9; color: #475569; padding: 0.2rem 0.6rem; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div class="trustmary-header">
        <div class="trustmary-badge">💼 Business Owner Action & Intelligence Engine</div>
        <h1>Review Intelligence & Revenue Protection Hub</h1>
        <p>As a business owner, you cannot view and analyze every customer review line-by-line. This hub turns scattered customer feedback into immediate revenue protection, automated support replies, and prioritized operational fixes.</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    configure_page()
    render_header()

    # Initialize session state for custom simulated reviews
    if 'custom_reviews' not in st.session_state:
        st.session_state['custom_reviews'] = []

    if 'action_status' not in st.session_state:
        st.session_state['action_status'] = {}

    # Sidebar Controls
    st.sidebar.image("https://img.icons8.com/color/96/rating.png", width=64)
    st.sidebar.title("💼 Business Controls & Data")

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

    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Business Assumptions")
    avg_customer_ltv = st.sidebar.slider(
        "Avg Customer LTV ($ Revenue per Customer)",
        min_value=10,
        max_value=500,
        value=120,
        step=10,
        help="Used to compute estimated Revenue at Risk from negative 1 & 2 star reviews."
    )

    n_clusters = st.sidebar.slider(
        "Complaint Clustering Granularity",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of AI complaint clusters to generate from negative reviews."
    )

    # Load base dataset
    df_raw = backend.load_review_data(data_source, uploaded_file, n_clusters=n_clusters)

    if df_raw.empty:
        st.warning("Please upload a valid CSV file or select a dataset source to begin.")
        st.stop()

    # Inject session state simulated reviews if any
    if st.session_state['custom_reviews']:
        sim_df = pd.DataFrame(st.session_state['custom_reviews'])
        df_raw = pd.concat([sim_df, df_raw], ignore_index=True)

    # Sidebar Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filter & Drill Down")

    # Date Range Filter
    min_date = df_raw['date'].min().date()
    max_date = df_raw['date'].max().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Channel filter
    all_channels = sorted(df_raw['channel'].dropna().unique()) if 'channel' in df_raw.columns else []
    selected_channels = st.sidebar.multiselect("Review Channel", options=all_channels, default=all_channels)

    # Category filter
    all_cats = sorted(df_raw['category'].dropna().unique()) if 'category' in df_raw.columns else []
    selected_cats = st.sidebar.multiselect("Product Category", options=all_cats, default=all_cats)

    # Star rating filter
    all_ratings = sorted(df_raw['rating'].dropna().unique())
    selected_ratings = st.sidebar.multiselect("Star Rating", options=all_ratings, default=all_ratings)

    # Sentiment filter
    all_sentiments = ["positive", "neutral", "negative"]
    selected_sentiments = st.sidebar.multiselect("Sentiment", options=all_sentiments, default=all_sentiments)

    # Search query
    search_query = st.sidebar.text_input("Search Review Keywords", value="")

    # Apply filters via backend
    filtered_df = backend.apply_filters(
        df_raw,
        date_range=date_range,
        selected_ratings=selected_ratings,
        selected_sentiments=selected_sentiments,
        selected_channels=selected_channels,
        selected_categories=selected_cats,
        search_query=search_query
    )

    # Calculate KPIs via backend
    kpis = backend.compute_business_kpis(filtered_df, avg_customer_ltv=avg_customer_ltv)

    # =========================================================================
    # TOP EXECUTIVE KPI METRICS CARDS
    # =========================================================================
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Revenue at Risk</div>
            <div class="metric-value" style="color:#DC2626;">${kpis['est_revenue_at_risk']:,.0f}</div>
            <div class="metric-sub sub-neg">{kpis['critical_negatives']} critical negative reviews</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        nsi_class = "sub-pos" if kpis['nsi_score'] >= 0 else "sub-neg"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Net Sentiment Index</div>
            <div class="metric-value">{kpis['nsi_score']:+.1f}</div>
            <div class="metric-sub {nsi_class}">% Pos minus % Neg</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        stars_str = "⭐" * int(round(kpis['avg_rating']))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Rating</div>
            <div class="metric-value">{kpis['avg_rating']:.2f} <span style="font-size:1rem;">{stars_str}</span></div>
            <div class="metric-sub sub-pos">Out of 5.00 stars</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Est. Customer Churn</div>
            <div class="metric-value">{kpis['est_churn_rate']:.1f}%</div>
            <div class="metric-sub sub-neg">Neg dissatisfaction rate</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Reviews</div>
            <div class="metric-value">{kpis['total_reviews']:,}</div>
            <div class="metric-sub sub-neu">{kpis['pos_count']} pos / {kpis['neg_count']} neg</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # MAIN TABBED DASHBOARD FOR BUSINESS OWNERS
    # =========================================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔥 Command Center & Channels",
        "🏷️ Pain Points & Action Tracker",
        "🧪 Live Review Simulator & AI Reply",
        "🔍 Filterable Review Explorer",
        "💡 Growth Drivers & Strategy"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: EXECUTIVE COMMAND CENTER & CHANNEL BENCHMARKING
    # -------------------------------------------------------------------------
    with tab1:
        # Critical Fire Banner
        if kpis['neg_count'] > 0:
            top_category = filtered_df[filtered_df['sentiment_label'] == 'negative']['category'].mode()
            top_cat_str = top_category.iloc[0] if not top_category.empty else "Product Quality"
            st.error(f"🚨 **Urgent Business Alert:** '{top_cat_str}' is your largest negative review driver with {kpis['neg_count']} negative reviews. Resolving these issues can recover up to **${kpis['est_revenue_at_risk']:,.0f}** in customer lifetime value.")

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("<div class='content-box'>", unsafe_allow_html=True)
            st.subheader("Review Sentiment by Channel")
            st.write("Compare customer experience across Google Reviews, App Store, Trustpilot, and Support Emails.")

            if 'channel' in filtered_df.columns:
                channel_sentiment = filtered_df.groupby(['channel', 'sentiment_label']).size().unstack(fill_value=0).reset_index()
                for c in ['positive', 'neutral', 'negative']:
                    if c not in channel_sentiment.columns:
                        channel_sentiment[c] = 0

                fig_channel = px.bar(
                    channel_sentiment,
                    x='channel',
                    y=['positive', 'neutral', 'negative'],
                    barmode='group',
                    color_discrete_map={
                        'positive': '#10B981',
                        'neutral': '#F59E0B',
                        'negative': '#EF4444'
                    },
                    labels={'value': 'Review Count', 'channel': 'Channel', 'variable': 'Sentiment'}
                )
                fig_channel.update_layout(margin=dict(t=20, b=20, l=20, r=20), legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_channel, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_c2:
            st.markdown("<div class='content-box'>", unsafe_allow_html=True)
            st.subheader("Friction Breakdown by Product Category")
            st.write("Identify which business departments require immediate management intervention.")

            if 'category' in filtered_df.columns:
                cat_sentiment = filtered_df.groupby(['category', 'sentiment_label']).size().unstack(fill_value=0).reset_index()
                for c in ['positive', 'neutral', 'negative']:
                    if c not in cat_sentiment.columns:
                        cat_sentiment[c] = 0

                fig_cat = px.bar(
                    cat_sentiment,
                    y='category',
                    x=['positive', 'neutral', 'negative'],
                    orientation='h',
                    barmode='stack',
                    color_discrete_map={
                        'positive': '#10B981',
                        'neutral': '#F59E0B',
                        'negative': '#EF4444'
                    },
                    labels={'value': 'Reviews', 'category': '', 'variable': 'Sentiment'}
                )
                fig_cat.update_layout(margin=dict(t=20, b=20, l=20, r=20), legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_cat, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Monthly Trend Chart
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.subheader("Monthly Sentiment & Volume Timeline")
        st.write("Track whether your operational improvements are lifting customer sentiment over time.")

        trend_summary = backend.get_monthly_trend_data(filtered_df)
        if not trend_summary.empty:
            fig_trend = go.Figure()

            # Bars for Review Volume
            fig_trend.add_trace(go.Bar(
                x=trend_summary['YearMonth'],
                y=trend_summary['Review_Count'],
                name='Review Volume',
                marker_color='#CBD5E1',
                opacity=0.6,
                yaxis='y1'
            ))

            # Line for Average Sentiment Compound
            fig_trend.add_trace(go.Scatter(
                x=trend_summary['YearMonth'],
                y=trend_summary['Avg_Compound'],
                name='Avg Sentiment Score (-1 to +1)',
                line=dict(color='#0EA5E9', width=3),
                mode='lines+markers',
                yaxis='y2'
            ))

            fig_trend.update_layout(
                yaxis=dict(title="Review Volume", side="left", showgrid=False),
                yaxis2=dict(title="Sentiment Score", side="right", overlaying="y", range=[-1.0, 1.0], showgrid=True),
                xaxis=dict(title="Month"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 2: COMPLAINT CLUSTERING & OPERATIONAL ACTION TRACKER
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.subheader("Negative Complaint Clustering (AI TF-IDF + KMeans)")
        st.write("Instead of reading hundreds of negative comments, these clusters group common complaints into root causes.")

        theme_summary = backend.get_negative_theme_summary(filtered_df)

        if theme_summary.empty:
            st.success("🎉 No negative reviews found matching current filter criteria!")
        else:
            col_t1, col_t2 = st.columns([1, 1])

            with col_t1:
                fig_themes = px.bar(
                    theme_summary,
                    y='Theme',
                    x='Count',
                    orientation='h',
                    text='Count',
                    color='Count',
                    color_continuous_scale='Reds',
                    labels={'Count': 'Negative Reviews', 'Theme': 'Complaint Cluster'}
                )
                fig_themes.update_traces(texttemplate='%{text}', textposition='outside')
                fig_themes.update_layout(
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False,
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_themes, use_container_width=True)

            with col_t2:
                st.write("**Complaint Cluster Drill-Down & Sample Quotes:**")
                for idx, row in theme_summary.iterrows():
                    t_name = row['Theme']
                    t_cnt = row['Count']
                    t_quote = row['SampleQuote']
                    t_cat = row.get('Category', 'General')

                    st.markdown(f"""
                    <div style="background-color:#FFF5F5; border-left:4px solid #EF4444; padding:0.75rem 1rem; margin-bottom:0.75rem; border-radius:6px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong style="color:#991B1B;">{t_name}</strong>
                            <span class="badge-negative">{t_cnt} reviews ({t_cat})</span>
                        </div>
                        <p style="color:#4B5563; font-style:italic; margin:0.4rem 0 0 0; font-size:0.875rem;">"{t_quote}"</p>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Interactive Business Action Tracker
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.subheader("📋 Business Owner Operational Action Tracker")
        st.write("Mark action items as resolved or in-progress as your engineering, support, and logistics teams fix customer issues.")

        action_df = backend.generate_business_action_plan(filtered_df)

        resolved_count = 0
        total_actions = len(action_df)

        for idx, row in action_df.iterrows():
            act_id = row['id']
            curr_status = st.session_state['action_status'].get(act_id, "To Do")

            col_a1, col_a2 = st.columns([3, 1])

            with col_a1:
                p_class = "action-card-p0" if "P0" in row['priority'] else ("action-card-p1" if "P1" in row['priority'] else "action-card-p2")
                st.markdown(f"""
                <div class="action-card {p_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong style="font-size:1.05rem;">{row['priority']} - {row['title']}</strong>
                        <span style="font-size:0.8rem; font-weight:700; background:#E2E8F0; padding:0.2rem 0.5rem; border-radius:4px;">{row['area']}</span>
                    </div>
                    <p style="margin:0.4rem 0; color:#334155; font-size:0.9rem;"><strong>Impact:</strong> {row['business_impact']}</p>
                    <p style="margin:0; color:#0F172A; font-size:0.9rem;"><strong>Recommended Action:</strong> {row['recommended_action']}</p>
                </div>
                """, unsafe_allow_html=True)

            with col_a2:
                new_status = st.selectbox(
                    f"Status for {act_id}",
                    ["To Do", "In Progress", "Resolved"],
                    index=["To Do", "In Progress", "Resolved"].index(curr_status),
                    key=f"status_select_{act_id}"
                )
                st.session_state['action_status'][act_id] = new_status
                if new_status == "Resolved":
                    resolved_count += 1

        # Resolution Progress Bar
        progress_pct = (resolved_count / total_actions) if total_actions > 0 else 1.0
        st.markdown("<br>", unsafe_allow_html=True)
        st.write(f"**Operational Fix Progress:** {resolved_count} of {total_actions} priority items resolved ({progress_pct*100:.0f}%)")
        st.progress(progress_pct)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 3: LIVE REVIEW SIMULATOR & AI RESPONSE GENERATOR
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.subheader("🧪 Interactive Customer Review Simulator & Response Generator")
        st.write("As a business owner, test how incoming customer reviews or support tickets will be analyzed, prioritized, and answered by AI.")

        col_sim1, col_sim2 = st.columns([1, 1])

        with col_sim1:
            st.markdown("#### Input New Review / Feedback")
            sim_text = st.text_area(
                "Customer Review Text:",
                value="The app crashed twice during checkout and money was deducted from my account, but support hasn't replied to my emails!",
                height=120
            )
            sim_rating = st.slider("Star Rating:", min_value=1, max_value=5, value=1)
            sim_channel = st.selectbox("Source Channel:", backend.CHANNELS)

            btn_analyze = st.button("🚀 Analyze Feedback & Auto-Generate AI Reply", use_container_width=True)

        with col_sim2:
            if btn_analyze or sim_text:
                res = backend.analyze_single_review(sim_text, star_rating=sim_rating)

                st.markdown("#### AI Analysis Results")

                st_col1, st_col2, st_col3 = st.columns(3)
                with st_col1:
                    sent_color = "#10B981" if res['sentiment'] == 'positive' else ("#EF4444" if res['sentiment'] == 'negative' else "#F59E0B")
                    st.markdown(f"**Sentiment:** <span style='color:{sent_color}; font-weight:800; font-size:1.1rem;'>{res['sentiment'].upper()}</span>", unsafe_allow_html=True)
                with st_col2:
                    st.markdown(f"**Score:** `{res['compound_score']:+.2f}`")
                with st_col3:
                    urg_color = "#DC2626" if "Critical" in res['urgency'] else "#D97706"
                    st.markdown(f"**Urgency:** <span style='color:{urg_color}; font-weight:800;'>{res['urgency']}</span>", unsafe_allow_html=True)

                if res['neg_keywords']:
                    st.write(f"**Risk Keywords Detected:** `{', '.join(res['neg_keywords'])}`")
                if res['pos_keywords']:
                    st.write(f"**Praise Keywords Detected:** `{', '.join(res['pos_keywords'])}`")

                st.markdown("#### ✉️ Suggested Professional Support Response")
                st.info(res['suggested_reply'])

                # Option to simulate adding to live dashboard
                if st.button("➕ Inject This Review into Live Dashboard Data"):
                    new_entry = {
                        'review_id': len(df_raw) + 1000,
                        'date': pd.Timestamp.now(),
                        'rating': sim_rating,
                        'review_text': sim_text,
                        'sentiment_compound': res['compound_score'],
                        'sentiment_label': res['sentiment'],
                        'channel': sim_channel,
                        'category': 'Mobile App' if 'crash' in sim_text.lower() else 'Customer Support',
                        'theme': 'Simulated Feedback'
                    }
                    st.session_state['custom_reviews'].append(new_entry)
                    st.success("✅ Review injected into active dashboard! Top metrics updated.")
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 4: FILTERABLE REVIEW EXPLORER
    # -------------------------------------------------------------------------
    with tab4:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.subheader("🔍 Filterable Review Explorer Table")
        st.write(f"Displaying **{len(filtered_df):,}** matching customer reviews. Search, inspect, and export filtered results.")

        # Export CSV Button
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)

        col_exp1, col_exp2 = st.columns([3, 1])
        with col_exp1:
            st.write("Use the sidebar filters to refine by Channel, Star Rating, or Keywords.")
        with col_exp2:
            st.download_button(
                label="📥 Export Filtered CSV",
                data=csv_buffer.getvalue(),
                file_name="business_reviews_export.csv",
                mime="text/csv",
                use_container_width=True
            )

        display_cols = ['review_id', 'date', 'channel', 'category', 'rating', 'sentiment_label', 'sentiment_compound', 'theme', 'review_text']
        available_cols = [c for c in display_cols if c in filtered_df.columns]

        display_df = filtered_df[available_cols].copy()
        if 'date' in display_df.columns:
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')

        st.dataframe(
            display_df,
            column_config={
                "review_id": "ID",
                "date": "Date",
                "channel": "Channel",
                "category": "Department",
                "rating": st.column_config.NumberColumn("Rating ⭐", format="%d"),
                "sentiment_label": "Sentiment",
                "sentiment_compound": st.column_config.NumberColumn("Score", format="%.3f"),
                "theme": "AI Theme Cluster",
                "review_text": "Full Customer Review"
            },
            use_container_width=True,
            hide_index=True,
            height=450
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 5: POSITIVE DRIVERS & REVENUE GROWTH STRATEGY
    # -------------------------------------------------------------------------
    with tab5:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.subheader("💡 What Customers Love & Revenue Growth Strategy")
        st.write("Understanding what delights customers allows business owners to double down on core strengths in marketing and sales.")

        pos_drivers = backend.extract_positive_drivers(filtered_df)

        if pos_drivers.empty:
            st.info("No positive drivers extracted from current filtered view.")
        else:
            col_p1, col_p2 = st.columns(2)

            with col_p1:
                fig_pos = px.bar(
                    pos_drivers,
                    x='Count',
                    y='Driver',
                    orientation='h',
                    color='Count',
                    color_continuous_scale='Greens',
                    labels={'Driver': 'Praise Driver', 'Count': 'Mentions'}
                )
                fig_pos.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
                st.plotly_chart(fig_pos, use_container_width=True)

            with col_p2:
                st.write("**Top Praise Keywords & Customer Quotes:**")
                for idx, row in pos_drivers.iterrows():
                    st.markdown(f"""
                    <div style="background-color:#F0FDF4; border-left:4px solid #10B981; padding:0.75rem 1rem; margin-bottom:0.75rem; border-radius:6px;">
                        <strong style="color:#065F46;">💚 {row['Driver']} ({row['Count']} mentions)</strong>
                        <p style="color:#374151; font-style:italic; margin:0.3rem 0 0 0; font-size:0.875rem;">"{row['SampleQuote']}"</p>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:1.5rem;">
            <h4 style="color:#0F172A; margin-top:0;">💼 Executive Business Summary</h4>
            <ol style="color:#334155; line-height:1.7;">
                <li><strong>Revenue Protection:</strong> Addressing top 2 complaint themes (Mobile App Crashes & Refund Delays) mitigates <strong>~80%</strong> of customer churn risk.</li>
                <li><strong>Marketing Value Proposition:</strong> Feature customer praise regarding <em>"fast support resolution"</em> and <em>"easy interface"</em> in marketing campaigns to drive user acquisition.</li>
                <li><strong>Multi-Channel Strategy:</strong> Monitor customer feedback across Google, App Store, and Support Emails in real-time to maintain a high Net Sentiment Index.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
        "Business Owner Review Intelligence & Revenue Protection Hub | Built with Streamlit, Plotly & Scikit-Learn"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
