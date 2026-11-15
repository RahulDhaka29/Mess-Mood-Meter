from pathlib import Path
import html

import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# MESS MOOD METER — refined Streamlit dashboard
# Data source: mess_mood_meter_one_week_menu_reviews.csv
# ============================================================

st.set_page_config(
    page_title="Mess Mood Meter | Insights",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "mess_mood_meter_one_week_menu_reviews.csv"
MODEL_FILE = BASE_DIR / "mess_mood_sentiment_model.joblib"

REQUIRED_COLUMNS = [
    "review_id", "date", "day", "meal_slot", "menu_item",
    "rating_out_of_5", "review_text", "sentiment_label_demo", "data_type",
]
SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
SENTIMENT_COLORS = {
    "Positive": "#159A78",
    "Neutral": "#E6A23C",
    "Negative": "#E36B68",
}
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MEAL_ORDER = ["Breakfast", "Lunch", "Snacks", "Dinner"]
PLOT_FONT = "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"


# ---------------------------- Visual theme ----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #182238;
        --muted: #758099;
        --line: #E8ECF3;
        --canvas: #F5F7FB;
        --panel: #FFFFFF;
        --violet: #6257D9;
        --mint: #159A78;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', Inter, ui-sans-serif, system-ui, sans-serif;
    }
    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: rgba(245,247,251,0.88);
    }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container {
        max-width: 1500px;
        padding-top: 2.15rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #172239;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.35rem;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #E8ECF5 !important;
    }
    section[data-testid="stSidebar"] .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 7px 4px 22px 4px;
        border-bottom: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 22px;
    }
    .brand-mark {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        border-radius: 13px;
        color: white;
        font: 800 19px 'Manrope', sans-serif;
        background: linear-gradient(135deg, #8277F4, #5B50C9);
        box-shadow: 0 7px 18px rgba(77,68,190,0.35);
    }
    .brand-name { color: #FFFFFF; font: 800 13px 'Manrope', sans-serif; letter-spacing: 1.45px; }
    .brand-caption { color: #AEB8CE; font-size: 11px; margin-top: 3px; }
    .sidebar-section-label {
        color: #AAB5CC;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.45px;
        text-transform: uppercase;
        margin: 4px 0 13px 2px;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-testid="stDateInput"] input {
        background: #25334E !important;
        border: 1px solid #3B4964 !important;
        border-radius: 10px !important;
        color: #F4F6FB !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="tag"] {
        background: #E8E7FF !important;
        border: 0 !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="tag"] span {
        color: #463DA8 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #AEB8CE !important;
        font-size: 11px;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.12);
        margin: 1.25rem 0;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.11);
        border-radius: 10px;
    }

    /* Main title / hero */
    .eyebrow {
        color: var(--violet);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin-bottom: 9px;
    }
    .hero {
        background: linear-gradient(112deg, #202D49 0%, #303D62 62%, #4C4B8D 100%);
        padding: 27px 30px;
        border-radius: 20px;
        color: #FFFFFF;
        margin-bottom: 17px;
        box-shadow: 0 12px 28px rgba(26,39,68,0.12);
        position: relative;
        overflow: hidden;
    }
    .hero:after {
        content: '';
        position: absolute;
        width: 250px; height: 250px;
        right: -65px; top: -135px;
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(255,255,255,0.035), 0 0 0 58px rgba(255,255,255,0.025);
    }
    .hero-kicker {
        color: #BFC8F8;
        text-transform: uppercase;
        letter-spacing: 1.7px;
        font-size: 10px;
        font-weight: 800;
        margin-bottom: 7px;
    }
    .hero-title {
        color: #FFFFFF;
        font: 800 clamp(26px, 3vw, 36px)/1.12 'Manrope', sans-serif;
        letter-spacing: -1.25px;
        margin: 0;
    }
    .hero-description {
        color: #D4DAE9;
        font-size: 13px;
        margin: 9px 0 0 0;
        max-width: 590px;
    }
    .hero-meta {
        position: relative;
        z-index: 1;
        display: inline-flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 5px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.16);
        padding: 12px 15px;
        border-radius: 13px;
        min-width: 160px;
    }
    .hero-meta-label { color: #BFC8F8; font-size: 9px; font-weight: 800; letter-spacing: 1.3px; text-transform: uppercase; }
    .hero-meta-value { color: #FFFFFF; font-size: 14px; font-weight: 700; }

    /* Sample data notice */
    .data-note {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        background: #FFF9EB;
        border: 1px solid #F4E5BC;
        color: #745B24;
        padding: 11px 14px;
        border-radius: 12px;
        margin: 0 0 21px 0;
        font-size: 12px;
        line-height: 1.5;
    }
    .data-note strong { color: #624A16; }
    .note-dot { width: 7px; height: 7px; margin-top: 5px; border-radius: 50%; background: #D9A441; flex: 0 0 auto; }

    /* KPI cards */
    .metric-card {
        height: 132px;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 17px 18px 15px 18px;
        box-shadow: 0 3px 12px rgba(30,43,72,0.035);
        position: relative;
        overflow: hidden;
    }
    .metric-card:before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        background: var(--accent, #6257D9);
    }
    .metric-label { color: #7B8498; font-size: 11px; font-weight: 700; letter-spacing: .35px; }
    .metric-value { color: #1E2940; font: 800 29px/1.35 'Manrope', sans-serif; letter-spacing: -.7px; margin-top: 8px; }
    .metric-foot { color: #8A93A5; font-size: 10px; margin-top: 3px; }

    /* Sections and tabs */
    .section-heading {
        color: #1E2940;
        font: 800 17px 'Manrope', sans-serif;
        letter-spacing: -.35px;
        margin: 8px 0 3px 0;
    }
    .section-subtitle { color: #8790A2; font-size: 11px; margin: 0 0 12px 0; }
    div[data-testid="stTabs"] button {
        font-weight: 700;
        font-size: 12px;
    }
    div[data-testid="stTabs"] [aria-selected="true"] {
        color: #554AC2 !important;
        border-bottom-color: #6257D9 !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: #6257D9; }
    div[data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 8px 8px 2px 8px;
        box-shadow: 0 3px 12px rgba(30,43,72,0.025);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        border-radius: 10px;
    }
    .empty-state {
        background: #FFFFFF;
        border: 1px dashed #CCD3E0;
        border-radius: 14px;
        padding: 24px;
        color: #657089;
        text-align: center;
    }
    .footer {
        color: #98A0B0;
        font-size: 10px;
        border-top: 1px solid #E7EBF2;
        margin-top: 28px;
        padding-top: 14px;
    }

    @media (max-width: 760px) {
        .block-container { padding-top: 1rem; }
        .hero { padding: 22px 20px; }
        .metric-card { height: 118px; padding: 13px; }
        .metric-value { font-size: 24px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_reviews(file_bytes: bytes) -> pd.DataFrame:
    """Read and validate the supplied CSV from bytes."""
    from io import BytesIO

    data = pd.read_csv(BytesIO(file_bytes), encoding="utf-8-sig")
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["rating_out_of_5"] = pd.to_numeric(data["rating_out_of_5"], errors="coerce")
    for column in ["day", "meal_slot", "menu_item", "sentiment_label_demo", "data_type"]:
        data[column] = data[column].fillna("").astype(str).str.strip()
    data["sentiment_label_demo"] = data["sentiment_label_demo"].str.title()
    data["review_text"] = data["review_text"].fillna("").astype(str).str.strip()

    data = data.dropna(subset=["date", "rating_out_of_5"])
    data = data[data["rating_out_of_5"].between(1, 5)]
    data = data[data["sentiment_label_demo"].isin(SENTIMENT_ORDER)]
    data = data[data["menu_item"].ne("")]
    if data.empty:
        raise ValueError("The CSV loaded, but no valid review rows were found.")
    return data


def section_title(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-heading">{html.escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def chart_layout(fig, height=320):
    fig.update_layout(
        template="plotly_white",
        font=dict(family=PLOT_FONT, color="#657089", size=11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=18, r=18, t=28, b=18),
        height=height,
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#182238", font_color="#FFFFFF", font_family=PLOT_FONT),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#E9EDF3")
    fig.update_yaxes(showgrid=True, gridcolor="#EEF1F6", zeroline=False, linecolor="#E9EDF3")
    return fig


def metric_card(label: str, value: str, foot: str, accent: str) -> str:
    return f"""
    <div class="metric-card" style="--accent:{accent}">
      <div class="metric-label">{html.escape(label)}</div>
      <div class="metric-value">{html.escape(value)}</div>
      <div class="metric-foot">{html.escape(foot)}</div>
    </div>
    """


def render_overview(data: pd.DataFrame) -> None:
    section_title("Sentiment overview", "A snapshot of how the selected meals are being rated.")
    left, right = st.columns([1, 1], gap="large")

    counts = data["sentiment_label_demo"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    sentiment_frame = counts.rename_axis("Sentiment").reset_index(name="Reviews")
    with left:
        st.markdown("**How students feel about the food**")
        fig = px.pie(
            sentiment_frame,
            names="Sentiment",
            values="Reviews",
            hole=0.70,
            color="Sentiment",
            category_orders={"Sentiment": SENTIMENT_ORDER},
            color_discrete_map=SENTIMENT_COLORS,
        )
        fig.update_traces(textposition="outside", textinfo="percent", marker=dict(line=dict(color="white", width=3)))
        fig.add_annotation(
            text=f"<b>{len(data):,}</b><br><span style='font-size:10px;color:#8790A2'>reviews</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=20, color="#202B43", family=PLOT_FONT),
        )
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.12, x=0.05))
        st.plotly_chart(chart_layout(fig, 310), use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown("**Rating breakdown**")
        rating_counts = (
            data["rating_out_of_5"].round().astype(int).value_counts()
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .rename_axis("Rating").reset_index(name="Reviews")
        )
        fig = px.bar(
            rating_counts, x="Rating", y="Reviews", text="Reviews",
            color="Rating", color_continuous_scale=["#E36B68", "#E6A23C", "#159A78"],
        )
        fig.update_traces(marker_line_width=0, textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False, xaxis=dict(dtick=1, title="Rating out of 5"), yaxis_title="Reviews")
        st.plotly_chart(chart_layout(fig, 310), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    section_title("Feedback through the week", "Compare the mix of positive, neutral and negative feedback by day.")
    day_counts = pd.crosstab(data["day"], data["sentiment_label_demo"])
    ordered_days = [day for day in DAY_ORDER if day in day_counts.index]
    ordered_days += [day for day in day_counts.index if day not in ordered_days]
    day_counts = day_counts.reindex(index=ordered_days, columns=SENTIMENT_ORDER, fill_value=0)
    day_long = day_counts.reset_index().melt(id_vars="day", var_name="Sentiment", value_name="Reviews")
    fig = px.bar(
        day_long, x="day", y="Reviews", color="Sentiment", barmode="stack",
        category_orders={"day": ordered_days, "Sentiment": SENTIMENT_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_layout(xaxis_title="", yaxis_title="Number of reviews", legend_title="")
    st.plotly_chart(chart_layout(fig, 330), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    section_title("Meal-time comparison", "See how feedback differs between breakfast, lunch, snacks and dinner.")
    meal_summary = (
        data.groupby("meal_slot", as_index=False)
        .agg(
            Reviews=("review_id", "count"),
            Average_Rating=("rating_out_of_5", "mean"),
            Positive=("sentiment_label_demo", lambda x: (x == "Positive").sum()),
            Neutral=("sentiment_label_demo", lambda x: (x == "Neutral").sum()),
            Negative=("sentiment_label_demo", lambda x: (x == "Negative").sum()),
        )
    )
    meal_summary["meal_slot"] = pd.Categorical(meal_summary["meal_slot"], categories=MEAL_ORDER, ordered=True)
    meal_summary = meal_summary.sort_values("meal_slot")
    meal_left, meal_right = st.columns(2, gap="large")
    with meal_left:
        st.markdown("**Average rating by meal**")
        fig = px.bar(
            meal_summary, x="meal_slot", y="Average_Rating", text=meal_summary["Average_Rating"].round(2),
            color_discrete_sequence=["#6257D9"],
        )
        fig.update_traces(marker_line_width=0, textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_title="", yaxis_title="Average rating", yaxis=dict(range=[0, 5.5], dtick=1))
        st.plotly_chart(chart_layout(fig, 300), use_container_width=True, config={"displayModeBar": False})
    with meal_right:
        st.markdown("**Sentiment by meal**")
        # The source dataframe already has a "Reviews" column (total reviews).
        # Use a different name for the melted sentiment counts to avoid a pandas
        # value_name collision.
        meal_long = meal_summary.melt(
            id_vars="meal_slot", value_vars=SENTIMENT_ORDER,
            var_name="Sentiment", value_name="Sentiment_Reviews",
        )
        fig = px.bar(
            meal_long, x="meal_slot", y="Sentiment_Reviews", color="Sentiment", barmode="group",
            color_discrete_map=SENTIMENT_COLORS,
        )
        fig.update_layout(xaxis_title="", yaxis_title="Number of reviews", legend_title="")
        st.plotly_chart(chart_layout(fig, 300), use_container_width=True, config={"displayModeBar": False})


def render_dishes(data: pd.DataFrame) -> None:
    section_title("Dish performance", "Rank dishes using average ratings, while showing how many reviews support each score.")
    max_reviews = int(data.groupby("menu_item").size().max())
    min_reviews = st.slider(
        "Minimum reviews per dish", min_value=1, max_value=max(1, max_reviews),
        value=min(3, max(1, max_reviews)),
        help="A minimum count helps avoid ranking a dish from only one review.",
    )
    summary = (
        data.groupby("menu_item", as_index=False)
        .agg(
            Reviews=("review_id", "count"),
            Average_Rating=("rating_out_of_5", "mean"),
            Positive=("sentiment_label_demo", lambda x: (x == "Positive").sum()),
            Neutral=("sentiment_label_demo", lambda x: (x == "Neutral").sum()),
            Negative=("sentiment_label_demo", lambda x: (x == "Negative").sum()),
        )
    )
    summary = summary[summary["Reviews"] >= min_reviews].copy()
    summary["Average_Rating"] = summary["Average_Rating"].round(2)
    if summary.empty:
        st.markdown('<div class="empty-state">No dishes meet that review threshold. Try lowering the minimum.</div>', unsafe_allow_html=True)
        return

    best, lowest = st.columns(2, gap="large")
    with best:
        st.markdown("**Highest rated**")
        top = summary.sort_values(["Average_Rating", "Reviews"], ascending=[False, False]).head(5)
        st.dataframe(top[["menu_item", "Average_Rating", "Reviews"]], hide_index=True, use_container_width=True)
    with lowest:
        st.markdown("**Needs a closer look**")
        bottom = summary.sort_values(["Average_Rating", "Reviews"], ascending=[True, False]).head(5)
        st.dataframe(bottom[["menu_item", "Average_Rating", "Reviews"]], hide_index=True, use_container_width=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    section_title("Average rating by dish", "The chart is sorted from lower to higher average rating.")
    plot_data = summary.sort_values("Average_Rating", ascending=True)
    fig = px.bar(
        plot_data, x="Average_Rating", y="menu_item", orientation="h",
        color="Average_Rating", color_continuous_scale=["#E36B68", "#E6A23C", "#159A78"],
        range_x=[0, 5], hover_data=["Reviews", "Positive", "Neutral", "Negative"],
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Average rating out of 5", yaxis_title="", height=max(420, 27 * len(plot_data)))
    st.plotly_chart(chart_layout(fig, max(420, 27 * len(plot_data))), use_container_width=True, config={"displayModeBar": False})


def render_reviews(data: pd.DataFrame) -> None:
    section_title("Review library", "Search individual comments and inspect the feedback behind the charts.")
    search_col, sentiment_col = st.columns([2, 1])
    with search_col:
        search = st.text_input("Search reviews or dishes", placeholder="Try: dosa, spicy, fresh...")
    with sentiment_col:
        selected_sentiment = st.selectbox("Sentiment", ["All sentiments"] + SENTIMENT_ORDER)

    view = data.copy()
    if selected_sentiment != "All sentiments":
        view = view[view["sentiment_label_demo"] == selected_sentiment]
    if search.strip():
        term = search.strip()
        match = (
            view["review_text"].str.contains(term, case=False, regex=False, na=False)
            | view["menu_item"].str.contains(term, case=False, regex=False, na=False)
        )
        view = view[match]

    display_data = view[[
        "date", "day", "meal_slot", "menu_item", "rating_out_of_5",
        "sentiment_label_demo", "review_text",
    ]].sort_values("date", ascending=False).copy()
    display_data["date"] = display_data["date"].dt.strftime("%d %b %Y")
    st.caption(f"{len(display_data):,} reviews match your search.")
    st.dataframe(display_data, hide_index=True, use_container_width=True, height=430)
    st.download_button(
        "Download filtered reviews",
        data=view.to_csv(index=False).encode("utf-8-sig"),
        file_name="mess_mood_meter_filtered_reviews.csv",
        mime="text/csv",
        type="primary",
    )


def render_model_lab() -> None:
    section_title("Sentiment test", "Try the model trained in your notebook on a new food review.")
    if not MODEL_FILE.exists():
        st.markdown(
            '<div class="empty-state"><strong>Model file not found yet</strong><br><br>'
            'Run the training notebook and save <code>mess_mood_sentiment_model.joblib</code> '
            'in the same folder as this app to enable live predictions.</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        import joblib
        model = joblib.load(MODEL_FILE)
    except Exception as error:
        st.error(f"The model could not be loaded: {error}")
        return

    review = st.text_area(
        "Enter a food review",
        placeholder="Example: The dosa was crispy and the chutney tasted fresh.",
        height=110,
    )
    if st.button("Analyse review", type="primary"):
        if not review.strip():
            st.warning("Enter a review before analysing it.")
        else:
            try:
                prediction = model.predict([review])[0]
                st.success(f"Predicted sentiment: {prediction}")
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba([review])[0]
                    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else None
                    classes = classifier.classes_ if classifier is not None else getattr(model, "classes_", [])
                    if len(classes) == len(probs):
                        prob_df = pd.DataFrame({"Sentiment": classes, "Confidence": probs})
                        prob_df["Confidence"] = (prob_df["Confidence"] * 100).round(1)
                        st.dataframe(prob_df.sort_values("Confidence", ascending=False), hide_index=True, use_container_width=True)
            except Exception as error:
                st.error(f"Prediction failed: {error}")


# ---------------------------- Load data ----------------------------
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
      <div class="brand-mark">M</div>
      <div>
        <div class="brand-name">MESS MOOD</div>
        <div class="brand-caption">Feedback intelligence</div>
      </div>
    </div>
    <div class="sidebar-section-label">Refine your view</div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = None
if DATA_FILE.exists():
    try:
        df = load_reviews(DATA_FILE.read_bytes())
    except Exception as error:
        st.error(f"The dataset could not be loaded: {error}")
        st.stop()
else:
    st.warning("The CSV file was not found beside app.py. Upload the project dataset below.")
    uploaded_file = st.file_uploader("Upload the Mess Mood Meter CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Place `mess_mood_meter_one_week_menu_reviews.csv` beside app.py, or upload it here.")
        st.stop()
    try:
        df = load_reviews(uploaded_file.getvalue())
    except Exception as error:
        st.error(f"The uploaded CSV could not be loaded: {error}")
        st.stop()

# ---------------------------- Sidebar filters ----------------------------
st.sidebar.markdown("<div class='sidebar-section-label'>Date and meal</div>", unsafe_allow_html=True)
day_options = [d for d in DAY_ORDER if d in set(df["day"])] + sorted(set(df["day"]) - set(DAY_ORDER))
meal_options = [m for m in MEAL_ORDER if m in set(df["meal_slot"])] + sorted(set(df["meal_slot"]) - set(MEAL_ORDER))
dish_options = sorted(df["menu_item"].dropna().unique().tolist())

selected_days = st.sidebar.multiselect("Days", day_options, default=[], placeholder="All days")
selected_meals = st.sidebar.multiselect("Meal slots", meal_options, default=[], placeholder="All meal slots")
st.sidebar.caption("Leave a filter empty to include everything.")
with st.sidebar.expander("Filter by dish", expanded=False):
    selected_dishes = st.multiselect("Menu items", dish_options, default=[], placeholder="All dishes", label_visibility="collapsed")

min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_value = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date,
)
if isinstance(date_value, (tuple, list)):
    if len(date_value) == 2:
        start_date, end_date = date_value
    elif len(date_value) == 1:
        start_date = end_date = date_value[0]
    else:
        start_date, end_date = min_date, max_date
else:
    start_date = end_date = date_value

filtered = df.copy()
if selected_days:
    filtered = filtered[filtered["day"].isin(selected_days)]
if selected_meals:
    filtered = filtered[filtered["meal_slot"].isin(selected_meals)]
if selected_dishes:
    filtered = filtered[filtered["menu_item"].isin(selected_dishes)]
filtered = filtered[filtered["date"].dt.date.between(start_date, end_date)]

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.caption(f"Showing {len(filtered):,} of {len(df):,} reviews")
st.sidebar.caption("Mess Mood Meter · Demo dashboard")

# ---------------------------- Main header ----------------------------
date_label = f"{start_date.strftime('%d %b')} – {end_date.strftime('%d %b %Y')}"
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-kicker">Dining feedback / Analytics</div>
      <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:22px;flex-wrap:wrap;position:relative;z-index:1;">
        <div>
          <h1 class="hero-title">Mess Mood Meter</h1>
          <p class="hero-description">A clearer picture of what is working on the menu — and what could be better.</p>
        </div>
        <div class="hero-meta">
          <div class="hero-meta-label">Selected period</div>
          <div class="hero-meta-value">{html.escape(date_label)}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Synthetic dataset disclosure grounded in the dataset's data_type field.
data_types = df["data_type"].dropna().astype(str).unique().tolist()
if data_types and all("synthetic" in x.lower() for x in data_types):
    st.markdown(
        '<div class="data-note"><span class="note-dot"></span><div><strong>Prototype dataset</strong> '
        'These are synthetic demo reviews for a one-week sample. The insights illustrate the dashboard workflow; '
        'they are not verified opinions from actual students.</div></div>',
        unsafe_allow_html=True,
    )

if filtered.empty:
    st.markdown(
        '<div class="empty-state"><strong>No reviews in this view</strong><br><br>'
        'Adjust the date range or clear one or more filters in the sidebar.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------- KPI row ----------------------------
total = len(filtered)
avg_rating = filtered["rating_out_of_5"].mean()
counts = filtered["sentiment_label_demo"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
positive_pct = counts["Positive"] / total * 100
negative_pct = counts["Negative"] / total * 100

kpi_cols = st.columns(4, gap="medium")
kpi_values = [
    ("REVIEWS ANALYSED", f"{total:,}", "In the current selection", "#6257D9"),
    ("AVERAGE FOOD RATING", f"{avg_rating:.2f} / 5", "Mean student rating", "#2C91B8"),
    ("POSITIVE FEEDBACK", f"{positive_pct:.1f}%", f"{int(counts['Positive']):,} positive reviews", "#159A78"),
    ("NEGATIVE FEEDBACK", f"{negative_pct:.1f}%", f"{int(counts['Negative']):,} negative reviews", "#E36B68"),
]
for col, (label, value, foot, accent) in zip(kpi_cols, kpi_values):
    with col:
        st.markdown(metric_card(label, value, foot, accent), unsafe_allow_html=True)

st.markdown("<div style='height:17px'></div>", unsafe_allow_html=True)

# ---------------------------- Dashboard tabs ----------------------------
overview_tab, dish_tab, review_tab, model_tab = st.tabs(
    ["Overview", "Dish insights", "Review library", "Sentiment test"]
)
with overview_tab:
    render_overview(filtered)
with dish_tab:
    render_dishes(filtered)
with review_tab:
    render_reviews(filtered)
with model_tab:
    render_model_lab()

st.markdown(
    '<div class="footer">MESS MOOD METER &nbsp;·&nbsp; Dashboard summaries use the supplied '
    '<code>sentiment_label_demo</code> field. Model predictions are shown separately.</div>',
    unsafe_allow_html=True,
)
