"""
dashboard/app.py - Plotly Dash live dashboard for SentiFi
Mounted inside FastAPI using WSGIMiddleware.
"""

import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

SIGNAL_COLORS = {"BUY": "#00C896", "SELL": "#FF4C61", "HOLD": "#F5A623"}

dash_app = dash.Dash(
    __name__,
    requests_pathname_prefix="/dashboard/",
    external_stylesheets=[dbc.themes.DARKLY],
    title="SentiFi — AI Market Sentiment",
)

dash_app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": "#0D1117", "minHeight": "100vh", "padding": "24px"},
    children=[
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("📈 SentiFi", style={"color": "#00C896", "fontFamily": "monospace", "fontWeight": 800}),
                html.P("AI-powered market sentiment from Reddit & Twitter", style={"color": "#8B949E"}),
            ])
        ], className="mb-4"),

        # Search bar
        dbc.Row([
            dbc.Col([
                dbc.InputGroup([
                    dbc.Input(id="ticker-input", placeholder="Enter ticker e.g. AAPL, TSLA, NVDA",
                              type="text", style={"backgroundColor": "#161B22", "color": "#E6EDF3", "border": "1px solid #30363D"}),
                    dbc.Button("Analyze", id="analyze-btn", color="success", n_clicks=0),
                ])
            ], width=6),
            dbc.Col([
                dbc.Checklist(
                    options=[{"label": "Reddit", "value": "reddit"}, {"label": "Twitter/X", "value": "twitter"}],
                    value=["reddit", "twitter"],
                    id="source-toggle",
                    inline=True,
                    style={"color": "#E6EDF3", "marginTop": "8px"},
                )
            ], width=3),
        ], className="mb-4"),

        # Loading spinner
        dcc.Loading(
            id="loading",
            type="circle",
            color="#00C896",
            children=html.Div(id="dashboard-content"),
        ),

        # Store
        dcc.Store(id="analysis-store"),
    ]
)


def make_signal_card(signal_data: dict) -> dbc.Card:
    sig = signal_data.get("signal", "HOLD")
    color = SIGNAL_COLORS.get(sig, "#F5A623")
    return dbc.Card(
        dbc.CardBody([
            html.H2(sig, style={"color": color, "fontFamily": "monospace", "fontWeight": 900, "fontSize": "3rem"}),
            html.P(f"Weighted Score: {signal_data.get('weighted_score', 0):+.4f}", style={"color": "#E6EDF3"}),
            html.P(f"Posts Analyzed: {signal_data.get('post_count', 0)}", style={"color": "#8B949E"}),
            html.P(f"Confidence: {signal_data.get('confidence', 0)*100:.1f}%", style={"color": "#8B949E"}),
            html.Hr(style={"borderColor": "#30363D"}),
            html.P(signal_data.get("reasoning", ""), style={"color": "#8B949E", "fontSize": "0.85rem"}),
        ]),
        style={"backgroundColor": "#161B22", "border": f"1px solid {color}"},
    )


def make_sentiment_pie(label_dist: dict) -> go.Figure:
    labels = list(label_dist.keys())
    values = list(label_dist.values())
    colors = ["#00C896", "#8B949E", "#FF4C61"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors),
        hole=0.5,
        textfont=dict(color="#E6EDF3"),
    ))
    fig.update_layout(
        paper_bgcolor="#161B22", plot_bgcolor="#161B22",
        font=dict(color="#E6EDF3"),
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(font=dict(color="#E6EDF3")),
        showlegend=True,
    )
    return fig


def make_price_chart(prices: list) -> go.Figure:
    if not prices:
        return go.Figure()
    df = pd.DataFrame(prices)
    df["Date"] = pd.to_datetime(df["Date"])
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00C896",
        decreasing_line_color="#FF4C61",
        name="Price",
    ))
    fig.update_layout(
        paper_bgcolor="#161B22", plot_bgcolor="#0D1117",
        font=dict(color="#E6EDF3"),
        xaxis=dict(gridcolor="#21262D", showgrid=True),
        yaxis=dict(gridcolor="#21262D", showgrid=True),
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis_rangeslider_visible=False,
    )
    return fig


def make_source_bar(reddit_score, twitter_score) -> go.Figure:
    sources, scores, colors = [], [], []
    if reddit_score is not None:
        sources.append("Reddit")
        scores.append(reddit_score)
        colors.append("#FF6314")
    if twitter_score is not None:
        sources.append("Twitter/X")
        scores.append(twitter_score)
        colors.append("#1DA1F2")

    fig = go.Figure(go.Bar(
        x=sources, y=scores,
        marker_color=colors,
        text=[f"{s:+.3f}" for s in scores],
        textposition="auto",
    ))
    fig.update_layout(
        paper_bgcolor="#161B22", plot_bgcolor="#0D1117",
        font=dict(color="#E6EDF3"),
        yaxis=dict(range=[-1, 1], gridcolor="#21262D"),
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False,
    )
    fig.add_hline(y=0, line_color="#8B949E", line_dash="dot")
    return fig


@dash_app.callback(
    Output("dashboard-content", "children"),
    Output("analysis-store", "data"),
    Input("analyze-btn", "n_clicks"),
    State("ticker-input", "value"),
    State("source-toggle", "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, ticker, sources):
    if not ticker:
        return dbc.Alert("Please enter a ticker symbol.", color="warning"), {}

    ticker = ticker.upper().strip()
    sources_str = ",".join(sources) if sources else "reddit"

    try:
        # Call the FastAPI backend
        import os
        base = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")
        analysis = requests.get(f"{base}/analyze/{ticker}?sources={sources_str}&limit=50", timeout=120).json()
        stock = requests.get(f"{base}/stock/{ticker}?days=30", timeout=30).json()
    except Exception as e:
        return dbc.Alert(f"API Error: {str(e)}", color="danger"), {}

    prices = stock.get("prices", [])
    info = stock.get("info", {})
    stats = stock.get("stats", {})

    content = [
        # Stock info bar
        dbc.Row([
            dbc.Col(html.H4(f"{info.get('name', ticker)} ({ticker})", style={"color": "#E6EDF3"})),
            dbc.Col(html.P(f"Sector: {info.get('sector', 'N/A')} | Current: ${info.get('current_price', 'N/A')} | PE: {info.get('pe_ratio', 'N/A')}",
                           style={"color": "#8B949E", "textAlign": "right"})),
        ], className="mb-3"),

        # Signal + Pie
        dbc.Row([
            dbc.Col(make_signal_card(analysis), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Sentiment Distribution", style={"color": "#8B949E"}),
                dcc.Graph(figure=make_sentiment_pie(analysis.get("label_distribution", {})), style={"height": "220px"}),
            ]), style={"backgroundColor": "#161B22", "border": "1px solid #30363D"}), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Score by Source", style={"color": "#8B949E"}),
                dcc.Graph(figure=make_source_bar(analysis.get("reddit_score"), analysis.get("twitter_score")), style={"height": "220px"}),
            ]), style={"backgroundColor": "#161B22", "border": "1px solid #30363D"}), width=4),
        ], className="mb-4"),

        # Price chart
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6(f"30-Day Price Chart | Volatility: {stats.get('volatility_pct', 'N/A')}%", style={"color": "#8B949E"}),
                dcc.Graph(figure=make_price_chart(prices), style={"height": "350px"}),
            ]), style={"backgroundColor": "#161B22", "border": "1px solid #30363D"})),
        ], className="mb-4"),

        # Stats row
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Avg Price", style={"color": "#8B949E", "marginBottom": "4px"}),
                html.H5(f"${stats.get('mean_price', 'N/A')}", style={"color": "#E6EDF3"}),
            ]), style={"backgroundColor": "#161B22", "border": "1px solid #30363D"}), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("52W High", style={"color": "#8B949E", "marginBottom": "4px"}),
                html.H5(f"${info.get('52w_high', 'N/A')}", style={"color": "#00C896"}),
            ]), style={"backgroundColor": "#161B22", "border": "1px solid #30363D"}), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("52W Low", style={"color": "#8B949E", "marginBottom": "4px"}),
                html.H5(f"${info.get('52w_low', 'N/A')}", style={"color": "#FF4C61"}),
            ]), style={"backgroundColor": "#161B22", "border": "1px solid #30363D"}), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Price Range (30d)", style={"color": "#8B949E", "marginBottom": "4px"}),
                html.H5(f"${stats.get('price_range', 'N/A')}", style={"color": "#E6EDF3"}),
            ]), style={"backgroundColor": "#161B22", "border": "1px solid #30363D"}), width=3),
        ]),
    ]

    return content, analysis
