"""
dashboard/app.py - SentiFi India — Premium Finance Dashboard
Clean, professional trading terminal aesthetic.
"""

import requests
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import os

SIGNAL_COLORS = {"BUY": "#00D4AA", "SELL": "#FF4757", "HOLD": "#FFA502"}
SIGNAL_BG = {"BUY": "rgba(0,212,170,0.08)", "SELL": "rgba(255,71,87,0.08)", "HOLD": "rgba(255,165,2,0.08)"}

INDIAN_STOCKS = [
    ("RELIANCE", "Reliance"),
    ("TCS", "TCS"),
    ("INFY", "Infosys"),
    ("HDFCBANK", "HDFC Bank"),
    ("WIPRO", "Wipro"),
    ("TATAMOTORS", "Tata Motors"),
    ("SBIN", "SBI"),
    ("ICICIBANK", "ICICI Bank"),
    ("BAJFINANCE", "Bajaj Fin"),
    ("ADANIENT", "Adani Ent"),
]

dash_app = dash.Dash(
    __name__,
    requests_pathname_prefix="/dashboard/",
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
    ],
    title="SentiFi — Indian Market Intelligence",
)

# Global CSS to fix input text visibility and button hover
dash_app.index_string = dash_app.index_string.replace(
    "</head>",
    """<style>
    #ticker-input { color: #FFFFFF !important; background-color: #1A2035 !important; caret-color: #00D4AA !important; border: 1px solid #2D3A52 !important; }
    #ticker-input::placeholder { color: #4A5568 !important; opacity: 1 !important; }
    #ticker-input:focus { border-color: #00D4AA !important; outline: none !important; box-shadow: 0 0 0 2px rgba(0,212,170,0.2) !important; }
    input[type=text] { color: #FFFFFF !important; }
    body { background-color: #0B0F1A !important; }
    * { box-sizing: border-box; }
    </style></head>"""
)

CARD_STYLE = {
    "backgroundColor": "#131720",
    "border": "1px solid #1E2636",
    "borderRadius": "8px",
}

LABEL_STYLE = {
    "fontSize": "11px",
    "fontWeight": "600",
    "letterSpacing": "0.08em",
    "color": "#4A5568",
    "textTransform": "uppercase",
    "marginBottom": "4px",
    "fontFamily": "Inter, sans-serif",
}

VALUE_STYLE = {
    "fontSize": "22px",
    "fontWeight": "700",
    "color": "#E2E8F0",
    "fontFamily": "JetBrains Mono, monospace",
    "margin": "0",
}

dash_app.layout = html.Div(
    style={"backgroundColor": "#0B0F1A", "minHeight": "100vh", "fontFamily": "Inter, sans-serif"},
    children=[
        # Top navbar
        html.Div(
            style={
                "backgroundColor": "#0D1117",
                "borderBottom": "1px solid #1E2636",
                "padding": "0 32px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "height": "56px",
            },
            children=[
                html.Div([
                    html.Span("SENTIFI", style={
                        "fontSize": "18px", "fontWeight": "700",
                        "color": "#00D4AA", "fontFamily": "JetBrains Mono, monospace",
                        "letterSpacing": "0.15em",
                    }),
                    html.Span(" INDIA", style={
                        "fontSize": "18px", "fontWeight": "300",
                        "color": "#4A5568", "fontFamily": "JetBrains Mono, monospace",
                        "letterSpacing": "0.15em",
                    }),
                ]),
                html.Div("AI-Powered Market Sentiment · NSE / BSE", style={
                    "fontSize": "12px", "color": "#4A5568",
                    "fontFamily": "Inter, sans-serif", "letterSpacing": "0.05em",
                }),
            ]
        ),

        # Main content
        html.Div(style={"padding": "24px 32px"}, children=[

            # Search row
            html.Div(style={"marginBottom": "20px"}, children=[
                html.Div(style={"display": "flex", "gap": "12px", "alignItems": "center"}, children=[
                    dcc.Input(
                        id="ticker-input",
                        placeholder="Ticker: RELIANCE, TCS, HDFCBANK, NATIONALUM, AAPL...",
                        type="text",
                        debounce=False,
                        style={
                            "flex": "1",
                            "backgroundColor": "#1A2035",
                            "border": "1px solid #2D3A52",
                            "borderRadius": "6px",
                            "color": "#FFFFFF",
                            "padding": "12px 16px",
                            "fontSize": "15px",
                            "fontFamily": "Inter, sans-serif",
                            "outline": "none",
                        }
                    ),
                    html.Button(
                        "ANALYZE",
                        id="analyze-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#00D4AA",
                            "color": "#0B0F1A",
                            "border": "none",
                            "borderRadius": "6px",
                            "padding": "12px 28px",
                            "fontSize": "13px",
                            "fontWeight": "700",
                            "letterSpacing": "0.1em",
                            "cursor": "pointer",
                            "fontFamily": "Inter, sans-serif",
                        }
                    ),
                    dbc.Checklist(
                        options=[
                            {"label": "News (ET/MC)", "value": "reddit"},
                            {"label": "Twitter/X", "value": "twitter"},
                        ],
                        value=["reddit", "twitter"],
                        id="source-toggle",
                        inline=True,
                        style={"color": "#718096", "fontSize": "13px", "whiteSpace": "nowrap"},
                    ),
                ]),
            ]),

            # Quick picks
            html.Div(style={"marginBottom": "28px"}, children=[
                html.Div("NSE TOP PICKS", style=LABEL_STYLE),
                html.Div(
                    style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
                    children=[
                        html.Button(
                            label,
                            id=f"quick-{ticker}",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#131720",
                                "color": "#A0AEC0",
                                "border": "1px solid #1E2636",
                                "borderRadius": "4px",
                                "padding": "6px 14px",
                                "fontSize": "12px",
                                "fontWeight": "500",
                                "cursor": "pointer",
                                "fontFamily": "JetBrains Mono, monospace",
                                "letterSpacing": "0.05em",
                            }
                        )
                        for ticker, label in INDIAN_STOCKS
                    ]
                ),
            ]),

            # Dashboard content
            dcc.Loading(
                id="loading", type="circle", color="#00D4AA",
                children=html.Div(id="dashboard-content"),
            ),

            dcc.Store(id="analysis-store"),
        ]),
    ]
)


def stat_card(label, value, color="#E2E8F0"):
    return html.Div(
        style={**CARD_STYLE, "padding": "20px 24px"},
        children=[
            html.Div(label, style=LABEL_STYLE),
            html.Div(value, style={**VALUE_STYLE, "color": color}),
        ]
    )


def make_signal_card(signal_data):
    sig = signal_data.get("signal", "HOLD")
    color = SIGNAL_COLORS.get(sig, "#FFA502")
    bg = SIGNAL_BG.get(sig, "rgba(255,165,2,0.08)")
    score = signal_data.get("weighted_score", 0)
    posts = signal_data.get("post_count", 0)
    conf = signal_data.get("confidence", 0) * 100
    reasoning = signal_data.get("reasoning", "")

    return html.Div(
        style={**CARD_STYLE, "padding": "28px", "backgroundColor": bg, "borderColor": color},
        children=[
            html.Div(LABEL_STYLE["textTransform"] and "SIGNAL", style=LABEL_STYLE),
            html.Div(sig, style={
                "fontSize": "52px", "fontWeight": "800",
                "color": color, "fontFamily": "JetBrains Mono, monospace",
                "lineHeight": "1", "marginBottom": "16px",
            }),
            html.Div(style={"display": "flex", "gap": "24px", "marginBottom": "16px"}, children=[
                html.Div([
                    html.Div("SCORE", style=LABEL_STYLE),
                    html.Div(f"{score:+.4f}", style={"color": color, "fontFamily": "JetBrains Mono", "fontWeight": "600", "fontSize": "16px"}),
                ]),
                html.Div([
                    html.Div("POSTS", style=LABEL_STYLE),
                    html.Div(str(posts), style={"color": "#E2E8F0", "fontFamily": "JetBrains Mono", "fontWeight": "600", "fontSize": "16px"}),
                ]),
                html.Div([
                    html.Div("CONFIDENCE", style=LABEL_STYLE),
                    html.Div(f"{conf:.1f}%", style={"color": "#E2E8F0", "fontFamily": "JetBrains Mono", "fontWeight": "600", "fontSize": "16px"}),
                ]),
            ]),
            html.Div(style={"borderTop": "1px solid #1E2636", "paddingTop": "12px"}, children=[
                html.Div(reasoning, style={"color": "#718096", "fontSize": "12px", "lineHeight": "1.6"}),
            ]),
        ]
    )


def make_price_chart(prices, currency="INR"):
    if not prices:
        return go.Figure()
    df = pd.DataFrame(prices)
    df["Date"] = pd.to_datetime(df["Date"])
    symbol = "₹" if currency == "INR" else "$"
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00D4AA", increasing_fillcolor="rgba(0,212,170,0.3)",
        decreasing_line_color="#FF4757", decreasing_fillcolor="rgba(255,71,87,0.3)",
        name="Price",
    ))
    fig.update_layout(
        paper_bgcolor="#131720", plot_bgcolor="#0B0F1A",
        font=dict(color="#718096", family="Inter, sans-serif", size=11),
        xaxis=dict(gridcolor="#1E2636", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#1E2636", showgrid=True, zeroline=False, tickprefix=symbol),
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    return fig


def make_sentiment_donut(label_dist):
    labels = ["Positive", "Neutral", "Negative"]
    values = [
        label_dist.get("positive", 0),
        label_dist.get("neutral", 0),
        label_dist.get("negative", 0),
    ]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=["#00D4AA", "#2D3748", "#FF4757"]),
        hole=0.65,
        textfont=dict(color="#E2E8F0", size=12),
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#131720", plot_bgcolor="#131720",
        font=dict(color="#E2E8F0", family="Inter, sans-serif"),
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(font=dict(color="#A0AEC0", size=11), orientation="h", y=-0.1),
        showlegend=True,
        annotations=[dict(
            text=f"{int(values[0]*100)}%<br><span style='font-size:10px'>positive</span>",
            x=0.5, y=0.5, font_size=18, showarrow=False,
            font=dict(color="#00D4AA", family="JetBrains Mono"),
        )]
    )
    return fig


def make_source_bar(reddit_score, twitter_score):
    sources, scores, colors = [], [], []
    if reddit_score is not None:
        sources.append("News (ET/MC)"); scores.append(reddit_score); colors.append("#FF6314")
    if twitter_score is not None:
        sources.append("Twitter / X"); scores.append(twitter_score); colors.append("#1DA1F2")

    fig = go.Figure(go.Bar(
        x=sources, y=scores,
        marker=dict(color=colors, opacity=0.85),
        text=[f"{s:+.3f}" for s in scores],
        textposition="outside",
        textfont=dict(color="#E2E8F0", family="JetBrains Mono", size=13),
        width=0.4,
    ))
    fig.update_layout(
        paper_bgcolor="#131720", plot_bgcolor="#0B0F1A",
        font=dict(color="#718096", family="Inter, sans-serif", size=11),
        yaxis=dict(range=[-1, 1.3], gridcolor="#1E2636", zeroline=True, zerolinecolor="#2D3748"),
        xaxis=dict(showgrid=False),
        margin=dict(t=20, b=10, l=10, r=10),
        showlegend=False,
    )
    return fig


def _do_analysis(ticker, sources):
    if not ticker:
        return html.Div("Enter a ticker symbol above to begin analysis.",
                        style={"color": "#4A5568", "textAlign": "center", "padding": "60px", "fontSize": "14px"}), {}
    ticker = ticker.upper().strip()
    sources_str = ",".join(sources) if sources else "reddit"
    base = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")
    try:
        analysis = requests.get(f"{base}/analyze/{ticker}?sources={sources_str}&limit=50", timeout=120).json()
        stock = requests.get(f"{base}/stock/{ticker}?days=30", timeout=60).json()
    except Exception as e:
        return html.Div(f"Error: {str(e)}", style={"color": "#FF4757", "padding": "20px"}), {}
    prices = stock.get("prices", [])
    info = stock.get("info", {})
    stats = stock.get("stats", {})
    currency = info.get("currency", "USD")
    symbol = "\u20b9" if currency == "INR" else "$"
    content = html.Div([
        html.Div(style={"marginBottom": "20px", "paddingBottom": "16px", "borderBottom": "1px solid #1E2636"}, children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-end"}, children=[
                html.Div([
                    html.Div(info.get("name", ticker), style={
                        "fontSize": "18px", "fontWeight": "700", "color": "#E2E8F0",
                        "marginBottom": "4px", "whiteSpace": "nowrap", "overflow": "hidden",
                        "textOverflow": "ellipsis", "maxWidth": "600px"
                    }),
                    html.Div(style={"display": "flex", "gap": "20px"}, children=[
                        html.Span(ticker, style={"color": "#00D4AA", "fontFamily": "JetBrains Mono", "fontSize": "13px", "fontWeight": "600"}),
                        html.Span(info.get("exchange", ""), style={"color": "#4A5568", "fontSize": "12px"}),
                        html.Span(info.get("sector", ""), style={"color": "#4A5568", "fontSize": "12px"}),
                    ])
                ]),
                html.Div(style={"textAlign": "right"}, children=[
                    html.Div(str(info.get("current_price", "N/A")), style={
                        "fontSize": "28px", "fontWeight": "700", "color": "#E2E8F0", "fontFamily": "JetBrains Mono",
                    }),
                    html.Div(f"MCap: {info.get('market_cap', 'N/A')} \u00b7 PE: {info.get('pe_ratio', 'N/A')}",
                             style={"color": "#4A5568", "fontSize": "12px"}),
                ]),
            ]),
        ]),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "16px", "marginBottom": "16px"}, children=[
            make_signal_card(analysis),
            html.Div(style=CARD_STYLE, children=[
                html.Div(style={"padding": "16px 20px 8px"}, children=[html.Div("SENTIMENT BREAKDOWN", style=LABEL_STYLE)]),
                dcc.Graph(figure=make_sentiment_donut(analysis.get("label_distribution", {})), style={"height": "240px"}, config={"displayModeBar": False}),
            ]),
            html.Div(style=CARD_STYLE, children=[
                html.Div(style={"padding": "16px 20px 8px"}, children=[html.Div("SCORE BY SOURCE", style=LABEL_STYLE)]),
                dcc.Graph(figure=make_source_bar(analysis.get("reddit_score"), analysis.get("twitter_score")), style={"height": "240px"}, config={"displayModeBar": False}),
            ]),
        ]),
        html.Div(style={**CARD_STYLE, "marginBottom": "16px"}, children=[
            html.Div(style={"padding": "16px 20px 8px", "display": "flex", "justifyContent": "space-between"}, children=[
                html.Div("30-DAY PRICE CHART", style=LABEL_STYLE),
                html.Div(f"Volatility: {stats.get('volatility_pct', 'N/A')}%  \u00b7  Range: {symbol}{stats.get('min_price', 'N/A')} \u2014 {symbol}{stats.get('max_price', 'N/A')}",
                         style={"color": "#4A5568", "fontSize": "11px", "fontFamily": "JetBrains Mono"}),
            ]),
            dcc.Graph(figure=make_price_chart(prices, currency), style={"height": "320px"}, config={"displayModeBar": False}),
        ]),
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px"}, children=[
            stat_card("AVG PRICE (30D)", f"{symbol}{stats.get('mean_price', 'N/A')}"),
            stat_card("52W HIGH", str(info.get("52w_high", "N/A")), "#00D4AA"),
            stat_card("52W LOW", str(info.get("52w_low", "N/A")), "#FF4757"),
            stat_card("VOLATILITY", f"{stats.get('volatility_pct', 'N/A')}%"),
        ]),
    ])
    return content, analysis


@dash_app.callback(
    Output("ticker-input", "value"),
    Output("dashboard-content", "children", allow_duplicate=True),
    Output("analysis-store", "data", allow_duplicate=True),
    [Input(f"quick-{t}", "n_clicks") for t, _ in INDIAN_STOCKS],
    State("source-toggle", "value"),
    prevent_initial_call=True,
)
def quick_pick_and_analyze(*args):
    from dash import ctx
    sources = args[-1]
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    ticker = ctx.triggered[0]["prop_id"].split(".")[0].replace("quick-", "")
    content, data = _do_analysis(ticker, sources or ["reddit", "twitter"])
    return ticker, content, data


@dash_app.callback(
    Output("dashboard-content", "children"),
    Output("analysis-store", "data"),
    Input("analyze-btn", "n_clicks"),
    State("ticker-input", "value"),
    State("source-toggle", "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, ticker, sources):
    return _do_analysis(ticker, sources or ["reddit", "twitter"])
