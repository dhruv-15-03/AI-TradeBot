#!/usr/bin/env python3
"""
 ███████╗████████╗ ██████╗  ██████╗██╗  ██╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗
 ██╔════╝╚══██╔══╝██╔═══██╗██╔════╝██║ ██╔╝    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
 ███████╗   ██║   ██║   ██║██║     █████╔╝     ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
 ╚════██║   ██║   ██║   ██║██║     ██╔═██╗     ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
 ███████║   ██║   ╚██████╔╝╚██████╗██║  ██╗    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║
 ╚══════╝   ╚═╝    ╚═════╝  ╚═════╝╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═══╝   ╚═╝

Interactive Stock Market Analysis Agent — Real-time data, Technical Analysis,
News Sentiment, and AI-powered recommendations.

DISCLAIMER: This tool is for educational/informational purposes only.
It is NOT financial advice. Always do your own research before investing.
"""

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich import box
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from market_data import get_stock_info, get_historical_data, get_multiple_quotes, get_market_movers
from news_engine import get_all_news, get_market_news
from technical_analysis import compute_all_indicators, generate_signals, get_support_resistance
from stock_analyzer import full_stock_analysis, scan_stocks, analyze_fundamentals, WATCHLISTS

console = Console()
history = InMemoryHistory()

# ─── Formatting Helpers ───────────────────────────────────────────────

def color_change(val: float, suffix: str = "") -> str:
    if val > 0:
        return f"[bold green]+{val:.2f}{suffix}[/]"
    elif val < 0:
        return f"[bold red]{val:.2f}{suffix}[/]"
    return f"[dim]{val:.2f}{suffix}[/]"


def verdict_style(verdict: str) -> str:
    styles = {
        "STRONG BUY": "[bold white on green]",
        "BUY": "[bold green]",
        "HOLD": "[bold yellow]",
        "SELL": "[bold red]",
        "STRONG SELL": "[bold white on red]",
    }
    prefix = styles.get(verdict, "[bold]")
    return f"{prefix} {verdict} [/]"


def sentiment_style(label: str) -> str:
    if label == "BULLISH":
        return "[bold green]BULLISH[/]"
    elif label == "BEARISH":
        return "[bold red]BEARISH[/]"
    return "[dim]NEUTRAL[/]"


def risk_style(level: str) -> str:
    if level == "HIGH":
        return "[bold red]HIGH RISK[/]"
    elif level == "MODERATE":
        return "[bold yellow]MODERATE[/]"
    return "[bold green]LOW RISK[/]"


# ─── Display Functions ────────────────────────────────────────────────

def show_market_overview():
    """Display major indices and trending stocks."""
    with Progress(SpinnerColumn(), TextColumn("[bold blue]Fetching market data..."),
                  console=console, transient=True) as p:
        p.add_task("", total=None)
        data = get_market_movers()

    # Indices table
    t = Table(title="Major Indices", box=box.ROUNDED, border_style="blue")
    t.add_column("Index", style="bold white")
    t.add_column("Price", justify="right")
    t.add_column("Change", justify="right")
    t.add_column("% Change", justify="right")
    for idx in data["indices"]:
        t.add_row(
            idx["name"],
            f"${idx['price']:,.2f}" if idx["price"] else "N/A",
            color_change(idx["change"]),
            color_change(idx["change_pct"], "%"),
        )
    console.print(t)

    # Trending stocks
    t2 = Table(title="Trending Stocks", box=box.ROUNDED, border_style="cyan")
    t2.add_column("Ticker", style="bold cyan")
    t2.add_column("Name", style="dim")
    t2.add_column("Price", justify="right")
    t2.add_column("Change", justify="right")
    t2.add_column("% Change", justify="right")
    t2.add_column("Volume", justify="right")
    for s in data["trending"]:
        t2.add_row(
            s["ticker"],
            s["name"][:25],
            f"${s['price']:,.2f}" if s["price"] else "N/A",
            color_change(s["change"]),
            color_change(s["change_pct"], "%"),
            f"{s['volume']:,}" if s['volume'] else "N/A",
        )
    console.print(t2)


def show_full_analysis(ticker: str):
    """Run and display comprehensive stock analysis."""
    with Progress(SpinnerColumn(), TextColumn(f"[bold blue]Analyzing {ticker.upper()}..."),
                  console=console, transient=True) as p:
        p.add_task("", total=None)
        result = full_stock_analysis(ticker)

    info = result["stock_info"]
    tech = result["technical"]
    news = result["news"]
    fund = result["fundamentals"]
    sr = result["support_resistance"]
    trade = result["trade_plan"]

    # === Header ===
    price_change = info["price"] - info["previous_close"]
    price_pct = (price_change / info["previous_close"] * 100) if info["previous_close"] else 0

    header = (
        f"[bold white]{info['name']}[/] ([cyan]{result['ticker']}[/])\n"
        f"[bold]${info['price']:,.2f}[/]  {color_change(price_change)}  "
        f"{color_change(price_pct, '%')}\n"
        f"Sector: {info['sector']}  |  Industry: {info['industry']}\n"
        f"Market Cap: ${info['market_cap']/1e9:.1f}B  |  "
        f"Volume: {info['volume']:,}  |  Avg Volume: {info['avg_volume']:,}"
    )
    console.print(Panel(header, title=f"Stock Overview", border_style="blue"))

    # === VERDICT ===
    verdict_box = (
        f"\n  VERDICT:  {verdict_style(result['verdict'])}\n\n"
        f"  Combined Score: [bold]{result['combined_score']:.2f}[/] / 10\n"
        f"  Risk: {risk_style(result['risk']['level'])} "
        f"(Daily volatility: {result['risk']['daily_volatility_pct']:.2f}%)\n"
    )
    console.print(Panel(verdict_box, title="AI Recommendation",
                        border_style="green" if "BUY" in result["verdict"] else
                        "red" if "SELL" in result["verdict"] else "yellow"))

    # === Trade Plan ===
    tp = Table(title="Trade Plan", box=box.SIMPLE_HEAVY, border_style="magenta")
    tp.add_column("Parameter", style="bold")
    tp.add_column("Value", justify="right")
    tp.add_row("Suggested Entry", f"[cyan]${trade['suggested_entry']}[/]")
    tp.add_row("Stop Loss", f"[red]${trade['stop_loss']}[/]")
    tp.add_row("Target 1", f"[green]${trade['target_1']}[/]")
    tp.add_row("Target 2", f"[bold green]${trade['target_2']}[/]")
    tp.add_row("Risk/Reward", f"[bold]{trade['risk_reward']:.2f}[/]")
    console.print(tp)

    # === Technical Signals ===
    st = Table(title="Technical Signals", box=box.ROUNDED, border_style="cyan")
    st.add_column("Signal", style="bold")
    st.add_column("Action")
    st.add_column("Weight", justify="center")
    st.add_column("Detail", style="dim")
    for sig in tech["signals"]:
        action_color = {"BUY": "green", "STRONG BUY": "bold green",
                        "SELL": "red", "STRONG SELL": "bold red",
                        "HOLD": "yellow", "ALERT": "magenta"}.get(sig["action"], "white")
        st.add_row(
            sig["name"],
            f"[{action_color}]{sig['action']}[/]",
            str(sig["weight"]),
            sig["detail"],
        )
    st.add_row("", "", "", "")
    st.add_row(
        "[bold]NET SCORE[/]",
        f"[bold]Buy: {tech['buy_score']} | Sell: {tech['sell_score']}[/]",
        f"[bold]{tech['net_score']:+d}[/]",
        f"Confidence: {tech['confidence']}%",
    )
    console.print(st)

    # === Key Indicators ===
    ind = tech.get("indicators", {})
    it = Table(title="Key Indicators", box=box.SIMPLE, border_style="blue")
    it.add_column("Indicator", style="bold")
    it.add_column("Value", justify="right")
    it.add_column("Interpretation", style="dim")
    it.add_row("RSI (14)", f"{ind.get('rsi', 0):.1f}",
               "Oversold" if ind.get("rsi", 50) < 30 else
               "Overbought" if ind.get("rsi", 50) > 70 else "Neutral")
    it.add_row("MACD", f"{ind.get('macd', 0):.4f}", "")
    it.add_row("ADX", f"{ind.get('adx', 0):.1f}",
               "Strong trend" if ind.get("adx", 0) > 25 else "Weak trend")
    it.add_row("BB Position", f"{ind.get('bb_position', 50):.1f}%",
               "Near lower band" if ind.get("bb_position", 50) < 20 else
               "Near upper band" if ind.get("bb_position", 50) > 80 else "Mid-range")
    it.add_row("Stochastic K", f"{ind.get('stoch_k', 0):.1f}", "")
    it.add_row("ATR", f"{ind.get('atr', 0):.4f}", "")
    it.add_row("Volume Ratio", f"{ind.get('volume_ratio', 1):.2f}x",
               "HIGH" if ind.get("volume_ratio", 1) > 2 else "Normal")
    console.print(it)

    # === Support / Resistance ===
    srt = Table(title="Support & Resistance", box=box.SIMPLE, border_style="yellow")
    srt.add_column("Level", style="bold")
    srt.add_column("Price", justify="right")
    srt.add_row("[green]Resistance 3[/]", f"[green]${sr.get('resistance_3', 'N/A')}[/]")
    srt.add_row("[green]Resistance 2[/]", f"[green]${sr.get('resistance_2', 'N/A')}[/]")
    srt.add_row("[green]Resistance 1[/]", f"[green]${sr.get('resistance_1', 'N/A')}[/]")
    srt.add_row("[bold white]Pivot[/]", f"[bold]${sr.get('pivot', 'N/A')}[/]")
    srt.add_row("[red]Support 1[/]", f"[red]${sr.get('support_1', 'N/A')}[/]")
    srt.add_row("[red]Support 2[/]", f"[red]${sr.get('support_2', 'N/A')}[/]")
    srt.add_row("[red]Support 3[/]", f"[red]${sr.get('support_3', 'N/A')}[/]")
    console.print(srt)

    # === Fundamentals ===
    ft = Table(title="Fundamental Analysis", box=box.ROUNDED, border_style="green")
    ft.add_column("Metric", style="bold")
    ft.add_column("Assessment")
    for d in fund["details"]:
        ft.add_row("", d)
    ft.add_row("[bold]Score[/]", f"[bold]{fund['fundamental_score']} / {fund['max_score']}[/]")
    console.print(ft)

    # === Analyst Targets ===
    if info.get("target_mean"):
        at = Table(title="Analyst Consensus", box=box.SIMPLE, border_style="magenta")
        at.add_column("", style="bold")
        at.add_column("Value", justify="right")
        at.add_row("Recommendation", f"[bold]{info['recommendation'].upper()}[/]")
        at.add_row("# Analysts", str(info['analyst_count']))
        at.add_row("Target Low", f"${info['target_low']:.2f}")
        at.add_row("Target Mean", f"[bold]${info['target_mean']:.2f}[/]")
        at.add_row("Target High", f"${info['target_high']:.2f}")
        upside = (info['target_mean'] - info['price']) / info['price'] * 100
        at.add_row("Upside", color_change(upside, "%"))
        console.print(at)

    # === News Sentiment ===
    ns = news["sentiment_summary"]
    console.print(Panel(
        f"  Overall: {sentiment_style(ns['overall'])}  |  "
        f"Avg Polarity: {ns['average_polarity']:.3f}\n"
        f"  Articles: {news['total_articles']}  |  "
        f"[green]Bullish: {ns['bullish_count']}[/]  |  "
        f"[red]Bearish: {ns['bearish_count']}[/]  |  "
        f"[dim]Neutral: {ns['neutral_count']}[/]",
        title="News Sentiment", border_style="cyan",
    ))

    # Top 5 headlines
    nt = Table(title="Recent Headlines", box=box.SIMPLE, show_lines=True)
    nt.add_column("Source", style="dim", width=12)
    nt.add_column("Headline")
    nt.add_column("Sentiment", justify="center", width=10)
    for article in news["articles"][:8]:
        sent = article.get("sentiment", {})
        nt.add_row(
            article["source"],
            article["title"][:120],
            sentiment_style(sent.get("label", "NEUTRAL")),
        )
    console.print(nt)


def show_news(ticker: str):
    """Display news for a ticker."""
    with Progress(SpinnerColumn(), TextColumn(f"[bold blue]Fetching news for {ticker.upper()}..."),
                  console=console, transient=True) as p:
        p.add_task("", total=None)
        news = get_all_news(ticker, limit_per_source=8)

    ns = news["sentiment_summary"]
    console.print(Panel(
        f"  Overall: {sentiment_style(ns['overall'])}  |  "
        f"Polarity: {ns['average_polarity']:.3f}  |  "
        f"[green]Bullish: {ns['bullish_count']}[/]  "
        f"[red]Bearish: {ns['bearish_count']}[/]  "
        f"[dim]Neutral: {ns['neutral_count']}[/]",
        title=f"News Sentiment — {ticker.upper()}", border_style="cyan",
    ))

    t = Table(box=box.SIMPLE, show_lines=True)
    t.add_column("Source", style="dim", width=14)
    t.add_column("Headline")
    t.add_column("Sent", justify="center", width=10)
    t.add_column("Score", justify="right", width=8)
    for a in news["articles"]:
        sent = a.get("sentiment", {})
        t.add_row(
            a["source"], a["title"][:120],
            sentiment_style(sent.get("label", "NEUTRAL")),
            f"{sent.get('polarity', 0):.2f}",
        )
    console.print(t)


def show_scan(tickers: list[str]):
    """Scan multiple stocks and show ranked results."""
    with Progress(SpinnerColumn(),
                  TextColumn(f"[bold blue]Scanning {len(tickers)} stocks..."),
                  console=console, transient=True) as p:
        p.add_task("", total=None)
        results = scan_stocks(tickers)

    t = Table(title="Stock Scanner Results (Ranked)", box=box.ROUNDED, border_style="cyan")
    t.add_column("#", style="dim", width=3)
    t.add_column("Ticker", style="bold cyan")
    t.add_column("Name", style="dim", max_width=20)
    t.add_column("Price", justify="right")
    t.add_column("Verdict")
    t.add_column("Score", justify="right")
    t.add_column("Technical", justify="center")
    t.add_column("Sentiment", justify="center")
    t.add_column("Risk", justify="center")

    for i, r in enumerate(results, 1):
        if "error" in r:
            t.add_row(str(i), r["ticker"], "", "", f"[red]Error[/]", "", "", "", "")
            continue
        t.add_row(
            str(i),
            r["ticker"],
            r.get("name", "")[:20],
            f"${r.get('price', 0):,.2f}",
            verdict_style(r["verdict"]),
            f"[bold]{r['combined_score']:.2f}[/]",
            r.get("tech_action", ""),
            sentiment_style(r.get("sentiment", "NEUTRAL")),
            risk_style(r.get("risk", "MODERATE")),
        )
    console.print(t)


def show_quick_quote(tickers: list[str]):
    """Show quick price quotes."""
    with Progress(SpinnerColumn(), TextColumn("[bold blue]Fetching quotes..."),
                  console=console, transient=True) as p:
        p.add_task("", total=None)
        quotes = get_multiple_quotes(tickers)

    t = Table(title="Quick Quotes", box=box.ROUNDED, border_style="blue")
    t.add_column("Ticker", style="bold cyan")
    t.add_column("Name", style="dim")
    t.add_column("Price", justify="right")
    t.add_column("Change", justify="right")
    t.add_column("% Change", justify="right")
    t.add_column("Volume", justify="right")
    for q in quotes:
        t.add_row(
            q["ticker"], q["name"][:25],
            f"${q['price']:,.2f}" if q["price"] else "N/A",
            color_change(q["change"]),
            color_change(q["change_pct"], "%"),
            f"{q['volume']:,}" if q['volume'] else "N/A",
        )
    console.print(t)


def show_help():
    """Show help menu."""
    help_text = """
[bold cyan]Available Commands:[/]

[bold white]analyze <TICKER>[/]       Full deep analysis (technicals + news + fundamentals + AI verdict)
[bold white]news <TICKER>[/]          Get latest news + sentiment analysis
[bold white]quote <TICKER,...>[/]     Quick price quotes (comma-separated)
[bold white]scan <TICKER,...>[/]      Scan & rank multiple stocks
[bold white]market[/]                 Market overview (indices + trending stocks)
[bold white]watchlist <name>[/]       Scan a preset watchlist

[bold cyan]Preset Watchlists:[/]
  [dim]mega_cap[/]   — AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK-B, JPM, V
  [dim]growth[/]     — NVDA, AMD, PLTR, CRWD, SNOW, NET, DDOG, MDB, SHOP, SQ
  [dim]dividend[/]   — JNJ, PG, KO, PEP, MMM, T, VZ, XOM, CVX, ABBV
  [dim]etf[/]        — SPY, QQQ, IWM, DIA, VTI, ARKK, XLF, XLE, XLK, SCHD
  [dim]volatile[/]   — TSLA, GME, AMC, RIVN, MARA, COIN, SOFI, PLTR, NIO, LCID

[bold cyan]Examples:[/]
  [dim]> analyze NVDA[/]
  [dim]> news TSLA[/]
  [dim]> quote AAPL, MSFT, GOOGL[/]
  [dim]> scan NVDA, AMD, INTC, TSM, AVGO[/]
  [dim]> watchlist growth[/]
  [dim]> market[/]

[bold yellow]Tip: For best results, combine 'analyze' with 'news' for a full picture.[/]

[bold red]DISCLAIMER: This is NOT financial advice. Always DYOR.[/]
"""
    console.print(Panel(help_text, title="Stock Agent Help", border_style="cyan"))


# ─── Main Loop ────────────────────────────────────────────────────────

def main():
    banner = """
[bold cyan]
 ███████╗████████╗ ██████╗  ██████╗██╗  ██╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗
 ██╔════╝╚══██╔══╝██╔═══██╗██╔════╝██║ ██╔╝    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
 ███████╗   ██║   ██║   ██║██║     █████╔╝     ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
 ╚════██║   ██║   ██║   ██║██║     ██╔═██╗     ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
 ███████║   ██║   ╚██████╔╝╚██████╗██║  ██╗    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║
 ╚══════╝   ╚═╝    ╚═════╝  ╚═════╝╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═══╝   ╚═╝
[/]
[bold white]  Real-time Market Data  |  Technical Analysis  |  AI-Powered Signals[/]
[bold yellow]  DISCLAIMER: For educational purposes only. Not financial advice.[/]
[dim]  Type 'help' for commands  |  'quit' to exit[/]
"""
    console.print(banner)

    while True:
        try:
            user_input = prompt(
                "\n📈 StockAgent > ",
                history=history,
                auto_suggest=AutoSuggestFromHistory(),
            ).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold cyan]Goodbye! Happy trading![/]")
            break

        if not user_input:
            continue

        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        try:
            if cmd in ("quit", "exit", "q"):
                console.print("[bold cyan]Goodbye! Happy trading![/]")
                break

            elif cmd == "help":
                show_help()

            elif cmd in ("analyze", "a", "analysis"):
                if not args:
                    console.print("[red]Usage: analyze <TICKER>[/]")
                else:
                    show_full_analysis(args.split()[0])

            elif cmd == "news":
                if not args:
                    console.print("[blue]Fetching general market news...[/]")
                    with Progress(SpinnerColumn(), TextColumn("[bold blue]..."),
                                  console=console, transient=True) as p:
                        p.add_task("", total=None)
                        articles = get_market_news(15)
                    t = Table(title="Market News", box=box.SIMPLE, show_lines=True)
                    t.add_column("Source", style="dim", width=14)
                    t.add_column("Headline")
                    t.add_column("Sent", justify="center", width=10)
                    for a in articles:
                        s = a.get("sentiment", {})
                        t.add_row(a["source"], a["title"][:120],
                                  sentiment_style(s.get("label", "NEUTRAL")))
                    console.print(t)
                else:
                    show_news(args.split()[0])

            elif cmd in ("quote", "q", "price"):
                if not args:
                    console.print("[red]Usage: quote <TICKER,...>[/]")
                else:
                    tickers = [t.strip().upper() for t in args.replace(",", " ").split() if t.strip()]
                    show_quick_quote(tickers)

            elif cmd in ("scan", "rank"):
                if not args:
                    console.print("[red]Usage: scan <TICKER,...>[/]")
                else:
                    tickers = [t.strip().upper() for t in args.replace(",", " ").split() if t.strip()]
                    show_scan(tickers)

            elif cmd in ("market", "overview", "m"):
                show_market_overview()

            elif cmd in ("watchlist", "wl"):
                name = args.lower() if args else "mega_cap"
                if name in WATCHLISTS:
                    console.print(f"[bold]Scanning watchlist: {name}[/]")
                    show_scan(WATCHLISTS[name])
                else:
                    console.print(f"[red]Unknown watchlist. Available: {', '.join(WATCHLISTS.keys())}[/]")

            else:
                # Treat unknown input as a ticker for quick analysis
                console.print(f"[dim]Unknown command. Trying as ticker...[/]")
                show_full_analysis(cmd.upper())

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/]")
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/]")
            console.print("[dim]Check the ticker symbol and try again.[/]")


if __name__ == "__main__":
    main()
