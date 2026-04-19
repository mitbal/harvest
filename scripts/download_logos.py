#!/usr/bin/env python3
"""
Download stock company logos from TradingView for IDX (Indonesia) and/or S&P 500.

TradingView URL patterns:
  IDX:    https://id.tradingview.com/symbols/IDX-{TICKER}/
  NYSE:   https://www.tradingview.com/symbols/NYSE-{TICKER}/
  NASDAQ: https://www.tradingview.com/symbols/NASDAQ-{TICKER}/
  AMEX:   https://www.tradingview.com/symbols/AMEX-{TICKER}/

Logo HTML:
  <img class="logo-PsAlMQQF ..." src="https://s3-symbol-logo.tradingview.com/...svg" ...>

Since TradingView renders logo <img> tags via JavaScript, this script uses
Playwright (headless browser) to load each page. If Playwright is unavailable
it falls back to a plain requests + BeautifulSoup parse (which may miss
dynamically loaded logos).

Install:
    pip install playwright beautifulsoup4 cairosvg
    playwright install chromium

Usage:
    python scripts/download_logos.py [OPTIONS]

Options:
    --market      -m   Which market to download: 'idx', 'sp500', or 'all' (default: idx)
    --output-dir  -o   Directory to save logos (default: asset/logos)
    --delay       -d   Seconds between page loads (default: 1.5)
    --limit       -n   Only process the first N tickers (for testing)
    --tickers          Explicit tickers to download (with optional EXCHANGE:TICKER syntax)
    --no-skip          Re-download even if file already exists
    --no-playwright    Force requests+BeautifulSoup mode (no headless browser)
    --size        -s   Output size in pixels, e.g. 128 means 128×128 (default: original)
    --format           Output format: 'svg' (default) or 'png' (requires cairosvg)
"""

import os
import sys
import re
import time
import argparse
import requests
from pathlib import Path
from typing import NamedTuple

# Add project root so we can import harvest modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harvest.data as hd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# TradingView URL template per exchange prefix
TV_URL_TEMPLATES = {
    "IDX":    "https://id.tradingview.com/symbols/IDX-{ticker}/",
    "NYSE":   "https://www.tradingview.com/symbols/NYSE-{ticker}/",
    "NASDAQ": "https://www.tradingview.com/symbols/NASDAQ-{ticker}/",
    "AMEX":   "https://www.tradingview.com/symbols/AMEX-{ticker}/",
    "CBOE":   "https://www.tradingview.com/symbols/CBOE-{ticker}/",
}

# FMP exchangeShortName → TradingView exchange prefix
FMP_TO_TV_EXCHANGE = {
    "JKT":    "IDX",
    "NYSE":   "NYSE",
    "NASDAQ": "NASDAQ",
    "AMEX":   "AMEX",
    "CBOE":   "CBOE",
    "BATS":   "NASDAQ",    # Bats is absorbed into Nasdaq/CBOE; try NASDAQ
}

# Fallback probe order for S&P 500 if exchange unknown
SP500_EXCHANGE_PROBE_ORDER = ["NYSE", "NASDAQ", "AMEX", "CBOE"]

LOGO_CLASS_PREFIX = "logo-"
LOGO_S3_HOST = "s3-symbol-logo.tradingview.com"
DEFAULT_OUTPUT_DIR = "asset/logos"
DEFAULT_DELAY = 1.5
DEFAULT_TIMEOUT = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class StockEntry(NamedTuple):
    ticker: str          # bare ticker, e.g. "EXCL" or "AAPL"
    tv_exchange: str     # TradingView exchange prefix, e.g. "IDX", "NYSE", "NASDAQ"

    @property
    def tv_url(self) -> str:
        template = TV_URL_TEMPLATES.get(self.tv_exchange, TV_URL_TEMPLATES["NYSE"])
        return template.format(ticker=self.ticker)

    @property
    def file_stem(self) -> str:
        """Unique filename stem that avoids collisions across markets."""
        return self.ticker   # exchange prefix omitted (tickers unique within a run)


# ---------------------------------------------------------------------------
# Stock lists
# ---------------------------------------------------------------------------

def get_idx_stocks() -> list[StockEntry]:
    """Return IDX stocks from harvest data."""
    df = hd.get_all_idx_stocks()
    tickers = df["symbol"].str.split(".").str[0].dropna().unique().tolist()
    return sorted([StockEntry(t, "IDX") for t in tickers], key=lambda e: e.ticker)


def get_sp500_stocks() -> list[StockEntry]:
    """
    Return S&P 500 stocks from harvest data.
    Exchange is resolved via FMP stock/list endpoint for accuracy; if
    not available for a stock, it will be probed at download time.
    """
    import os
    sp_df = hd.get_all_sp500_stocks()
    sp_tickers = set(sp_df["symbol"].dropna().tolist())

    # Try to pull exchange info from the full stock list
    exchange_map: dict[str, str] = {}
    try:
        api_key = os.environ.get("FMP_API_KEY", "")
        if api_key:
            url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={api_key}"
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            import pandas as pd
            all_df = pd.DataFrame(r.json())
            sp_rows = all_df[all_df["symbol"].isin(sp_tickers)]
            for _, row in sp_rows.iterrows():
                fmp_exc = row.get("exchangeShortName", "")
                tv_exc = FMP_TO_TV_EXCHANGE.get(fmp_exc, "NYSE")
                exchange_map[row["symbol"]] = tv_exc
        else:
            print("[WARN] FMP_API_KEY not set – exchange lookup skipped; will probe.")
    except Exception as e:
        print(f"[WARN] Could not fetch exchange map ({e}); will probe per ticker.")

    entries = []
    for ticker in sorted(sp_tickers):
        tv_exc = exchange_map.get(ticker, "PROBE")   # "PROBE" = try at download time
        entries.append(StockEntry(ticker, tv_exc))
    return entries


def build_stock_list(market: str) -> list[StockEntry]:
    if market == "idx":
        return get_idx_stocks()
    elif market == "sp500":
        return get_sp500_stocks()
    elif market == "all":
        # Merge; use exchange-prefixed stem to avoid filename collisions
        return get_idx_stocks() + get_sp500_stocks()
    else:
        raise ValueError(f"Unknown market: {market!r}")


# ---------------------------------------------------------------------------
# Logo URL extraction – Playwright (JS-rendered) path
# ---------------------------------------------------------------------------

def _extract_logo_from_playwright_page(page) -> str | None:
    """Scan the currently loaded Playwright page for a TradingView logo img src."""
    try:
        imgs = page.query_selector_all("img")
        for img in imgs:
            classes = img.get_attribute("class") or ""
            if LOGO_CLASS_PREFIX in classes:
                src = img.get_attribute("src") or ""
                if LOGO_S3_HOST in src:
                    return src
    except Exception:
        pass
    return None


def fetch_logo_url_playwright(entry: StockEntry, page) -> str | None:
    """Navigate to TradingView page(s) and extract the logo src."""
    exchanges_to_try = (
        SP500_EXCHANGE_PROBE_ORDER if entry.tv_exchange == "PROBE"
        else [entry.tv_exchange]
    )

    for exchange in exchanges_to_try:
        template = TV_URL_TEMPLATES.get(exchange, TV_URL_TEMPLATES["NYSE"])
        url = template.format(ticker=entry.ticker)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=DEFAULT_TIMEOUT * 1000)
            page.wait_for_selector(f'img[class*="{LOGO_CLASS_PREFIX}"]', timeout=8000)
        except Exception:
            pass  # still try to read whatever loaded

        src = _extract_logo_from_playwright_page(page)
        if src:
            return src

    return None


# ---------------------------------------------------------------------------
# Logo URL extraction – requests + BeautifulSoup (fallback)
# ---------------------------------------------------------------------------

def fetch_logo_url_requests(entry: StockEntry, session: requests.Session) -> str | None:
    """Fetch TradingView page with requests and parse logo src."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("  [ERROR] beautifulsoup4 not installed. Run: pip install beautifulsoup4")
        return None

    exchanges_to_try = (
        SP500_EXCHANGE_PROBE_ORDER if entry.tv_exchange == "PROBE"
        else [entry.tv_exchange]
    )

    for exchange in exchanges_to_try:
        template = TV_URL_TEMPLATES.get(exchange, TV_URL_TEMPLATES["NYSE"])
        url = template.format(ticker=entry.ticker)
        try:
            resp = session.get(url, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"\n  [WARN] HTTP error ({url}): {e}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        for img in soup.find_all("img"):
            classes = " ".join(img.get("class", []))
            if LOGO_CLASS_PREFIX in classes:
                src = img.get("src", "")
                if LOGO_S3_HOST in src:
                    return src

        # Raw-HTML regex fallback
        matches = re.findall(
            r"https://s3-symbol-logo\.tradingview\.com/[^\"'>\s]+", resp.text
        )
        if matches:
            return matches[0]

    return None


# ---------------------------------------------------------------------------
# File download & resize
# ---------------------------------------------------------------------------

def download_file(url: str, dest_path: Path, session: requests.Session) -> bool:
    try:
        resp = session.get(url, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(resp.content)
        return True
    except requests.RequestException as e:
        print(f"\n  [ERROR] Failed to download {url}: {e}")
        return False


def resize_logo(
    src_path: Path,
    dest_path: Path,
    size: int | None,
    fmt: str,
) -> bool:
    """
    Resize/convert a downloaded SVG logo.

    Args:
        src_path:  Path to the raw SVG file just downloaded.
        dest_path: Final output path (may differ in extension).
        size:      Target pixel dimension (width = height). None = keep original.
        fmt:       'svg' or 'png'.

    Returns True on success.
    """
    svg_bytes = src_path.read_bytes()

    if fmt == "png":
        try:
            import cairosvg
            kwargs: dict = {}
            if size:
                kwargs["output_width"] = size
                kwargs["output_height"] = size
            png_bytes = cairosvg.svg2png(bytestring=svg_bytes, **kwargs)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(png_bytes)
            return True
        except ImportError:
            print("\n  [ERROR] cairosvg not installed. Run: pip install cairosvg")
            return False
        except Exception as e:
            print(f"\n  [ERROR] cairosvg conversion failed: {e}")
            return False

    else:  # fmt == 'svg' – edit width/height attributes in-place
        if size is None:
            if src_path != dest_path:
                import shutil
                shutil.copy2(src_path, dest_path)
            return True
        try:
            import xml.etree.ElementTree as ET
            ET.register_namespace("", "http://www.w3.org/2000/svg")
            tree = ET.parse(src_path)
            root = tree.getroot()
            root.set("width", str(size))
            root.set("height", str(size))
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            tree.write(dest_path, xml_declaration=True, encoding="utf-8")
            return True
        except Exception as e:
            print(f"\n  [ERROR] SVG resize failed: {e}")
            return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download stock logos from TradingView (IDX and/or S&P 500)."
    )
    parser.add_argument(
        "--market", "-m",
        choices=["idx", "sp500", "all"],
        default="idx",
        help="Which market to download: 'idx' (default), 'sp500', or 'all'.",
    )
    parser.add_argument("--output-dir", "-o", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--delay", "-d", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--limit", "-n", type=int, default=None)
    parser.add_argument(
        "--tickers", nargs="*",
        help=(
            "Explicit list of tickers. Optionally prefix with exchange: "
            "'NYSE:AAPL' 'NASDAQ:MSFT' 'IDX:BBCA'. "
            "If no prefix, uses the exchange from --market (default: NYSE for sp500)."
        ),
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Re-download even if logo file already exists.",
    )
    parser.add_argument(
        "--no-playwright", action="store_true",
        help="Use requests+BeautifulSoup only (no headless browser).",
    )
    parser.add_argument(
        "--size", "-s", type=int, default=None, metavar="PX",
        help="Resize logos to PX×PX pixels (e.g. --size 128). Default: keep original.",
    )
    parser.add_argument(
        "--format", choices=["svg", "png"], default="svg",
        help="Output format: 'svg' (default) or 'png' (requires cairosvg).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resolve ticker list ----
    if args.tickers:
        entries: list[StockEntry] = []
        default_exchange = "IDX" if args.market == "idx" else "NYSE"
        for raw in args.tickers:
            if ":" in raw:
                exc, tkr = raw.upper().split(":", 1)
            else:
                exc, tkr = default_exchange, raw.upper()
            entries.append(StockEntry(tkr, exc))
        print(f"Using {len(entries)} explicitly provided ticker(s).")
    else:
        print(f"Fetching {args.market.upper()} stock list from harvest data (FMP API)…")
        entries = build_stock_list(args.market)
        print(f"Found {len(entries)} stocks.")

    if args.limit:
        entries = entries[: args.limit]
        print(f"Limiting to first {args.limit} ticker(s).")

    # ---- Set up backends ----
    use_playwright = not args.no_playwright
    playwright_ctx = pw_browser = pw_page = None

    if use_playwright:
        try:
            from playwright.sync_api import sync_playwright
            playwright_ctx = sync_playwright().start()
            pw_browser = playwright_ctx.chromium.launch(headless=True)
            pw_page = pw_browser.new_page()
            pw_page.set_extra_http_headers(HEADERS)
            print("Playwright (headless Chromium) initialised ✓")
        except Exception as e:
            print(f"[WARN] Playwright not available ({e}). Falling back to requests.")
            use_playwright = False
            playwright_ctx = None

    http_session = requests.Session()
    ok_count = skip_count = fail_count = 0

    try:
        for i, entry in enumerate(entries, start=1):
            prefix = f"[{i:4d}/{len(entries)}] {entry.tv_exchange}:{entry.ticker:<8}"

            # Check existing
            if not args.no_skip:
                stem = entry.file_stem
                existing = list(output_dir.glob(f"{stem}.*"))
                if existing:
                    print(f"{prefix} – already exists ({existing[0].name}), skipping.")
                    skip_count += 1
                    continue

            print(f"{prefix} – fetching logo URL…", end=" ", flush=True)

            if use_playwright:
                logo_url = fetch_logo_url_playwright(entry, pw_page)
            else:
                logo_url = fetch_logo_url_requests(entry, http_session)

            if logo_url is None:
                print("[FAIL] no logo found.")
                fail_count += 1
                time.sleep(args.delay)
                continue

            # Download raw SVG, then resize/convert
            url_path = logo_url.split("?")[0]
            raw_ext = Path(url_path).suffix or ".svg"
            out_ext = f".{args.format}"
            raw_path = output_dir / f"{entry.file_stem}{raw_ext}"
            dest = output_dir / f"{entry.file_stem}{out_ext}"

            print(f"downloading {Path(url_path).name}…", end=" ", flush=True)
            ok = download_file(logo_url, raw_path, http_session)
            if not ok:
                fail_count += 1
                time.sleep(args.delay)
                continue

            needs_transform = (args.size is not None) or (args.format == "png")
            if needs_transform:
                print(f"resizing → {dest.name}…", end=" ", flush=True)
                ok = resize_logo(raw_path, dest, args.size, args.format)
                if ok and dest != raw_path:
                    raw_path.unlink(missing_ok=True)

            if ok:
                print(f"saved → {dest}")
                ok_count += 1
            else:
                fail_count += 1

            time.sleep(args.delay)

    finally:
        if pw_page:
            pw_page.close()
        if pw_browser:
            pw_browser.close()
        if playwright_ctx:
            playwright_ctx.stop()

    print("\n" + "=" * 60)
    print(f"Done.  OK: {ok_count}  |  Skipped: {skip_count}  |  Failed: {fail_count}")
    print(f"Logos saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
