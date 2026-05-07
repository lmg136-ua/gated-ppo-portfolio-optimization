"""Definicion del universo de 100 activos del S&P 500."""

UNIVERSE_100 = {
    "Technology": [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "META",
        "AVGO",
        "ORCL",
        "AMD",
        "QCOM",
        "TXN",
        "INTC",
        "MU",
        "AMAT",
        "LRCX",
        "ADI",
        "NOW",
        "INTU",
        "SNPS",
        "CDNS",
        "MRVL",
    ],
    "Healthcare": [
        "UNH",
        "LLY",
        "JNJ",
        "ABBV",
        "MRK",
        "TMO",
        "ABT",
        "DHR",
        "PFE",
        "AMGN",
        "ISRG",
        "SYK",
        "BMY",
        "GILD",
        "REGN",
    ],
    "Financials": [
        "BRK-B",
        "JPM",
        "V",
        "MA",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "BLK",
        "SPGI",
        "AXP",
        "C",
        "CB",
        "PGR",
        "USB",
    ],
    "Consumer Discretionary": [
        "AMZN",
        "TSLA",
        "HD",
        "MCD",
        "NKE",
        "SBUX",
        "TJX",
        "LOW",
        "BKNG",
        "GM",
    ],
    "Consumer Staples": [
        "PG",
        "KO",
        "PEP",
        "WMT",
        "COST",
        "PM",
        "MO",
        "CL",
    ],
    "Industrials": [
        "GE",
        "CAT",
        "HON",
        "UPS",
        "RTX",
        "LMT",
        "DE",
        "BA",
        "MMM",
        "FDX",
    ],
    "Energy": [
        "XOM",
        "CVX",
        "COP",
        "EOG",
        "SLB",
        "PXD",
        "OXY",
    ],
    "Utilities": [
        "NEE",
        "DUK",
        "SO",
        "D",
        "EXC",
    ],
    "Real Estate": [
        "PLD",
        "AMT",
        "EQIX",
        "SPG",
        "O",
    ],
    "Materials": [
        "LIN",
        "APD",
        "SHW",
        "NEM",
        "FCX",
    ],
}

TICKERS = []
for sector_tickers in UNIVERSE_100.values():
    TICKERS.extend(sector_tickers)

assert len(TICKERS) == 100, f"El universo debe tener 100 activos, tiene {len(TICKERS)}"

TICKER_TO_SECTOR = {}
for sector, tickers in UNIVERSE_100.items():
    for ticker in tickers:
        TICKER_TO_SECTOR[ticker] = sector


def get_tickers() -> list[str]:
    return TICKERS.copy()


def get_tickers_from_config(config: dict | None = None) -> list[str]:
    """
    Resuelve el universo configurado.

    Soporta:
      - source=sp500
      - source=custom con `custom_file`
    """
    if config is None:
        return get_tickers()

    universe_cfg = config.get("universe", {})
    source = universe_cfg.get("source", "sp500")
    if source == "sp500":
        return get_tickers()
    if source == "custom":
        custom_file = universe_cfg.get("custom_file")
        if not custom_file:
            raise ValueError("universe.source=custom requiere universe.custom_file")
        import pandas as pd

        df = pd.read_csv(custom_file)
        if "ticker" not in df.columns:
            raise ValueError("El CSV custom debe tener una columna 'ticker'")
        return df["ticker"].dropna().astype(str).tolist()
    raise ValueError(f"Fuente de universo no soportada: {source}")


def get_universe_dict() -> dict[str, list[str]]:
    return UNIVERSE_100.copy()


def get_ticker_to_sector() -> dict[str, str]:
    return TICKER_TO_SECTOR.copy()


def get_sector_weights() -> dict[str, float]:
    total = len(TICKERS)
    return {sector: len(tickers) / total for sector, tickers in UNIVERSE_100.items()}
