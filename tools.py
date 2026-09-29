"""
Real tools for the Jarvis agent.

- get_weather      -> Open-Meteo (free, no API key)
- get_stock_price  -> Yahoo Finance via yfinance (free, no API key)

Install:  pip install requests yfinance langchain-core
"""

import requests
import yfinance as yf
from langchain_core.tools import tool

HTTP_TIMEOUT = 10  # seconds

# WMO weather codes used by Open-Meteo -> readable text
WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name only, e.g. "Bengaluru", "Indore", "London".
    """
    name = city.split(",")[0].strip()  # geocoder matches on the place name only
    try:
        # 1) City name -> coordinates
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 1, "language": "en", "format": "json"},
            timeout=HTTP_TIMEOUT,
        ).json()
        if not geo.get("results"):
            return f"Could not find a place called '{city}'."
        place = geo["results"][0]

        # 2) Coordinates -> current conditions
        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                           "weather_code,wind_speed_10m,precipitation",
                "timezone": "auto",
            },
            timeout=HTTP_TIMEOUT,
        ).json()
        cur = wx["current"]
    except Exception as e:
        return f"Weather service error: {e}"

    return (
        f"Weather in {place['name']}, {place.get('country', '')} "
        f"(as of {cur['time']} local time): "
        f"{WMO_CODES.get(cur['weather_code'], 'Unknown conditions')}, "
        f"{cur['temperature_2m']}°C (feels like {cur['apparent_temperature']}°C), "
        f"humidity {cur['relative_humidity_2m']}%, "
        f"wind {cur['wind_speed_10m']} km/h, "
        f"precipitation {cur['precipitation']} mm. "
        f"Data: Open-Meteo.com"
    )


def _resolve_symbol(query: str) -> str | None:
    """Turn a company name into a Yahoo ticker using Yahoo's search."""
    try:
        quotes = yf.Search(query, max_results=8).quotes
    except Exception:
        return None
    equities = [q for q in quotes if q.get("quoteType") == "EQUITY"]
    if not equities:
        return None
    # Prefer an NSE listing if one exists, otherwise take Yahoo's top match
    for q in equities:
        if q["symbol"].endswith(".NS"):
            return q["symbol"]
    return equities[0]["symbol"]


def _last_price(symbol: str):
    """Return (price, previous_close, currency) or None if no data."""
    t = yf.Ticker(symbol)
    try:
        fi = t.fast_info
        price, prev, cur = fi.last_price, fi.previous_close, fi.currency
        if price:
            return price, prev, cur
    except Exception:
        pass
    hist = t.history(period="5d")
    if hist.empty:
        return None
    prev = hist["Close"].iloc[-2] if len(hist) > 1 else None
    return hist["Close"].iloc[-1], prev, None


@tool
def get_stock_price(company: str) -> str:
    """Get the latest stock price for a company.

    Args:
        company: A Yahoo Finance ticker if you know it (Indian NSE stocks end
            with .NS, e.g. "INFY.NS", "HAL.NS", "RELIANCE.NS"; US stocks like
            "AAPL", "MSFT"), otherwise the company name, e.g. "Infosys".
    """
    candidates = [company.strip().upper()]
    resolved = _resolve_symbol(company)
    if resolved and resolved not in candidates:
        candidates.append(resolved)

    for symbol in candidates:
        try:
            data = _last_price(symbol)
        except Exception:
            data = None
        if not data:
            continue
        price, prev, currency = data
        text = f"{symbol}: {price:,.2f} {currency or ''}".strip()
        if prev:
            change = price - prev
            text += f" ({change:+,.2f}, {change / prev * 100:+.2f}% vs previous close {prev:,.2f})"
        return text + ". Source: Yahoo Finance (may be delayed ~15 min)."

    return f"Could not find a stock price for '{company}'. Try the exact ticker, e.g. 'INFY.NS'."


if __name__ == "__main__":
    # Quick manual test:  python tools.py
    print(get_weather.invoke({"city": "Bengaluru"}))
    print(get_stock_price.invoke({"company": "Infosys"}))
    print(get_stock_price.invoke({"company": "AAPL"}))
