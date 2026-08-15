import asyncio
import math
import os
from datetime import datetime
from numbers import Real
from typing import Annotated
from typing import Any

import yfinance as yf
from loguru import logger
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent
from mcp.types import ToolAnnotations
from pydantic import Field
from yfinance.const import SECTOR_INDUSTY_MAPPING
from yfinance.exceptions import YFDataException
from yfinance.exceptions import YFInvalidPeriodError
from yfinance.exceptions import YFPricesMissingError
from yfinance.exceptions import YFRateLimitError
from yfinance.exceptions import YFTzMissingError

from yfmcp.chart import generate_chart
from yfmcp.screener import build_screener_query
from yfmcp.types import ChartType
from yfmcp.types import Interval
from yfmcp.types import OptionChainType
from yfmcp.types import Period
from yfmcp.types import ScreenerQueryType
from yfmcp.types import SearchType
from yfmcp.types import Sector
from yfmcp.types import TopType
from yfmcp.utils import create_error_response
from yfmcp.utils import dump_json

# https://github.com/jlowin/fastmcp/issues/81#issuecomment-2714245145
mcp = FastMCP(
    "yfinance_mcp",
    log_level="ERROR",
    host="0.0.0.0",
    port=int(os.environ.get("PORT", "10000")),
)


_RETRYABLE_YFINANCE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
    YFRateLimitError,
)

_ANALYST_ESTIMATE_SECTIONS: dict[str, tuple[str, str | None, bool]] = {
    "recommendations": ("recommendations", None, False),
    "earnings_estimate": ("earnings_estimate", "period", False),
    "revenue_estimate": ("revenue_estimate", "period", False),
    "eps_trend": ("eps_trend", "period", False),
    "eps_revisions": ("eps_revisions", "period", False),
    "earnings_history": ("earnings_history", "date", True),
    "growth_estimates": ("growth_estimates", "period", False),
}

_FUND_DATA_SECTIONS = {
    "description",
    "fund_overview",
    "fund_operations",
    "asset_classes",
    "top_holdings",
    "equity_holdings",
    "bond_holdings",
    "bond_ratings",
    "sector_weightings",
}


def _is_retryable_yfinance_error(exc: BaseException) -> bool:
    return isinstance(exc, _RETRYABLE_YFINANCE_EXCEPTIONS)


def _is_rate_limit_error(exc: BaseException) -> bool:
    return isinstance(exc, YFRateLimitError)


def _create_retryable_error_response(action: str, exc: BaseException, details: dict[str, Any]) -> str:
    if _is_rate_limit_error(exc):
        message = f"Rate limit reached while {action}. Try again later."
    else:
        message = f"Temporary network issue while {action}. Try again later."

    return create_error_response(message, error_code="NETWORK_ERROR", details={**details, "exception": str(exc)})


def _price_history_details(
    symbol: str,
    period: Period,
    interval: Interval,
    prepost: bool,
    exc: BaseException | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {"symbol": symbol, "period": period, "interval": interval, "prepost": prepost}
    if exc is not None:
        details["exception"] = str(exc)
    return details


def _create_price_history_no_data_response(
    symbol: str,
    period: Period,
    interval: Interval,
    prepost: bool,
    exc: BaseException | None = None,
) -> str:
    return create_error_response(
        f"No price data available for '{symbol}' with period='{period}' and interval='{interval}'. "
        "Common issues: (1) Invalid symbol, (2) Incompatible period/interval combination "
        "(e.g., '1m' interval requires '1d' or '5d' period), (3) Market holidays or insufficient history. "
        "Try a longer period or daily interval.",
        error_code="NO_DATA",
        details=_price_history_details(symbol, period, interval, prepost, exc),
    )


def _create_price_history_api_error_response(
    symbol: str,
    period: Period,
    interval: Interval,
    prepost: bool,
    exc: BaseException,
) -> str:
    return create_error_response(
        f"Failed to fetch price history for '{symbol}'. "
        "Verify the symbol is correct and the period/interval combination is valid.",
        error_code="API_ERROR",
        details=_price_history_details(symbol, period, interval, prepost, exc),
    )


def _is_price_history_rate_limit_message(message: str) -> bool:
    normalized = message.lower()
    return any(
        indicator in normalized
        for indicator in (
            "too many requests",
            "rate limit",
            "rate-limit",
            "ratelimit",
            "status_code = 429",
            "status_code=429",
        )
    )


def _is_price_history_no_data_prices_missing_error(exc: YFPricesMissingError) -> bool:
    debug_info = str(getattr(exc, "debug_info", "") or "").lower()

    if "yahoo status_code" in debug_info:
        return False

    if "yahoo error" in debug_info:
        return any(
            indicator in debug_info
            for indicator in (
                "no data found",
                "symbol may be delisted",
                "possibly delisted",
            )
        )

    return True


def _create_price_history_prices_missing_error_response(
    symbol: str,
    period: Period,
    interval: Interval,
    prepost: bool,
    exc: YFPricesMissingError,
) -> str:
    rate_limit_context = f"{getattr(exc, 'debug_info', '')} {getattr(exc, 'rationale', '')}"
    if _is_price_history_rate_limit_message(rate_limit_context):
        return create_error_response(
            f"Rate limit reached while fetching price history for '{symbol}'. Try again later.",
            error_code="NETWORK_ERROR",
            details=_price_history_details(symbol, period, interval, prepost, exc),
        )

    if _is_price_history_no_data_prices_missing_error(exc):
        return _create_price_history_no_data_response(symbol, period, interval, prepost, exc)

    return _create_price_history_api_error_response(symbol, period, interval, prepost, exc)


def _select_retryable_exception(exceptions: list[Exception]) -> BaseException:
    rate_limit_exception = next((exc for exc in exceptions if _is_rate_limit_error(exc)), None)
    return rate_limit_exception or exceptions[0]


def _normalize_json_value(value: Any) -> Any:
    """Replace non-finite numeric values recursively so responses remain strict JSON."""
    if isinstance(value, dict):
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, Real) and not math.isfinite(float(value)):
        return None
    return value


def _dataframe_records(
    frame: Any,
    max_rows: int,
    *,
    include_index: bool,
    index_name: str | None = None,
    sort_descending: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = frame.copy()
    if sort_descending:
        prepared = prepared.sort_index(ascending=False)
    if include_index:
        if index_name is not None:
            prepared.index.name = index_name
        prepared = prepared.reset_index()

    total_rows = len(prepared)
    limited = prepared if max_rows == 0 else prepared.head(max_rows)
    limited = limited.astype(object).where(limited.notna(), None)
    records = _normalize_json_value(limited.to_dict(orient="records"))
    return records, {
        "total_rows": total_rows,
        "returned_rows": len(records),
        "truncated": len(records) < total_rows,
    }


def _serialize_fund_section(value: Any, max_rows: int) -> tuple[Any, dict[str, Any] | None] | None:
    if value is None or (hasattr(value, "empty") and value.empty):
        return None
    if isinstance(value, (dict, list, str)) and not value:
        return None
    if hasattr(value, "columns") and hasattr(value, "index"):
        records, metadata = _dataframe_records(value, max_rows, include_index=True)
        return records, metadata
    return _normalize_json_value(value), None


async def _fetch_fund_sections(
    funds_data: Any,
    selected_sections: list[str],
    max_rows: int,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[tuple[str, Exception]]]:
    response: dict[str, Any] = {}
    section_metadata: dict[str, Any] = {}
    unavailable_sections: list[str] = []
    fetch_errors: list[tuple[str, Exception]] = []

    for section in selected_sections:
        try:
            value = await asyncio.to_thread(lambda section=section: getattr(funds_data, section))
            serialized = _serialize_fund_section(value, max_rows)
            if serialized is None:
                unavailable_sections.append(section)
                continue
            response[section], metadata = serialized
            if metadata is not None:
                section_metadata[section] = metadata
        except YFDataException:
            unavailable_sections.append(section)
        except Exception as exc:
            fetch_errors.append((section, exc))

    return response, section_metadata, unavailable_sections, fetch_errors


def _validate_sections(
    sections: list[str] | None,
    valid_sections: set[str],
    *,
    max_rows: int,
) -> tuple[list[str] | None, str | None]:
    if max_rows < 0:
        return None, create_error_response(
            "max_rows must be greater than or equal to 0.",
            error_code="INVALID_PARAMS",
            details={"max_rows": max_rows},
        )
    if sections is not None and not sections:
        return None, create_error_response(
            "sections must include at least one section when provided.",
            error_code="INVALID_PARAMS",
            details={"sections": sections, "valid_sections": sorted(valid_sections)},
        )

    selected = sorted(valid_sections) if sections is None else list(dict.fromkeys(sections))
    invalid = sorted(set(selected).difference(valid_sections))
    if invalid:
        return None, create_error_response(
            "One or more requested sections are invalid.",
            error_code="INVALID_PARAMS",
            details={"invalid_sections": invalid, "valid_sections": sorted(valid_sections)},
        )
    return selected, None


def _section_fetch_error_response(
    symbol: str,
    subject: str,
    selected_sections: list[str],
    fetch_errors: list[tuple[str, Exception]],
) -> str:
    retryable = [exc for _, exc in fetch_errors if _is_retryable_yfinance_error(exc)]
    failed_sections = [section for section, _ in fetch_errors]
    details: dict[str, Any] = {
        "symbol": symbol,
        "requested_sections": selected_sections,
        "failed_sections": failed_sections,
    }
    if retryable:
        return _create_retryable_error_response(
            f"fetching {subject} for '{symbol}'",
            _select_retryable_exception(retryable),
            details,
        )
    details["exception"] = str(fetch_errors[0][1])
    return create_error_response(
        f"Failed to fetch {subject} for '{symbol}'.",
        error_code="API_ERROR",
        details=details,
    )


def _create_option_dates_fetch_error(symbol: str, exc: Exception, api_message: str) -> str:
    if _is_retryable_yfinance_error(exc):
        return _create_retryable_error_response(f"fetching option dates for '{symbol}'", exc, {"symbol": symbol})

    return create_error_response(
        api_message,
        error_code="API_ERROR",
        details={"symbol": symbol, "exception": str(exc)},
    )


def _create_option_chain_fetch_error(
    symbol: str,
    dates_to_fetch: list[str],
    fetch_errors: list[tuple[str, Exception]],
) -> str:
    failed_dates = [date for date, _ in fetch_errors]

    if len(dates_to_fetch) == 1:
        failed_date, exc = fetch_errors[0]
        if _is_retryable_yfinance_error(exc):
            return _create_retryable_error_response(
                f"fetching option chain for '{symbol}' on '{failed_date}'",
                exc,
                {"symbol": symbol, "expiration_date": failed_date},
            )

        return create_error_response(
            f"Failed to fetch option chain for '{symbol}' on '{failed_date}'.",
            error_code="API_ERROR",
            details={"symbol": symbol, "expiration_date": failed_date, "exception": str(exc)},
        )

    retryable_exceptions = [exc for _, exc in fetch_errors if _is_retryable_yfinance_error(exc)]

    if retryable_exceptions:
        return _create_retryable_error_response(
            f"fetching option chain for '{symbol}'",
            _select_retryable_exception(retryable_exceptions),
            {
                "symbol": symbol,
                "dates_requested": dates_to_fetch,
                "failed_dates": failed_dates,
            },
        )

    representative_exception = fetch_errors[0][1]
    return create_error_response(
        f"Failed to fetch option chain for '{symbol}' for all requested dates.",
        error_code="API_ERROR",
        details={
            "symbol": symbol,
            "dates_requested": dates_to_fetch,
            "failed_dates": failed_dates,
            "exception": str(representative_exception),
        },
    )


@mcp.tool(
    name="yfinance_get_ticker_info",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_ticker_info(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
) -> str:
    """Retrieve comprehensive stock data including company information, financials, trading metrics and governance.

    Returns JSON object with fields including:
    - Company: symbol, longName, sector, industry, longBusinessSummary, website, city, country
    - Price: currentPrice, previousClose, open, dayHigh, dayLow, fiftyTwoWeekHigh, fiftyTwoWeekLow
    - Valuation: marketCap, enterpriseValue, trailingPE, forwardPE, priceToBook, pegRatio
    - Trading: volume, averageVolume, averageVolume10days, bid, ask, bidSize, askSize
    - Dividends: dividendRate, dividendYield, exDividendDate, payoutRatio
    - Financials: totalRevenue, revenueGrowth, earningsGrowth, profitMargins, operatingMargins
    - Performance: beta, fiftyDayAverage, twoHundredDayAverage, trailingEps, forwardEps

    Note: Available fields vary by security type. Timestamps are converted to readable dates.
    """
    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
        info = await asyncio.to_thread(lambda: ticker.info)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching ticker info for '{symbol}'", exc, {"symbol": symbol})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch ticker info for '{symbol}'. Verify the symbol is correct and try again.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    if not info:
        return create_error_response(
            f"No information available for symbol '{symbol}'. "
            "The symbol may be invalid or delisted. Try searching for the company "
            "name using the 'yfinance_search' tool to find the correct symbol.",
            error_code="INVALID_SYMBOL",
            details={"symbol": symbol},
        )

    # Convert timestamps to human-readable format when they look numeric.
    for key, value in list(info.items()):
        if not isinstance(key, str):
            continue

        if not isinstance(value, int | float):
            continue

        if key.lower().endswith(("date", "start", "end", "timestamp", "time", "quarter")):
            try:
                info[key] = datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as exc:
                logger.error("Unable to convert {}: {} to datetime: {}", key, value, exc)

    return dump_json(info)


@mcp.tool(
    name="yfinance_get_analyst_price_targets",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_analyst_price_targets(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
) -> str:
    """Fetch the current price and analyst consensus price targets for a stock.

    Returns a JSON object with the fields supplied by Yahoo Finance:
    - current: Current market price
    - low: Lowest analyst price target
    - high: Highest analyst price target
    - mean: Mean analyst price target
    - median: Median analyst price target

    Analyst coverage and available fields vary by symbol.
    """
    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
        targets = await asyncio.to_thread(lambda: ticker.analyst_price_targets)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(
            f"fetching analyst price targets for '{symbol}'",
            exc,
            {"symbol": symbol},
        )
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch analyst price targets for '{symbol}'. Verify the symbol is correct.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    if not targets:
        return create_error_response(
            f"No analyst price targets available for '{symbol}'.",
            error_code="NO_DATA",
            details={"symbol": symbol},
        )

    cleaned_targets: dict[str, Any] = {}
    for key, value in targets.items():
        if value is None:
            cleaned_targets[key] = None
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            cleaned_targets[key] = value
            continue
        cleaned_targets[key] = None if numeric != numeric else numeric

    return dump_json(cleaned_targets)


@mcp.tool(
    name="yfinance_get_analyst_estimates",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_analyst_estimates(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
    sections: Annotated[
        list[str] | None,
        Field(
            description=(
                "Optional analyst sections: recommendations, earnings_estimate, revenue_estimate, eps_trend, "
                "eps_revisions, earnings_history, and growth_estimates. Omit to return all sections."
            )
        ),
    ] = None,
    max_rows: Annotated[
        int,
        Field(description="Maximum rows per section. Use 0 to return all rows."),
    ] = 12,
) -> str:
    """Fetch consensus estimates, EPS/revenue trends, revisions, recommendations, and earnings history."""
    selected_sections, validation_error = _validate_sections(
        sections,
        set(_ANALYST_ESTIMATE_SECTIONS),
        max_rows=max_rows,
    )
    if validation_error is not None:
        return validation_error
    assert selected_sections is not None

    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(
            f"fetching analyst estimates for '{symbol}'",
            exc,
            {"symbol": symbol},
        )
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch analyst estimates for '{symbol}'.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    response: dict[str, Any] = {}
    section_metadata: dict[str, Any] = {}
    unavailable_sections: list[str] = []
    fetch_errors: list[tuple[str, Exception]] = []

    for section in selected_sections:
        attribute, index_name, sort_descending = _ANALYST_ESTIMATE_SECTIONS[section]
        try:
            frame = await asyncio.to_thread(lambda attribute=attribute: getattr(ticker, attribute))
            if frame is None or frame.empty:
                unavailable_sections.append(section)
                continue
            records, metadata = _dataframe_records(
                frame,
                max_rows,
                include_index=index_name is not None,
                index_name=index_name,
                sort_descending=sort_descending,
            )
            response[section] = records
            section_metadata[section] = metadata
        except Exception as exc:
            fetch_errors.append((section, exc))

    if not response:
        if fetch_errors:
            return _section_fetch_error_response(
                symbol,
                "analyst estimates",
                selected_sections,
                fetch_errors,
            )
        return create_error_response(
            f"No analyst estimates available for '{symbol}'.",
            error_code="NO_DATA",
            details={"symbol": symbol, "requested_sections": selected_sections},
        )

    response["_metadata"] = {
        "symbol": symbol,
        "max_rows": max_rows,
        "requested_sections": selected_sections,
        "sections": section_metadata,
        "unavailable_sections": unavailable_sections,
        "failed_sections": [section for section, _ in fetch_errors],
    }
    return dump_json(response)


@mcp.tool(
    name="yfinance_get_fund_data",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_fund_data(
    symbol: Annotated[str, Field(description="ETF or mutual fund ticker symbol (e.g., 'SPY', 'BND', 'VFIAX')")],
    sections: Annotated[
        list[str] | None,
        Field(
            description=(
                "Optional fund sections: description, fund_overview, fund_operations, asset_classes, "
                "top_holdings, equity_holdings, bond_holdings, bond_ratings, and sector_weightings. "
                "Omit to return all sections."
            )
        ),
    ] = None,
    max_rows: Annotated[
        int,
        Field(description="Maximum rows per tabular section. Use 0 to return all rows."),
    ] = 25,
) -> str:
    """Fetch ETF or mutual-fund composition, holdings, exposures, ratings, and operating details."""
    selected_sections, validation_error = _validate_sections(
        sections,
        _FUND_DATA_SECTIONS,
        max_rows=max_rows,
    )
    if validation_error is not None:
        return validation_error
    assert selected_sections is not None

    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
        funds_data = await asyncio.to_thread(lambda: ticker.funds_data)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching fund data for '{symbol}'", exc, {"symbol": symbol})
    except YFDataException as exc:
        return create_error_response(
            f"No fund data available for '{symbol}'.",
            error_code="NO_DATA",
            details={"symbol": symbol, "exception": str(exc)},
        )
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch fund data for '{symbol}'. Verify that the symbol is an ETF or mutual fund.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    response, section_metadata, unavailable_sections, fetch_errors = await _fetch_fund_sections(
        funds_data,
        selected_sections,
        max_rows,
    )

    if not response:
        if fetch_errors:
            return _section_fetch_error_response(symbol, "fund data", selected_sections, fetch_errors)
        return create_error_response(
            f"No fund data available for '{symbol}'.",
            error_code="NO_DATA",
            details={"symbol": symbol, "requested_sections": selected_sections},
        )

    response["_metadata"] = {
        "symbol": symbol,
        "max_rows": max_rows,
        "requested_sections": selected_sections,
        "sections": section_metadata,
        "unavailable_sections": unavailable_sections,
        "failed_sections": [section for section, _ in fetch_errors],
    }
    return dump_json(response)


@mcp.tool(
    name="yfinance_get_upgrades_downgrades",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_upgrades_downgrades(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
    max_rows: Annotated[
        int,
        Field(description="Maximum analyst actions to return, newest first. Use 0 to return all rows."),
    ] = 25,
) -> str:
    """Fetch analyst upgrades, downgrades, initiations, and price target changes.

    Returns analyst actions newest first. Fields supplied by Yahoo Finance can include:
    - GradeDate: Date and time of the analyst action
    - Firm: Analyst firm name
    - ToGrade and FromGrade: New and previous ratings
    - Action: Rating action, such as upgrade, downgrade, initiation, or reiteration
    - priceTargetAction: Price target action, such as Raises, Lowers, or Maintains
    - currentPriceTarget and priorPriceTarget: New and previous price targets

    Available fields vary by symbol and analyst action.
    """
    if max_rows < 0:
        return create_error_response(
            "max_rows must be greater than or equal to 0.",
            error_code="INVALID_PARAMS",
            details={"max_rows": max_rows},
        )

    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
        history = await asyncio.to_thread(lambda: ticker.upgrades_downgrades)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(
            f"fetching upgrades and downgrades for '{symbol}'",
            exc,
            {"symbol": symbol},
        )
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch upgrades and downgrades for '{symbol}'. Verify the symbol is correct.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    if history is None or history.empty:
        return create_error_response(
            f"No upgrades or downgrades available for '{symbol}'.",
            error_code="NO_DATA",
            details={"symbol": symbol},
        )

    history = history.sort_index(ascending=False).reset_index()
    total_rows = len(history)
    limited_history = history if max_rows == 0 else history.head(max_rows)
    limited_history = limited_history.astype(object).where(limited_history.notna(), None)
    records = limited_history.to_dict(orient="records")

    return dump_json(
        {
            "upgrades_downgrades": records,
            "_metadata": {
                "symbol": symbol,
                "max_rows": max_rows,
                "total_rows": total_rows,
                "returned_rows": len(records),
                "truncated": len(records) < total_rows,
            },
        }
    )


@mcp.tool(
    name="yfinance_get_ticker_news",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_ticker_news(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
) -> str:
    """Fetch recent news articles and press releases for a specific stock.

    Returns JSON array where each news item has:
    - id: Unique article identifier
    - content: Object containing:
        - title: Article headline
        - summary: Brief article summary
        - pubDate: Publication date (ISO 8601 format)
        - provider: Object with displayName (e.g., "Yahoo Finance") and url
        - canonicalUrl: Object with article url, site, region, lang
        - thumbnail: Object with image URLs and resolutions
        - contentType: Type of content (e.g., "STORY", "VIDEO")

    Use this to track company announcements, market sentiment, and breaking news.
    """
    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
        news = await asyncio.to_thread(ticker.get_news)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching news for '{symbol}'", exc, {"symbol": symbol})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch news for '{symbol}'. Verify the symbol is correct.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    if not news:
        return create_error_response(
            f"No news articles available for '{symbol}'. "
            "This may indicate an invalid symbol or no recent news coverage.",
            error_code="NO_DATA",
            details={"symbol": symbol},
        )

    return dump_json(news)


@mcp.tool(
    name="yfinance_search",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def search(
    query: Annotated[str, Field(description="Search query - company name, ticker symbol, or keywords")],
    search_type: Annotated[
        SearchType,
        Field(
            description="Filter results: 'all' (quotes + news), 'quotes' (stocks/ETFs only), or 'news' (articles only)"
        ),
    ],
) -> str:
    """Search Yahoo Finance for stocks, ETFs, and news articles.

    Returns JSON with search results based on search_type:

    - 'quotes': Array of securities with:
        - symbol: Ticker symbol
        - shortname/longname: Company name
        - quoteType: Security type (EQUITY, ETF, MUTUALFUND, etc.)
        - exchange: Exchange code
        - sector: Business sector
        - industry: Industry classification
        - score: Search relevance score

    - 'news': Array of articles with:
        - uuid: Article identifier
        - title: Headline
        - publisher: News source
        - link: Article URL
        - providerPublishTime: Unix timestamp
        - relatedTickers: Array of related symbols
        - thumbnail: Image URLs

    - 'all': Object with both 'quotes' and 'news' arrays

    Use this to find ticker symbols, discover related securities, or search financial news.
    """
    try:
        s = await asyncio.to_thread(yf.Search, query)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"searching for '{query}'", exc, {"query": query})
    except Exception as exc:
        return create_error_response(
            f"Search failed for '{query}'. Try simplifying your query or using different keywords.",
            error_code="API_ERROR",
            details={"query": query, "exception": str(exc)},
        )

    match search_type.lower():
        case "all":
            return dump_json(s.all)
        case "quotes":
            return dump_json(s.quotes)
        case "news":
            return dump_json(s.news)
        case _:
            return create_error_response(
                f"Invalid search_type '{search_type}'. Valid options: 'all', 'quotes', 'news'.",
                error_code="INVALID_PARAMS",
                details={"search_type": search_type, "valid_options": ["all", "quotes", "news"]},
            )


@mcp.tool(
    name="yfinance_screen",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def screen(
    query: Annotated[
        str | dict[str, Any],
        Field(
            description=(
                "Screener query. For query_type='predefined': string key like 'day_gainers'. "
                "For query_type='equity', 'fund', or 'etf': query tree object with {operator, operands} nodes."
            )
        ),
    ],
    query_type: Annotated[
        ScreenerQueryType,
        Field(description="Query mode: 'predefined', 'equity', 'fund', or 'etf'."),
    ] = "predefined",
    offset: Annotated[int | None, Field(description="Result offset.", ge=0)] = None,
    size: Annotated[
        int | None,
        Field(description="Rows to return for custom queries. Yahoo maximum is 250.", ge=1, le=250),
    ] = None,
    count: Annotated[
        int | None,
        Field(description="Rows to return for predefined queries. Yahoo maximum is 250.", ge=1, le=250),
    ] = None,
    sort_field: Annotated[str | None, Field(description="Sort field, for example 'percentchange'.")] = None,
    sort_asc: Annotated[bool | None, Field(description="Sort ascending if true, descending if false.")] = None,
    user_id: Annotated[str | None, Field(description="Optional Yahoo user id.")] = None,
    user_id_type: Annotated[str | None, Field(description="Optional Yahoo user id type, commonly 'guid'.")] = None,
) -> str:
    """Run a Yahoo Finance screener query.

    Supports predefined Yahoo screener keys and custom equity, mutual-fund, or ETF query trees.
    """
    try:
        if query_type == "predefined" and size is not None:
            return create_error_response(
                "For query_type='predefined', use count instead of size.",
                error_code="INVALID_PARAMS",
                details={"query_type": query_type, "invalid_parameter": "size", "expected_parameter": "count"},
            )
        if query_type in {"equity", "fund", "etf"} and count is not None:
            return create_error_response(
                "For query_type='equity', 'fund', or 'etf', use size instead of count.",
                error_code="INVALID_PARAMS",
                details={"query_type": query_type, "invalid_parameter": "count", "expected_parameter": "size"},
            )

        if query_type == "predefined":
            if not isinstance(query, str):
                return create_error_response(
                    "For query_type='predefined', query must be a string screener key.",
                    error_code="INVALID_PARAMS",
                    details={"query_type": query_type, "expected_query_type": "string"},
                )

            predefined = getattr(yf, "PREDEFINED_SCREENER_QUERIES", {})
            if query not in predefined:
                return create_error_response(
                    f"Unknown predefined screener '{query}'.",
                    error_code="INVALID_PARAMS",
                    details={
                        "query": query,
                        "query_type": query_type,
                        "valid_predefined_queries": sorted(predefined.keys()),
                    },
                )

            resolved_query: str | Any = query
        else:
            if not isinstance(query, dict):
                return create_error_response(
                    "For query_type='equity', 'fund', or 'etf', query must be an object with "
                    "'operator' and 'operands'.",
                    error_code="INVALID_PARAMS",
                    details={"query_type": query_type, "expected_query_type": "object"},
                )

            resolved_query = build_screener_query(query_type=query_type, query=query)

        upstream_size = count if query_type == "predefined" else size
        result = await asyncio.to_thread(
            yf.screen,
            resolved_query,
            offset=offset,
            size=upstream_size,
            count=None,
            sortField=sort_field,
            sortAsc=sort_asc,
            userId=user_id,
            userIdType=user_id_type,
        )
    except (TypeError, ValueError) as exc:
        return create_error_response(
            "Invalid screener query. Check operators, operands, and field values for the selected query_type.",
            error_code="INVALID_PARAMS",
            details={"query_type": query_type, "exception": str(exc)},
        )
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response("running screener query", exc, {"query_type": query_type})
    except Exception as exc:
        return create_error_response(
            "Failed to run screener query.",
            error_code="API_ERROR",
            details={"query_type": query_type, "exception": str(exc)},
        )

    return dump_json(result)


@mcp.tool(
    name="yfinance_screen_gappers",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def screen_gappers(
    min_percent_change: Annotated[
        float,
        Field(description="Minimum percent change from prior close, for example 3.0 for +3%.", ge=0),
    ] = 3.0,
    min_price: Annotated[
        float,
        Field(description="Minimum current intraday price.", ge=0),
    ] = 5.0,
    min_volume: Annotated[
        int,
        Field(description="Minimum intraday trading volume.", ge=0),
    ] = 500000,
    min_market_cap: Annotated[
        int,
        Field(description="Minimum intraday market cap in USD.", ge=0),
    ] = 2000000000,
    region: Annotated[
        str,
        Field(description="Yahoo screener region code, for example 'us'."),
    ] = "us",
    size: Annotated[
        int,
        Field(description="Rows to return. Yahoo maximum is 250.", ge=1, le=250),
    ] = 50,
    offset: Annotated[
        int,
        Field(description="Result offset for pagination.", ge=0),
    ] = 0,
    sort_asc: Annotated[
        bool,
        Field(description="Sort by percentchange ascending if true, descending if false."),
    ] = False,
) -> str:
    """Run a custom equity screener tuned for opening-session stock gappers."""
    query = {
        "operator": "and",
        "operands": [
            {"operator": "gte", "operands": ["percentchange", min_percent_change]},
            {"operator": "eq", "operands": ["region", region]},
            {"operator": "gte", "operands": ["intradaymarketcap", min_market_cap]},
            {"operator": "gte", "operands": ["intradayprice", min_price]},
            {"operator": "gte", "operands": ["dayvolume", min_volume]},
        ],
    }

    try:
        resolved_query = build_screener_query(query_type="equity", query=query)
        result = await asyncio.to_thread(
            yf.screen,
            resolved_query,
            offset=offset,
            size=size,
            sortField="percentchange",
            sortAsc=sort_asc,
        )
    except (TypeError, ValueError) as exc:
        return create_error_response(
            "Invalid gappers screener parameters.",
            error_code="INVALID_PARAMS",
            details={"exception": str(exc)},
        )
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response("running gappers screener", exc, {})
    except Exception as exc:
        return create_error_response(
            "Failed to run gappers screener.",
            error_code="API_ERROR",
            details={"exception": str(exc)},
        )

    return dump_json(result)


async def get_top_etfs(
    sector: Annotated[Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare')")],
    top_n: Annotated[int, Field(description="Number of top ETFs to retrieve", ge=1)],
) -> str:
    """Get the most popular ETFs for a specific sector.

    Returns JSON array where each ETF has:
    - symbol: ETF ticker symbol
    - name: Full ETF name
    """
    try:
        s = await asyncio.to_thread(yf.Sector, _sector_key(sector))
        etfs = await asyncio.to_thread(lambda: s.top_etfs)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching top ETFs for '{sector}'", exc, {"sector": sector})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch top ETFs for '{sector}'. Verify the sector name is valid.",
            error_code="API_ERROR",
            details={"sector": sector, "exception": str(exc)},
        )

    if not etfs:
        return create_error_response(
            f"No ETF data available for sector '{sector}'.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    result = [{"symbol": symbol, "name": name} for symbol, name in list(etfs.items())[:top_n]]
    return dump_json(result)


async def get_top_mutual_funds(
    sector: Annotated[Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare')")],
    top_n: Annotated[int, Field(description="Number of top mutual funds to retrieve", ge=1)],
) -> str:
    """Get the most popular mutual funds for a specific sector.

    Returns JSON array where each mutual fund has:
    - symbol: Fund ticker symbol
    - name: Full fund name
    """
    try:
        s = await asyncio.to_thread(yf.Sector, _sector_key(sector))
        funds = await asyncio.to_thread(lambda: s.top_mutual_funds)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(
            f"fetching top mutual funds for '{sector}'",
            exc,
            {"sector": sector},
        )
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch top mutual funds for '{sector}'. Verify the sector name is valid.",
            error_code="API_ERROR",
            details={"sector": sector, "exception": str(exc)},
        )

    if not funds:
        return create_error_response(
            f"No mutual fund data available for sector '{sector}'.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    result = [{"symbol": symbol, "name": name} for symbol, name in list(funds.items())[:top_n]]
    return dump_json(result)


async def get_top_companies(
    sector: Annotated[Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare')")],
    top_n: Annotated[int, Field(description="Number of top companies to retrieve", ge=1)],
) -> str:
    """Get top companies in a sector by market capitalization.

    Returns JSON array with company data from Yahoo Finance sector data.
    Typically includes company identifiers, market metrics, and analyst information.
    """
    try:
        s = await asyncio.to_thread(yf.Sector, _sector_key(sector))
        df = await asyncio.to_thread(lambda: s.top_companies)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching top companies for '{sector}'", exc, {"sector": sector})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch top companies for '{sector}'. Verify the sector name is valid.",
            error_code="API_ERROR",
            details={"sector": sector, "exception": str(exc)},
        )

    if df is None or df.empty:
        return create_error_response(
            f"No company data available for '{sector}'. This sector may not have enough listed companies.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    return dump_json(df.head(top_n).to_dict(orient="records"))


def _sector_key(name: str) -> str:
    """Convert human-readable sector name to Yahoo Finance API key format."""
    return name.lower().replace(" ", "-")


def _industry_key(name: str) -> str:
    """Convert human-readable industry name to Yahoo Finance API key format.

    SECTOR_INDUSTY_MAPPING uses em dashes (—) and title case,
    but the API expects lowercase with regular hyphens.
    """
    return name.lower().replace("& ", "").replace("- ", "").replace(", ", " ").replace("—", "-").replace(" ", "-")


async def get_top_growth_companies(
    sector: Annotated[Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare')")],
    top_n: Annotated[int, Field(description="Number of top growth companies per industry", ge=1)],
) -> str:
    """Get fastest-growing companies organized by industry within a sector.

    Returns JSON array grouped by industry. Each industry entry contains company data
    with growth-related metrics from Yahoo Finance.

    Results are organized by industry to show growth leaders across the sector.
    """
    try:
        industries = SECTOR_INDUSTY_MAPPING[sector]
    except KeyError:
        return create_error_response(
            f"Unknown sector '{sector}'. Valid sectors: {', '.join(SECTOR_INDUSTY_MAPPING.keys())}",
            error_code="INVALID_PARAMS",
            details={"sector": sector, "valid_sectors": list(SECTOR_INDUSTY_MAPPING.keys())},
        )

    results = []
    for industry_name in industries:
        try:
            industry = await asyncio.to_thread(yf.Industry, _industry_key(industry_name))
        except Exception as exc:
            logger.warning("Failed to load industry {}: {}", industry_name, exc)
            continue

        df = await asyncio.to_thread(lambda i=industry: i.top_growth_companies)
        if df is None or df.empty:
            continue

        results.append(
            {
                "industry": industry_name,
                "top_growth_companies": df.head(top_n).to_dict(orient="records"),
            }
        )

    if not results:
        return create_error_response(
            f"No growth company data available for '{sector}'. Try a different sector or check back later.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    return dump_json(results)


async def get_top_performing_companies(
    sector: Annotated[Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare')")],
    top_n: Annotated[int, Field(description="Number of top performing companies per industry", ge=1)],
) -> str:
    """Get best-performing companies by stock price performance, organized by industry.

    Returns JSON array grouped by industry. Each industry entry contains company data
    with performance-related metrics from Yahoo Finance.

    Results are organized by industry to show top performers across the sector.
    """
    try:
        industries = SECTOR_INDUSTY_MAPPING[sector]
    except KeyError:
        return create_error_response(
            f"Unknown sector '{sector}'. Valid sectors: {', '.join(SECTOR_INDUSTY_MAPPING.keys())}",
            error_code="INVALID_PARAMS",
            details={"sector": sector, "valid_sectors": list(SECTOR_INDUSTY_MAPPING.keys())},
        )

    results = []
    for industry_name in industries:
        try:
            industry = await asyncio.to_thread(yf.Industry, _industry_key(industry_name))
        except Exception as exc:
            logger.warning("Failed to load industry {}: {}", industry_name, exc)
            continue

        df = await asyncio.to_thread(lambda i=industry: i.top_performing_companies)
        if df is None or df.empty:
            continue

        results.append(
            {
                "industry": industry_name,
                "top_performing_companies": df.head(top_n).to_dict(orient="records"),
            }
        )

    if not results:
        return create_error_response(
            f"No performance data available for '{sector}'. Try a different sector or check back later.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    return dump_json(results)


@mcp.tool(
    name="yfinance_get_top",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_top(
    sector: Annotated[
        Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare', 'Financial Services')")
    ],
    top_type: Annotated[
        TopType,
        Field(
            description=(
                "Type of entities to retrieve: "
                "'top_etfs' (sector ETFs), "
                "'top_mutual_funds' (sector mutual funds), "
                "'top_companies' (largest by market cap), "
                "'top_growth_companies' (fastest revenue/earnings growth), "
                "'top_performing_companies' (best stock price performance)"
            )
        ),
    ],
    top_n: Annotated[
        int,
        Field(
            description="Number of top entities to retrieve per category/industry",
            ge=1,
            le=100,
        ),
    ] = 10,
) -> str:
    """Get top-ranked financial entities within a sector.

    This unified tool provides access to various rankings:
    - ETFs and mutual funds focused on the sector
    - Largest companies by market capitalization
    - Fastest-growing companies by revenue/earnings
    - Best-performing stocks by price appreciation

    Returns JSON data with relevant metrics for each entity type.
    """
    match top_type:
        case "top_etfs":
            return await get_top_etfs(sector, top_n)
        case "top_mutual_funds":
            return await get_top_mutual_funds(sector, top_n)
        case "top_companies":
            return await get_top_companies(sector, top_n)
        case "top_growth_companies":
            return await get_top_growth_companies(sector, top_n)
        case "top_performing_companies":
            return await get_top_performing_companies(sector, top_n)
        case _:
            return create_error_response(
                f"Invalid top_type '{top_type}'. "
                "Valid options: 'top_etfs', 'top_mutual_funds', 'top_companies', "
                "'top_growth_companies', 'top_performing_companies'.",
                error_code="INVALID_PARAMS",
                details={
                    "top_type": top_type,
                    "valid_options": [
                        "top_etfs",
                        "top_mutual_funds",
                        "top_companies",
                        "top_growth_companies",
                        "top_performing_companies",
                    ],
                },
            )


@mcp.tool(
    name="yfinance_get_price_history",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_price_history(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
    period: Annotated[
        Period,
        Field(
            description=(
                "Time range: '1d'/'5d' (days), '1mo'/'3mo'/'6mo' (months), "
                "'1y'/'2y'/'5y'/'10y' (years), 'ytd' (year-to-date), 'max' (all available data)"
            )
        ),
    ] = "1mo",
    interval: Annotated[
        Interval,
        Field(
            description=(
                "Data granularity: '1m'/'5m'/'15m'/'30m' (minutes), '1h' (hour), "
                "'1d'/'5d' (days), '1wk' (week), '1mo'/'3mo' (months). "
                "Short intervals require short periods (e.g., '1m' interval only works with '1d'/'5d' period)"
            )
        ),
    ] = "1d",
    chart_type: Annotated[
        ChartType | None,
        Field(
            description=(
                "Optional visualization: "
                "'price_volume' (candlestick chart with volume bars), "
                "'vwap' (Volume Weighted Average Price overlay), "
                "'volume_profile' (volume distribution by price level). "
                "Omit for tabular data"
            )
        ),
    ] = None,
    prepost: Annotated[
        bool,
        Field(description="Include pre-market and post-market data when available"),
    ] = False,
) -> str | ImageContent:
    """Fetch historical price data and optionally generate technical analysis charts.

    When chart_type is None, returns Markdown table with columns:
    - Date: Trading date (index)
    - Open: Opening price
    - High: Highest price
    - Low: Lowest price
    - Close: Closing price
    - Volume: Trading volume
    - Dividends: Dividend payments (if any)
    - Stock Splits: Split events (if any)

    When chart_type is specified, returns a chart image:
    - 'price_volume': Candlestick chart with volume bars
    - 'vwap': Price with Volume Weighted Average Price overlay
    - 'volume_profile': Volume distribution by price level

    Set prepost=True to include pre-market and post-market data when available.

    Note: Not all period/interval combinations are valid. Minute intervals (1m, 5m, etc.)
    only work with short periods (1d, 5d).
    """
    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
        df = await asyncio.to_thread(
            ticker.history,
            period=period,
            interval=interval,
            prepost=prepost,
            rounding=True,
            raise_errors=True,
        )
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(
            f"fetching price history for '{symbol}'",
            exc,
            _price_history_details(symbol, period, interval, prepost),
        )
    except (YFTzMissingError, YFInvalidPeriodError) as exc:
        return _create_price_history_no_data_response(symbol, period, interval, prepost, exc)
    except YFPricesMissingError as exc:
        return _create_price_history_prices_missing_error_response(symbol, period, interval, prepost, exc)
    except Exception as exc:
        return _create_price_history_api_error_response(symbol, period, interval, prepost, exc)

    if df.empty:
        return _create_price_history_no_data_response(symbol, period, interval, prepost)

    if chart_type is None:
        return df.to_markdown()

    return generate_chart(symbol=symbol, df=df, chart_type=chart_type)


@mcp.tool(
    name="yfinance_get_financials",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_financials(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
    frequency: Annotated[
        str,
        Field(
            description=(
                "Reporting frequency: 'annual' for yearly, 'quarterly' for quarterly, "
                "or 'ttm' for trailing twelve months"
            )
        ),
    ] = "annual",
) -> str:
    """Fetch financial statements (income statement, balance sheet, and cash flow) with historical data.

    Returns JSON with income statement, balance sheet, and cash flow data across reporting periods.

    Use the data to analyze trends, calculate ratios, or compare periods.
    """
    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching financials for '{symbol}'", exc, {"symbol": symbol})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch financials for '{symbol}'. Verify the symbol is correct.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    income_stmt = None
    balance_sheet = None
    cash_flow = None

    if frequency not in {"annual", "quarterly", "ttm"}:
        return create_error_response(
            f"Invalid frequency '{frequency}'. Valid options: 'annual', 'quarterly', 'ttm'.",
            error_code="INVALID_PARAMS",
            details={"frequency": frequency, "valid_options": ["annual", "quarterly", "ttm"]},
        )

    try:
        if frequency == "annual":
            income_stmt = await asyncio.to_thread(lambda: ticker.income_stmt)
            balance_sheet = await asyncio.to_thread(lambda: ticker.balance_sheet)
            cash_flow = await asyncio.to_thread(lambda: ticker.cashflow)
        elif frequency == "quarterly":
            income_stmt = await asyncio.to_thread(lambda: ticker.quarterly_income_stmt)
            balance_sheet = await asyncio.to_thread(lambda: ticker.quarterly_balance_sheet)
            cash_flow = await asyncio.to_thread(lambda: ticker.quarterly_cashflow)
        else:
            income_stmt = await asyncio.to_thread(lambda: ticker.ttm_income_stmt)
            balance_sheet = None  # TTM balance sheet not directly available
            cash_flow = None  # TTM cash flow not directly available

        result = _build_financials_response(income_stmt, balance_sheet, cash_flow)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(
            f"fetching financials for '{symbol}'",
            exc,
            {"symbol": symbol, "frequency": frequency},
        )
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch financials for '{symbol}'. Verify the symbol is correct.",
            error_code="API_ERROR",
            details={"symbol": symbol, "frequency": frequency, "exception": str(exc)},
        )
    if not result:
        return create_error_response(
            f"No financial data available for '{symbol}' with frequency='{frequency}'.",
            error_code="NO_DATA",
            details={"symbol": symbol, "frequency": frequency},
        )

    return dump_json(result)


def _build_financials_response(income_stmt, balance_sheet, cash_flow=None) -> dict:
    """Build financials response from income statement, balance sheet, and cash flow DataFrames."""
    result = {}

    if income_stmt is not None and not income_stmt.empty:
        income_fields = [
            "EBIT",
            "Net Income",
            "Tax Provision",
            "Pretax Income",
            "Interest Expense",
            "Total Revenue",
            "Operating Income",
            "EBITDA",
            "Normalized Income",
        ]
        available_income_fields = [f for f in income_fields if f in income_stmt.index]
        result["income_statement"] = {}
        for field in available_income_fields:
            result["income_statement"][field] = {
                str(col.date()): income_stmt.loc[field, col] for col in income_stmt.columns
            }

    if balance_sheet is not None and not balance_sheet.empty:
        balance_fields = [
            "Stockholders Equity",
            "Total Debt",
            "Cash And Cash Equivalents",
            "Invested Capital",
            "Net Debt",
            "Total Assets",
            "Total Liabilities Net Minority Interest",
            "Net Tangible Assets",
            "Tangible Book Value",
        ]
        available_balance_fields = [f for f in balance_fields if f in balance_sheet.index]
        result["balance_sheet"] = {}
        for field in available_balance_fields:
            result["balance_sheet"][field] = {
                str(col.date()): balance_sheet.loc[field, col] for col in balance_sheet.columns
            }

    if cash_flow is not None and not cash_flow.empty:
        cash_flow_fields = [
            "Operating Cash Flow",
            "Free Cash Flow",
            "Capital Expenditure",
            "Net Income From Continuing Operations",
            "Depreciation And Amortization",
            "Change In Working Capital",
            "Cash Dividends Paid",
        ]
        available_cash_flow_fields = [f for f in cash_flow_fields if f in cash_flow.index]
        result["cash_flow"] = {}
        for field in available_cash_flow_fields:
            result["cash_flow"][field] = {str(col.date()): cash_flow.loc[field, col] for col in cash_flow.columns}

    return result


async def _fetch_option_chain_for_date(
    ticker: yf.Ticker,
    date: str,
    option_type: OptionChainType,
) -> dict[str, Any]:
    """Fetch option chain for a single expiration date."""
    opt = await asyncio.to_thread(lambda d=date: ticker.option_chain(d))

    calls_df = opt.calls
    puts_df = opt.puts
    date_data: dict[str, Any] = {}

    if calls_df is not None and not calls_df.empty and option_type in {"all", "calls"}:
        calls_df = calls_df.copy()
        calls_df["optionType"] = "CALL"
        date_data["calls"] = calls_df.to_dict(orient="records")

    if puts_df is not None and not puts_df.empty and option_type in {"all", "puts"}:
        puts_df = puts_df.copy()
        puts_df["optionType"] = "PUT"
        date_data["puts"] = puts_df.to_dict(orient="records")

    return {date: date_data} if date_data else {}


@mcp.tool(
    name="yfinance_get_option_chain",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_option_chain(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
    expiration_date: Annotated[
        str | None,
        Field(
            description=(
                "Option expiration date in YYYY-MM-DD format. "
                "Use the 'yfinance_get_option_dates' tool to find available dates, "
                "or omit to fetch all available expiration dates."
            )
        ),
    ] = None,
    option_type: Annotated[
        OptionChainType,
        Field(description=("Which options to return: 'calls', 'puts', or 'all' (both calls and puts).")),
    ] = "all",
) -> str:
    """Fetch option chain data (calls and puts) for a stock with available strike prices.

    Returns JSON with calls and/or puts data for each expiration date.

    JSON fields include:
    - contractSymbol: Option contract identifier
    - strike: Strike price
    - lastPrice: Last traded price
    - bid/ask: Bid and ask prices
    - volume: Trading volume
    - openInterest: Open interest
    - impliedVolatility: Implied volatility (IV)
    - inTheMoney: Whether option is ITM
    - contractSize: Contract size (REGULAR)
    - currency: Currency (USD)

    Use this to analyze options pricing, IV surfaces, and strike levels.
    """
    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching options for '{symbol}'", exc, {"symbol": symbol})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch options for '{symbol}'. Verify the symbol is correct.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    try:
        available_dates = await asyncio.to_thread(lambda: ticker.options)
    except Exception as exc:
        return _create_option_dates_fetch_error(
            symbol,
            exc,
            f"Failed to fetch option dates for '{symbol}'. The symbol may not have options.",
        )

    if not available_dates:
        return create_error_response(
            f"No options available for symbol '{symbol}'. "
            "This symbol may not have listed options (e.g., ETFs, stocks without options).",
            error_code="NO_DATA",
            details={"symbol": symbol},
        )

    if expiration_date is not None and expiration_date not in available_dates:
        return create_error_response(
            f"Invalid expiration date '{expiration_date}' for '{symbol}'. Valid dates: {', '.join(available_dates)}",
            error_code="INVALID_PARAMS",
            details={
                "symbol": symbol,
                "expiration_date": expiration_date,
                "valid_dates": available_dates,
            },
        )

    dates_to_fetch = [expiration_date] if expiration_date else list(available_dates)
    result: dict[str, Any] = {}
    fetch_errors: list[tuple[str, Exception]] = []

    for date in dates_to_fetch:
        try:
            date_result = await _fetch_option_chain_for_date(ticker, date, option_type)
        except Exception as exc:
            logger.warning("Failed to fetch option chain for {} {}: {}", symbol, date, exc)
            fetch_errors.append((date, exc))
            continue
        result.update(date_result)

    if result:
        return dump_json(result)

    if fetch_errors:
        return _create_option_chain_fetch_error(symbol, dates_to_fetch, fetch_errors)

    return create_error_response(
        f"No option data retrieved for '{symbol}'.",
        error_code="NO_DATA",
        details={"symbol": symbol, "dates_requested": list(dates_to_fetch)},
    )


@mcp.tool(
    name="yfinance_get_option_dates",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_option_dates(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
) -> str:
    """Fetch available option expiration dates for a stock.

    Returns JSON array of expiration dates in YYYY-MM-DD format.

    Use these dates with the 'yfinance_get_option_chain' tool to fetch
    the options chain for a specific date.
    """
    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
        dates = await asyncio.to_thread(lambda: ticker.options)
    except Exception as exc:
        return _create_option_dates_fetch_error(
            symbol,
            exc,
            f"Failed to fetch option dates for '{symbol}'. Verify the symbol is correct.",
        )

    if not dates:
        return create_error_response(
            f"No options available for symbol '{symbol}'. "
            "This symbol may not have listed options (e.g., ETFs, stocks without options).",
            error_code="NO_DATA",
            details={"symbol": symbol},
        )

    return dump_json(dates)


async def _fetch_holder_section(
    symbol: str,
    ticker: yf.Ticker,
    attr_name: str,
    result_key: str,
    result: dict[str, Any],
    section_metadata: dict[str, dict[str, int | bool]],
    fetch_errors: list[Exception],
    max_rows: int,
) -> None:
    """Fetch a single holder data section, adding successful data to result and failures to fetch_errors."""
    try:
        df = await asyncio.to_thread(lambda t=ticker: getattr(t, attr_name))
    except Exception as exc:
        logger.warning("Failed to fetch {} for {}: {}", attr_name, symbol, exc)
        fetch_errors.append(exc)
        return
    if df is not None and not df.empty:
        if attr_name == "major_holders":
            df = df.reset_index()

        total_rows = len(df)
        limited_df = df if max_rows == 0 else df.head(max_rows)
        limited_records = limited_df.to_dict(orient="records")
        result[result_key] = limited_records
        section_metadata[result_key] = {
            "total_rows": total_rows,
            "returned_rows": len(limited_records),
            "truncated": len(limited_records) < total_rows,
        }


@mcp.tool(
    name="yfinance_get_holders",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_holders(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
    max_rows: Annotated[
        int,
        Field(description="Maximum rows returned per holder section. Use 0 to return all rows."),
    ] = 10,
) -> str:
    """Fetch major holders, institutional holders, mutual fund holders, and insider data.

    Returns JSON with:
    - major_holders: Aggregated breakdown including insider % held, institutional % held,
      institutional % float held, and institution count.
    - institutional_holders: List of institutional investors with shares held, date reported,
      value, and % change.
    - mutualfund_holders: List of mutual fund holders with same fields.
    - insider_transactions: Recent insider transactions including shares, value, transaction
      type, and date.
    - insider_purchases: Summary of insider buy/sell activity over the last 6 months.
    - insider_roster: List of known insiders by name and position.

    Use this to analyze ownership concentration, insider activity, and institutional interest.
    """
    if max_rows < 0:
        return create_error_response(
            "max_rows must be greater than or equal to 0.",
            error_code="INVALID_PARAMS",
            details={"max_rows": max_rows},
        )

    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching holders for '{symbol}'", exc, {"symbol": symbol})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch holders for '{symbol}'. Verify the symbol is correct.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    result: dict[str, Any] = {}
    section_metadata: dict[str, dict[str, int | bool]] = {}
    fetch_errors: list[Exception] = []
    await _fetch_holder_section(
        symbol, ticker, "major_holders", "major_holders", result, section_metadata, fetch_errors, max_rows
    )
    await _fetch_holder_section(
        symbol,
        ticker,
        "institutional_holders",
        "institutional_holders",
        result,
        section_metadata,
        fetch_errors,
        max_rows,
    )
    await _fetch_holder_section(
        symbol,
        ticker,
        "mutualfund_holders",
        "mutualfund_holders",
        result,
        section_metadata,
        fetch_errors,
        max_rows,
    )
    await _fetch_holder_section(
        symbol,
        ticker,
        "insider_transactions",
        "insider_transactions",
        result,
        section_metadata,
        fetch_errors,
        max_rows,
    )
    await _fetch_holder_section(
        symbol, ticker, "insider_purchases", "insider_purchases", result, section_metadata, fetch_errors, max_rows
    )
    await _fetch_holder_section(
        symbol,
        ticker,
        "insider_roster_holders",
        "insider_roster",
        result,
        section_metadata,
        fetch_errors,
        max_rows,
    )

    if not result:
        retryable_exceptions = [exc for exc in fetch_errors if _is_retryable_yfinance_error(exc)]
        if retryable_exceptions:
            return _create_retryable_error_response(
                f"fetching holders for '{symbol}'",
                _select_retryable_exception(retryable_exceptions),
                {"symbol": symbol},
            )

        return create_error_response(
            f"No holder data available for '{symbol}'. Verify the symbol is correct.",
            error_code="NO_DATA",
            details={"symbol": symbol},
        )

    result["_metadata"] = {"max_rows": max_rows, "sections": section_metadata}
    return dump_json(result)


@mcp.tool(
    name="yfinance_backtest",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def backtest(
    symbol: Annotated[
        str,
        Field(description="Yahoo Finance NSE symbol, e.g. SBIN.NS, RELIANCE.NS")
    ],
    period: Annotated[
        str,
        Field(description="Backtest period: 1y, 2y, 5y, 10y, max")
    ] = "5y",
    initial_capital: Annotated[
        float,
        Field(description="Starting capital", gt=0)
    ] = 100000,
) -> str:
    """
    FINAL LOCKED SWING / POSITIONAL STRATEGY

    ==============================================================
    STRATEGY PURPOSE
    ==============================================================

    Primary style:
        Swing + positional

    Holding period:
        2 to 15 trading days

    Main timeframe:
        Weekly trend + Daily setup

    Strategy is intentionally NOT intraday.

    ==============================================================

    HARD STRATEGY RULES
    ==============================================================

    WEEKLY STRUCTURE
        Weekly Close > Weekly EMA21
        Weekly EMA21 > Weekly EMA55

    DAILY STRUCTURE
        Close > EMA21 > EMA55 > EMA100 > EMA200

    MOMENTUM
        RSI 50 - 70
        MACD > Signal
        MACD Histogram > 0
        ADX >= 20

    VOLUME
        Current Volume >= 20-day Average Volume

    ENTRY
        Either:

        A) 20-day breakout

        OR

        B) Pullback + EMA21 reclaim:
           Previous Close <= Previous EMA21
           Current Close > EMA21
           Current Close > Previous High

    NO CHASING
        Close must not be more than 5% above EMA21.

    EXECUTION
        Signal confirmed at today's close.
        Entry = next trading day's OPEN.

    STOP LOSS
        1.5 x ATR(14)

    TARGET
        2R

    POSITION SIZING
        Maximum 1% of current equity risk per trade.

    EXIT
        1. ATR stop
        2. 2R target
        3. Daily close below EMA21
        4. Maximum 15 trading days
        5. End of test

    ==============================================================

    IMPORTANT
    ==============================================================

    This backtest uses equity OHLCV data.

    It does NOT fake:
        - F&O OI
        - CE/PE liquidity
        - institutional flow
        - sector heatmap
        - live option liquidity

    Those are scanner-stage filters.

    The backtest therefore tests the PRICE/ACTION core strategy
    honestly instead of inventing unavailable historical data.

    No look-ahead:
        Weekly confirmation uses only the previous completed
        weekly candle.

        Daily signal enters on the next day's OPEN.
    """

    try:
        import numpy as np
        import pandas as pd

        # ==========================================================
        # LOCKED STRATEGY CONFIGURATION
        # DO NOT CHANGE THESE FOR INDIVIDUAL TESTS
        # ==========================================================

        EMA_FAST = 21
        EMA_MID = 55
        EMA_LONG = 100
        EMA_TREND = 200

        RSI_PERIOD = 14
        RSI_MIN = 50.0
        RSI_MAX = 70.0

        MACD_FAST = 12
        MACD_SLOW = 26
        MACD_SIGNAL = 9

        ATR_PERIOD = 14
        ATR_MULTIPLIER = 1.5

        ADX_PERIOD = 14
        MIN_ADX = 20.0

        VOLUME_PERIOD = 20
        MIN_VOLUME_RATIO = 1.0

        BREAKOUT_PERIOD = 20

        MAX_EXTENSION_PCT = 5.0

        RISK_PER_TRADE_PCT = 1.0

        REWARD_RISK = 2.0

        MAX_HOLDING_DAYS = 15

        MIN_REQUIRED_BARS = 300

        # ==========================================================
        # DOWNLOAD DAILY DATA
        # ==========================================================

        df = await asyncio.to_thread(
            yf.download,
            symbol,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )

        if df is None or df.empty:
            return create_error_response(
                f"No historical data available for '{symbol}'.",
                error_code="NO_DATA",
                details={
                    "symbol": symbol,
                    "period": period,
                },
            )

        df = df.copy()

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            return create_error_response(
                "Required OHLCV columns are missing.",
                error_code="NO_DATA",
                details={
                    "symbol": symbol,
                    "missing_columns": missing,
                },
            )

        df = df[required_columns].copy()

        for column in required_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df = df.dropna()

        if len(df) < MIN_REQUIRED_BARS:
            return create_error_response(
                f"Insufficient historical data for '{symbol}'. "
                f"At least {MIN_REQUIRED_BARS} daily candles are required.",
                error_code="NO_DATA",
                details={
                    "symbol": symbol,
                    "available_bars": len(df),
                    "required_bars": MIN_REQUIRED_BARS,
                },
            )

        # ==========================================================
        # DAILY EMA STRUCTURE
        # ==========================================================

        close = df["Close"]

        df["EMA21"] = close.ewm(
            span=EMA_FAST,
            adjust=False,
        ).mean()

        df["EMA55"] = close.ewm(
            span=EMA_MID,
            adjust=False,
        ).mean()

        df["EMA100"] = close.ewm(
            span=EMA_LONG,
            adjust=False,
        ).mean()

        df["EMA200"] = close.ewm(
            span=EMA_TREND,
            adjust=False,
        ).mean()

        # ==========================================================
        # RSI 14
        # ==========================================================

        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / RSI_PERIOD,
            adjust=False,
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / RSI_PERIOD,
            adjust=False,
        ).mean()

        rs = avg_gain / avg_loss.replace(
            0,
            np.nan,
        )

        df["RSI"] = (
            100 -
            (
                100 /
                (1 + rs)
            )
        )

        # ==========================================================
        # MACD
        # ==========================================================

        ema12 = close.ewm(
            span=MACD_FAST,
            adjust=False,
        ).mean()

        ema26 = close.ewm(
            span=MACD_SLOW,
            adjust=False,
        ).mean()

        df["MACD"] = ema12 - ema26

        df["MACD_SIGNAL"] = df["MACD"].ewm(
            span=MACD_SIGNAL,
            adjust=False,
        ).mean()

        df["MACD_HIST"] = (
            df["MACD"] -
            df["MACD_SIGNAL"]
        )

        # ==========================================================
        # ATR 14
        # ==========================================================

        previous_close = close.shift(1)

        tr1 = (
            df["High"] -
            df["Low"]
        )

        tr2 = (
            df["High"] -
            previous_close
        ).abs()

        tr3 = (
            df["Low"] -
            previous_close
        ).abs()

        true_range = pd.concat(
            [
                tr1,
                tr2,
                tr3,
            ],
            axis=1,
        ).max(axis=1)

        df["ATR14"] = true_range.ewm(
            alpha=1 / ATR_PERIOD,
            adjust=False,
        ).mean()

        # ==========================================================
        # ADX 14
        # ==========================================================

        up_move = (
            df["High"] -
            df["High"].shift(1)
        )

        down_move = (
            df["Low"].shift(1) -
            df["Low"]
        )

        plus_dm = pd.Series(
            np.where(
                (up_move > down_move) &
                (up_move > 0),
                up_move,
                0.0,
            ),
            index=df.index,
        )

        minus_dm = pd.Series(
            np.where(
                (down_move > up_move) &
                (down_move > 0),
                down_move,
                0.0,
            ),
            index=df.index,
        )

        atr = df["ATR14"]

        plus_di = (
            100 *
            plus_dm.ewm(
                alpha=1 / ADX_PERIOD,
                adjust=False,
            ).mean() /
            atr.replace(
                0,
                np.nan,
            )
        )

        minus_di = (
            100 *
            minus_dm.ewm(
                alpha=1 / ADX_PERIOD,
                adjust=False,
            ).mean() /
            atr.replace(
                0,
                np.nan,
            )
        )

        di_sum = (
            plus_di +
            minus_di
        ).replace(
            0,
            np.nan,
        )

        dx = (
            100 *
            (
                plus_di -
                minus_di
            ).abs() /
            di_sum
        )

        df["ADX"] = dx.ewm(
            alpha=1 / ADX_PERIOD,
            adjust=False,
        ).mean()

        # ==========================================================
        # VOLUME
        # ==========================================================

        df["VOL_AVG20"] = (
            df["Volume"]
            .rolling(VOLUME_PERIOD)
            .mean()
        )

        df["VOLUME_RATIO"] = (
            df["Volume"] /
            df["VOL_AVG20"]
        )

        # ==========================================================
        # PREVIOUS 20-DAY HIGH
        # IMPORTANT:
        # shift FIRST, then rolling
        # so today's candle cannot be part of breakout level.
        # ==========================================================

        df["PREV20_HIGH"] = (
            df["High"]
            .shift(1)
            .rolling(BREAKOUT_PERIOD)
            .max()
        )

        # ==========================================================
        # WEEKLY DATA
        # ==========================================================

        weekly = df[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        ].resample(
            "W-FRI"
        ).agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        ).dropna()

        weekly["EMA21"] = weekly["Close"].ewm(
            span=21,
            adjust=False,
        ).mean()

        weekly["EMA55"] = weekly["Close"].ewm(
            span=55,
            adjust=False,
        ).mean()

        # ----------------------------------------------------------
        # USE PREVIOUS COMPLETED WEEK ONLY
        # ----------------------------------------------------------

        weekly_confirmed = weekly[
            [
                "Close",
                "EMA21",
                "EMA55",
            ]
        ].shift(1)

        weekly_confirmed = weekly_confirmed.rename(
            columns={
                "Close": "W_CLOSE",
                "EMA21": "W_EMA21",
                "EMA55": "W_EMA55",
            }
        )

        # ----------------------------------------------------------
        # Map weekly values to daily data.
        # Forward fill means Monday-Friday uses the last
        # completed weekly candle.
        # ----------------------------------------------------------

        df = df.join(
            weekly_confirmed,
            how="left",
        )

        df[
            [
                "W_CLOSE",
                "W_EMA21",
                "W_EMA55",
            ]
        ] = df[
            [
                "W_CLOSE",
                "W_EMA21",
                "W_EMA55",
            ]
        ].ffill()

        # ==========================================================
        # CAPITAL
        # ==========================================================

        capital = float(
            initial_capital
        )

        equity = capital

        # ==========================================================
        # POSITION STATE
        # ==========================================================

        position = False

        entry_price = 0.0
        stop_price = 0.0
        target_price = 0.0

        quantity = 0

        entry_date = None
        entry_index = None

        initial_risk_per_share = 0.0

        trades = []

        equity_curve = []

        # ==========================================================
        # BACKTEST
        # ==========================================================

        for i in range(
            MIN_REQUIRED_BARS,
            len(df) - 1,
        ):

            row = df.iloc[i]

            current_date = df.index[i]

            current_close = float(
                row["Close"]
            )

            current_high = float(
                row["High"]
            )

            current_low = float(
                row["Low"]
            )

            # ======================================================
            # EXIT LOGIC
            # ======================================================

            if position:

                holding_days = (
                    i -
                    entry_index
                )

                exit_price = None
                exit_reason = None

                # --------------------------------------------------
                # STOP LOSS
                # --------------------------------------------------

                if current_low <= stop_price:

                    exit_price = stop_price
                    exit_reason = "ATR_STOP"

                # --------------------------------------------------
                # TARGET
                # --------------------------------------------------

                elif current_high >= target_price:

                    exit_price = target_price
                    exit_reason = "TARGET_2R"

                # --------------------------------------------------
                # MAX HOLD
                # --------------------------------------------------

                elif holding_days >= MAX_HOLDING_DAYS:

                    exit_price = current_close
                    exit_reason = "MAX_HOLD"

                # --------------------------------------------------
                # TREND FAILURE
                # --------------------------------------------------

                elif current_close < float(
                    row["EMA21"]
                ):

                    exit_price = current_close
                    exit_reason = "EMA21_EXIT"

                # --------------------------------------------------
                # EXECUTE EXIT
                # --------------------------------------------------

                if exit_price is not None:

                    pnl = (
                        exit_price -
                        entry_price
                    ) * quantity

                    pnl_pct = (
                        (
                            exit_price /
                            entry_price
                        ) - 1
                    ) * 100

                    initial_risk = (
                        initial_risk_per_share *
                        quantity
                    )

                    realized_r = (
                        pnl /
                        initial_risk
                        if initial_risk > 0
                        else None
                    )

                    equity += pnl

                    trades.append(
                        {
                            "entry_date": str(
                                entry_date.date()
                            ),
                            "exit_date": str(
                                current_date.date()
                            ),
                            "holding_days": int(
                                holding_days
                            ),
                            "entry_price": round(
                                entry_price,
                                2,
                            ),
                            "exit_price": round(
                                exit_price,
                                2,
                            ),
                            "quantity": int(
                                quantity
                            ),
                            "initial_risk_per_share": round(
                                initial_risk_per_share,
                                2,
                            ),
                            "pnl": round(
                                pnl,
                                2,
                            ),
                            "pnl_pct": round(
                                pnl_pct,
                                2,
                            ),
                            "realized_R": round(
                                realized_r,
                                2,
                            ) if realized_r is not None else None,
                            "exit_reason": exit_reason,
                        }
                    )

                    position = False

                    entry_price = 0.0
                    stop_price = 0.0
                    target_price = 0.0
                    quantity = 0

                    entry_date = None
                    entry_index = None

                    initial_risk_per_share = 0.0

            # ======================================================
            # ENTRY LOGIC
            # ======================================================

            if not position:

                # --------------------------------------------------
                # WEEKLY TREND
                # --------------------------------------------------

                weekly_trend = (
                    pd.notna(row["W_CLOSE"])
                    and
                    pd.notna(row["W_EMA21"])
                    and
                    pd.notna(row["W_EMA55"])
                    and
                    float(row["W_CLOSE"]) >
                    float(row["W_EMA21"])
                    and
                    float(row["W_EMA21"]) >
                    float(row["W_EMA55"])
                )

                # --------------------------------------------------
                # DAILY EMA STRUCTURE
                # --------------------------------------------------

                daily_structure = (
                    current_close >
                    float(row["EMA21"])
                    >
                    float(row["EMA55"])
                    >
                    float(row["EMA100"])
                    >
                    float(row["EMA200"])
                )

                # --------------------------------------------------
                # RSI
                # --------------------------------------------------

                rsi = float(
                    row["RSI"]
                )

                rsi_ok = (
                    RSI_MIN <= rsi <= RSI_MAX
                )

                # --------------------------------------------------
                # MACD
                # --------------------------------------------------

                macd_ok = (
                    float(row["MACD"]) >
                    float(row["MACD_SIGNAL"])
                    and
                    float(row["MACD_HIST"]) >
                    0
                )

                # --------------------------------------------------
                # ADX
                # --------------------------------------------------

                adx = float(
                    row["ADX"]
                )

                adx_ok = (
                    adx >= MIN_ADX
                )

                # --------------------------------------------------
                # VOLUME
                # --------------------------------------------------

                volume_ratio = float(
                    row["VOLUME_RATIO"]
                )

                volume_ok = (
                    volume_ratio >=
                    MIN_VOLUME_RATIO
                )

                # --------------------------------------------------
                # NO CHASING
                # --------------------------------------------------

                extension_pct = (
                    (
                        current_close /
                        float(row["EMA21"])
                    ) - 1
                ) * 100

                not_extended = (
                    extension_pct <=
                    MAX_EXTENSION_PCT
                )

                # --------------------------------------------------
                # BREAKOUT
                # --------------------------------------------------

                breakout = (
                    pd.notna(row["PREV20_HIGH"])
                    and
                    current_close >
                    float(row["PREV20_HIGH"])
                )

                # --------------------------------------------------
                # PULLBACK RECLAIM
                # --------------------------------------------------

                previous_close = float(
                    df["Close"].iloc[i - 1]
                )

                previous_ema21 = float(
                    df["EMA21"].iloc[i - 1]
                )

                previous_high = float(
                    df["High"].iloc[i - 1]
                )

                pullback_reclaim = (
                    previous_close <=
                    previous_ema21
                    and
                    current_close >
                    float(row["EMA21"])
                    and
                    current_close >
                    previous_high
                )

                trigger = (
                    breakout or
                    pullback_reclaim
                )

                # --------------------------------------------------
                # FINAL SIGNAL
                # --------------------------------------------------

                signal = (
                    weekly_trend
                    and
                    daily_structure
                    and
                    rsi_ok
                    and
                    macd_ok
                    and
                    adx_ok
                    and
                    volume_ok
                    and
                    not_extended
                    and
                    trigger
                )

                if signal:

                    # ==================================================
                    # NEXT DAY OPEN = ACTUAL ENTRY
                    # ==================================================

                    next_row = df.iloc[i + 1]

                    next_open = float(
                        next_row["Open"]
                    )

                    atr = float(
                        row["ATR14"]
                    )

                    if (
                        not np.isfinite(atr)
                        or
                        atr <= 0
                    ):
                        continue

                    # ==================================================
                    # ATR STOP
                    # ==================================================

                    risk_per_share = (
                        ATR_MULTIPLIER *
                        atr
                    )

                    stop = (
                        next_open -
                        risk_per_share
                    )

                    # ==================================================
                    # 2R TARGET
                    # ==================================================

                    target = (
                        next_open +
                        (
                            risk_per_share *
                            REWARD_RISK
                        )
                    )

                    # ==================================================
                    # 1% CAPITAL RISK
                    # ==================================================

                    allowed_risk = (
                        equity *
                        RISK_PER_TRADE_PCT /
                        100
                    )

                    quantity_by_risk = int(
                        allowed_risk /
                        risk_per_share
                    )

                    quantity_by_capital = int(
                        equity /
                        next_open
                    )

                    quantity = min(
                        quantity_by_risk,
                        quantity_by_capital,
                    )

                    if quantity <= 0:
                        continue

                    # ==================================================
                    # OPEN POSITION
                    # ==================================================

                    position = True

                    entry_price = next_open

                    stop_price = stop

                    target_price = target

                    quantity = quantity

                    entry_date = df.index[
                        i + 1
                    ]

                    entry_index = i + 1

                    initial_risk_per_share = (
                        risk_per_share
                    )

            # ======================================================
            # EQUITY CURVE
            # ======================================================

            equity_curve.append(
                {
                    "date": str(
                        current_date.date()
                    ),
                    "equity": round(
                        equity,
                        2,
                    ),
                }
            )

        # ==========================================================
        # CLOSE OPEN POSITION AT END OF TEST
        # ==========================================================

        if position:

            last_close = float(
                df["Close"].iloc[-1]
            )

            last_date = df.index[-1]

            holding_days = (
                len(df) -
                1 -
                entry_index
            )

            pnl = (
                last_close -
                entry_price
            ) * quantity

            pnl_pct = (
                (
                    last_close /
                    entry_price
                ) - 1
            ) * 100

            initial_risk = (
                initial_risk_per_share *
                quantity
            )

            realized_r = (
                pnl /
                initial_risk
                if initial_risk > 0
                else None
            )

            equity += pnl

            trades.append(
                {
                    "entry_date": str(
                        entry_date.date()
                    ),
                    "exit_date": str(
                        last_date.date()
                    ),
                    "holding_days": int(
                        holding_days
                    ),
                    "entry_price": round(
                        entry_price,
                        2,
                    ),
                    "exit_price": round(
                        last_close,
                        2,
                    ),
                    "quantity": int(
                        quantity
                    ),
                    "initial_risk_per_share": round(
                        initial_risk_per_share,
                        2,
                    ),
                    "pnl": round(
                        pnl,
                        2,
                    ),
                    "pnl_pct": round(
                        pnl_pct,
                        2,
                    ),
                    "realized_R": round(
                        realized_r,
                        2,
                    ) if realized_r is not None else None,
                    "exit_reason": "END_OF_TEST",
                }
            )

        # ==========================================================
        # PERFORMANCE
        # ==========================================================

        total_trades = len(
            trades
        )

        winners = [
            trade
            for trade in trades
            if trade["pnl"] > 0
        ]

        losers = [
            trade
            for trade in trades
            if trade["pnl"] <= 0
        ]

        wins = len(winners)
        losses = len(losers)

        win_rate = (
            wins /
            total_trades *
            100
            if total_trades > 0
            else 0
        )

        net_profit = (
            equity -
            capital
        )

        return_pct = (
            net_profit /
            capital *
            100
        )

        gross_profit = sum(
            trade["pnl"]
            for trade in winners
        )

        gross_loss = abs(
            sum(
                trade["pnl"]
                for trade in losers
            )
        )

        profit_factor = (
            gross_profit /
            gross_loss
            if gross_loss > 0
            else None
        )

        average_win = (
            gross_profit /
            wins
            if wins > 0
            else 0
        )

        average_loss = (
            gross_loss /
            losses
            if losses > 0
            else 0
        )

        average_holding_days = (
            sum(
                trade["holding_days"]
                for trade in trades
            ) /
            total_trades
            if total_trades > 0
            else 0
        )

        # ==========================================================
        # MAX DRAWDOWN
        # ==========================================================

        running_equity = capital
        peak_equity = capital
        max_drawdown_pct = 0.0

        for trade in trades:

            running_equity += (
                trade["pnl"]
            )

            peak_equity = max(
                peak_equity,
                running_equity,
            )

            if peak_equity > 0:

                drawdown = (
                    (
                        running_equity -
                        peak_equity
                    ) /
                    peak_equity
                ) * 100

                max_drawdown_pct = min(
                    max_drawdown_pct,
                    drawdown,
                )

        # ==========================================================
        # BUY & HOLD
        # ==========================================================

        first_close = float(
            df["Close"].iloc[0]
        )

        final_close = float(
            df["Close"].iloc[-1]
        )

        buy_hold_return = (
            (
                final_close /
                first_close
            ) - 1
        ) * 100

        alpha = (
            return_pct -
            buy_hold_return
        )

        # ==========================================================
        # EXIT BREAKDOWN
        # ==========================================================

        target_exits = sum(
            1
            for trade in trades
            if trade["exit_reason"] ==
            "TARGET_2R"
        )

        stop_exits = sum(
            1
            for trade in trades
            if trade["exit_reason"] ==
            "ATR_STOP"
        )

        ema_exits = sum(
            1
            for trade in trades
            if trade["exit_reason"] ==
            "EMA21_EXIT"
        )

        max_hold_exits = sum(
            1
            for trade in trades
            if trade["exit_reason"] ==
            "MAX_HOLD"
        )

        # ==========================================================
        # BEST / WORST TRADE
        # ==========================================================

        best_trade = (
            max(
                trades,
                key=lambda x: x["pnl"]
            )
            if trades
            else None
        )

        worst_trade = (
            min(
                trades,
                key=lambda x: x["pnl"]
            )
            if trades
            else None
        )

        # ==========================================================
        # FINAL RESULT
        # ==========================================================

        result = {

            "backtest": {

                "status": "COMPLETED",

                "symbol": symbol,

                "period": period,

                "strategy_name": (
                    "FINAL LOCKED "
                    "MULTI-FACTOR SWING/POSITIONAL"
                ),

                "timeframe": (
                    "Weekly trend + Daily setup"
                ),

                "style": (
                    "Swing / Positional only"
                ),

                "initial_capital": round(
                    capital,
                    2,
                ),

                "final_capital": round(
                    equity,
                    2,
                ),

                "net_profit": round(
                    net_profit,
                    2,
                ),

                "return_pct": round(
                    return_pct,
                    2,
                ),

                "buy_hold_return_pct": round(
                    buy_hold_return,
                    2,
                ),

                "alpha_vs_buy_hold_pct": round(
                    alpha,
                    2,
                ),

                "total_trades": (
                    total_trades
                ),

                "winning_trades": (
                    wins
                ),

                "losing_trades": (
                    losses
                ),

                "win_rate_pct": round(
                    win_rate,
                    2,
                ),

                "profit_factor": (
                    round(
                        profit_factor,
                        2,
                    )
                    if profit_factor is not None
                    else None
                ),

                "average_win": round(
                    average_win,
                    2,
                ),

                "average_loss": round(
                    average_loss,
                    2,
                ),

                "average_holding_days": round(
                    average_holding_days,
                    2,
                ),

                "max_drawdown_pct": round(
                    max_drawdown_pct,
                    2,
                ),

                "best_trade": best_trade,

                "worst_trade": worst_trade,

                "exit_breakdown": {

                    "TARGET_2R": (
                        target_exits
                    ),

                    "ATR_STOP": (
                        stop_exits
                    ),

                    "EMA21_EXIT": (
                        ema_exits
                    ),

                    "MAX_HOLD": (
                        max_hold_exits
                    ),
                },

                "LOCKED_RULES": {

                    "weekly": (
                        "Weekly Close > Weekly EMA21 "
                        "> Weekly EMA55"
                    ),

                    "daily_structure": (
                        "Close > EMA21 > EMA55 "
                        "> EMA100 > EMA200"
                    ),

                    "rsi": (
                        "RSI 50-70"
                    ),

                    "macd": (
                        "MACD > Signal AND "
                        "Histogram > 0"
                    ),

                    "adx": (
                        "ADX >= 20"
                    ),

                    "volume": (
                        "Volume >= 20-day average"
                    ),

                    "entry_trigger": (
                        "20-day breakout OR "
                        "pullback reclaim"
                    ),

                    "no_chasing": (
                        "Close <= 5% above EMA21"
                    ),

                    "entry_execution": (
                        "Next trading day OPEN"
                    ),

                    "stop_loss": (
                        "1.5 x ATR14"
                    ),

                    "target": (
                        "2R"
                    ),

                    "risk_per_trade": (
                        "1% of current equity"
                    ),

                    "maximum_holding": (
                        "15 trading days"
                    ),

                    "exit": (
                        "ATR Stop / 2R Target / "
                        "EMA21 / Max Hold"
                    ),
                },

                "data_integrity": {

                    "look_ahead_bias": (
                        "AVOIDED"
                    ),

                    "weekly_confirmation": (
                        "Previous completed week"
                    ),

                    "entry_price": (
                        "Next trading day Open"
                    ),

                    "intraday_execution": (
                        "NOT USED"
                    ),

                    "options_OI": (
                        "NOT USED IN BACKTEST"
                    ),

                    "options_liquidity": (
                        "NOT USED IN BACKTEST"
                    ),

                    "institutional_flow": (
                        "NOT USED IN BACKTEST"
                    ),
                },

                "parameters_locked": {

                    "EMA21": EMA_FAST,
                    "EMA55": EMA_MID,
                    "EMA100": EMA_LONG,
                    "EMA200": EMA_TREND,

                    "RSI_PERIOD": RSI_PERIOD,
                    "RSI_MIN": RSI_MIN,
                    "RSI_MAX": RSI_MAX,

                    "MACD_FAST": MACD_FAST,
                    "MACD_SLOW": MACD_SLOW,
                    "MACD_SIGNAL": MACD_SIGNAL,

                    "ATR_PERIOD": ATR_PERIOD,
                    "ATR_MULTIPLIER": ATR_MULTIPLIER,

                    "ADX_PERIOD": ADX_PERIOD,
                    "MIN_ADX": MIN_ADX,

                    "VOLUME_PERIOD": VOLUME_PERIOD,
                    "MIN_VOLUME_RATIO": MIN_VOLUME_RATIO,

                    "BREAKOUT_PERIOD": BREAKOUT_PERIOD,

                    "MAX_EXTENSION_PCT": (
                        MAX_EXTENSION_PCT
                    ),

                    "RISK_PER_TRADE_PCT": (
                        RISK_PER_TRADE_PCT
                    ),

                    "REWARD_RISK": (
                        REWARD_RISK
                    ),

                    "MAX_HOLDING_DAYS": (
                        MAX_HOLDING_DAYS
                    ),
                },
            },

            "trades": trades,

            "equity_curve": equity_curve,
        }

        return dump_json(result)

    except Exception as exc:

        logger.exception(
            "Final strategy backtest failed for {}",
            symbol,
        )

        return create_error_response(
            f"Backtest failed for '{symbol}'.",
            error_code="BACKTEST_ERROR",
            details={
                "symbol": symbol,
                "period": period,
                "exception": str(exc),
            },
        )


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
