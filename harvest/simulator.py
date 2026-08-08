from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


LOT_SIZE = 100
STRATEGY_DRIP = 'With DRIP'
STRATEGY_NO_DRIP = 'No DRIP'


class SimulatorValidationError(ValueError):
    """Raised when simulator inputs cannot produce a meaningful result."""


def build_allocations(rows: Sequence[tuple[object, object]]) -> dict[str, float]:
    allocations = {}
    for raw_stock, raw_value in rows:
        stock = '' if pd.isna(raw_stock) else str(raw_stock).strip().upper()
        if not stock:
            continue
        if stock in allocations:
            raise SimulatorValidationError('Stock tickers must be unique.')
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise SimulatorValidationError(
                f'Initial investment for {stock} must be a number.'
            ) from None
        if not np.isfinite(value) or value <= 0:
            raise SimulatorValidationError(
                f'Initial investment for {stock} must be greater than zero.'
            )
        allocations[stock] = value
    if not allocations:
        raise SimulatorValidationError('Please add at least one stock ticker.')
    return allocations


def normalize_price_history(price_df: pd.DataFrame, stock: str) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        raise SimulatorValidationError(f'No price data found for {stock}.')
    missing = {'date', 'close'} - set(price_df.columns)
    if missing:
        raise SimulatorValidationError(
            f'Price data for {stock} is missing: {", ".join(sorted(missing))}.'
        )

    normalized = price_df[['date', 'close']].copy()
    normalized['date'] = pd.to_datetime(normalized['date'], errors='coerce')
    normalized['close'] = pd.to_numeric(normalized['close'], errors='coerce')
    normalized = normalized.dropna(subset=['date', 'close'])
    normalized = normalized[normalized['close'] > 0]
    normalized = normalized.sort_values('date').drop_duplicates('date', keep='last')
    normalized = normalized.reset_index(drop=True)
    if normalized.empty:
        raise SimulatorValidationError(f'No valid price data found for {stock}.')
    return normalized


def normalize_dividend_history(dividend_df: pd.DataFrame | None, stock: str) -> pd.DataFrame:
    if dividend_df is None or dividend_df.empty:
        return pd.DataFrame(columns=['date', 'adjDividend', 'stock'])
    missing = {'date', 'adjDividend'} - set(dividend_df.columns)
    if missing:
        raise SimulatorValidationError(
            f'Dividend data for {stock} is missing: {", ".join(sorted(missing))}.'
        )

    normalized = dividend_df[['date', 'adjDividend']].copy()
    normalized['date'] = pd.to_datetime(normalized['date'], errors='coerce')
    normalized['adjDividend'] = pd.to_numeric(
        normalized['adjDividend'], errors='coerce'
    )
    normalized = normalized.dropna(subset=['date', 'adjDividend'])
    normalized = normalized[normalized['adjDividend'] >= 0]
    normalized['stock'] = stock
    return normalized.sort_values('date').reset_index(drop=True)


def simulate_historical_portfolio(
    allocations: Mapping[str, float],
    prices: Mapping[str, pd.DataFrame],
    dividends: Mapping[str, pd.DataFrame | None],
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if start_year > end_year:
        raise SimulatorValidationError('Start year must not be after end year.')
    if not allocations:
        raise SimulatorValidationError('Add at least one stock allocation.')
    if len(set(allocations)) != len(allocations):
        raise SimulatorValidationError('Stock tickers must be unique.')

    clean_allocations = {}
    for stock, allocation in allocations.items():
        value = float(allocation)
        if not np.isfinite(value) or value <= 0:
            raise SimulatorValidationError(
                f'Initial investment for {stock} must be greater than zero.'
            )
        clean_allocations[stock] = value

    normalized_prices = {
        stock: normalize_price_history(prices.get(stock), stock)
        for stock in clean_allocations
    }
    normalized_dividends = {
        stock: normalize_dividend_history(dividends.get(stock), stock)
        for stock in clean_allocations
    }

    first_day = pd.Timestamp(year=start_year, month=1, day=1)
    last_day = pd.Timestamp(year=end_year, month=12, day=31)
    lots = {STRATEGY_DRIP: {}, STRATEGY_NO_DRIP: {}}
    cash = {STRATEGY_DRIP: {}, STRATEGY_NO_DRIP: {}}
    purchase_dates = {}
    transactions = []

    for stock, allocation in clean_allocations.items():
        buy_row = _price_on_or_after(normalized_prices[stock], first_day)
        if buy_row is None or buy_row['date'].year != start_year:
            raise SimulatorValidationError(
                f'Price data for {stock} is not available in {start_year}.'
            )
        purchased_lots = int(allocation // (buy_row['close'] * LOT_SIZE))
        purchase_cost = purchased_lots * buy_row['close'] * LOT_SIZE
        residual_cash = allocation - purchase_cost
        purchase_dates[stock] = buy_row['date']

        for strategy in (STRATEGY_DRIP, STRATEGY_NO_DRIP):
            lots[strategy][stock] = purchased_lots
            cash[strategy][stock] = residual_cash

        transactions.append({
            'Date': buy_row['date'],
            'Strategy': 'Both',
            'Stock': stock,
            'Activity': (
                f'Buy {purchased_lots:,} lots @ {buy_row["close"]:,.2f} '
                f'for IDR {purchase_cost:,.0f}'
            ),
        })

    event_frames = []
    for stock, frame in normalized_dividends.items():
        if frame.empty:
            continue
        event_frames.append(
            frame[(frame['date'] >= purchase_dates[stock]) & (frame['date'] <= last_day)]
        )
    if event_frames:
        events = pd.concat(event_frames, ignore_index=True).sort_values(
            ['date', 'stock']
        )
    else:
        events = pd.DataFrame(columns=['date', 'adjDividend', 'stock'])

    aggregate_rows = []
    detail_rows = []
    for year in range(start_year, end_year + 1):
        annual_income = {STRATEGY_DRIP: 0.0, STRATEGY_NO_DRIP: 0.0}
        year_events = events[
            (events['date'] >= pd.Timestamp(year=year, month=1, day=1))
            & (events['date'] <= pd.Timestamp(year=year, month=12, day=31))
        ]

        for event in year_events.itertuples(index=False):
            stock = event.stock
            dividend_per_share = float(event.adjDividend)
            for strategy in (STRATEGY_DRIP, STRATEGY_NO_DRIP):
                payment = lots[strategy][stock] * LOT_SIZE * dividend_per_share
                cash[strategy][stock] += payment
                annual_income[strategy] += payment
                transactions.append({
                    'Date': event.date,
                    'Strategy': strategy,
                    'Stock': stock,
                    'Activity': f'Receive dividend of IDR {payment:,.0f}',
                })

            buy_row = _price_on_or_after(normalized_prices[stock], event.date)
            if buy_row is None or buy_row['date'].year != year:
                continue
            available_cash = cash[STRATEGY_DRIP][stock]
            purchased_lots = int(available_cash // (buy_row['close'] * LOT_SIZE))
            if purchased_lots == 0:
                continue
            purchase_cost = purchased_lots * buy_row['close'] * LOT_SIZE
            lots[STRATEGY_DRIP][stock] += purchased_lots
            cash[STRATEGY_DRIP][stock] -= purchase_cost
            transactions.append({
                'Date': buy_row['date'],
                'Strategy': STRATEGY_DRIP,
                'Stock': stock,
                'Activity': (
                    f'Reinvest into {purchased_lots:,} lots @ '
                    f'{buy_row["close"]:,.2f} for IDR {purchase_cost:,.0f}'
                ),
            })

        year_end = pd.Timestamp(year=year, month=12, day=31)
        for strategy in (STRATEGY_DRIP, STRATEGY_NO_DRIP):
            holdings_value = 0.0
            for stock in clean_allocations:
                price_row = _price_on_or_before(normalized_prices[stock], year_end)
                if price_row is None or price_row['date'].year != year:
                    raise SimulatorValidationError(
                        f'Price data for {stock} is not available through {year}.'
                    )
                value = lots[strategy][stock] * price_row['close'] * LOT_SIZE
                holdings_value += value
                detail_rows.append({
                    'stock': stock,
                    'year': f'Year {year}',
                    'lot': lots[strategy][stock],
                    'price': float(price_row['close']),
                    'value': value,
                    'Strategy': strategy,
                })

            cash_value = float(sum(cash[strategy].values()))
            portfolio_value = holdings_value + cash_value
            detail_rows.append({
                'stock': 'Cash',
                'year': f'Year {year}',
                'lot': np.nan,
                'price': np.nan,
                'value': cash_value,
                'Strategy': strategy,
            })
            aggregate_rows.append({
                'year': f'Year {year}',
                'strategy': strategy,
                'holdings_value': holdings_value,
                'cash': cash_value,
                'portfolio_value': portfolio_value,
                'dividend_income': annual_income[strategy],
                'investment': portfolio_value,
                'returns': annual_income[strategy],
            })

    aggregate_df = pd.DataFrame(aggregate_rows)
    detail_df = pd.DataFrame(detail_rows)
    transaction_df = pd.DataFrame(transactions)
    if not transaction_df.empty:
        transaction_df = transaction_df.sort_values(
            ['Date', 'Strategy', 'Stock'], ascending=[False, True, True]
        ).reset_index(drop=True)
        transaction_df['Date'] = transaction_df['Date'].dt.strftime('%Y-%m-%d')
    return aggregate_df, detail_df, transaction_df


def _price_on_or_after(price_df: pd.DataFrame, date: pd.Timestamp):
    matches = price_df[price_df['date'] >= date]
    return None if matches.empty else matches.iloc[0]


def _price_on_or_before(price_df: pd.DataFrame, date: pd.Timestamp):
    matches = price_df[price_df['date'] <= date]
    return None if matches.empty else matches.iloc[-1]
