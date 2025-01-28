import datetime as dt
from typing import List, Optional

import polars as pl


def format_xgboost_params(solution: List[float], scale: float) -> dict:
    """
    Format model parameters.

    Parameters
    ----------
    solution : List[float]
        GA solution encoded as a list.
    scale : float
        Class imbalance scale.
    """
    return {
        "objective": "binary:logistic",
        "learning_rate": round(solution[0], 3),
        "n_estimators": round(solution[1]),
        "max_depth": round(solution[2]),
        "min_child_weight": round(solution[3], 2),
        "gamma": round(solution[4], 2),
        "subsample": round(solution[5], 2),
        "colsample_bytree": round(solution[6], 2),
        "reg_alpha": round(solution[7], 2),
        "reg_lambda": round(solution[8], 2),
        "scale_pos_weight": scale,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "nthread": -1,
        "random_state": 100,
    }


def get_train_val_splits(
    data: pl.DataFrame, min_train_years: int = 5, val_years: int = 1, max_splits: int = 3
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Generate training/validation splits using expanding window approach,
    starting from most recent years and moving backwards.

    Parameters
    ----------
    data : pl.DataFrame
        Training data to split.
    min_train_years : int
        Minimum number of years required for training
    max_splits : int
        Maximum number of splits to return

    Returns
    -------
    list[tuple[pl.DataFrame]]
        List of (train, validation) splits, ordered from most recent to oldest.
    """
    # Get sorted unique quarters
    quarters = (
        data.select(pl.col("tdq")).unique().sort("tdq", descending=True).get_column("tdq").to_list()
    )

    # Convert years to quarters
    min_train_quarters = min_train_years * 4
    val_window = val_years * 4

    # Validate enough data exists
    if len(quarters) < min_train_quarters + val_window:
        raise ValueError(
            f"Not enough years in dataset. Need at least {min_train_years + 2} years "
            f"({min_train_years} for training, 2 for validation)."
        )

    # Generate expanding window splits
    splits = []
    for i in range(0, len(quarters) - min_train_quarters - val_window - 1, val_window):
        val_quarters = quarters[i : (i + val_window)]
        train_quarters = quarters[(i + val_window + 4) :]
        if len(train_quarters) < min_train_quarters:
            break

        train = data.filter(pl.col("tdq").is_in(train_quarters))
        val = data.filter(pl.col("tdq").is_in(val_quarters))
        splits.append((train, val))

    if max_splits and max_splits > 0:
        splits = splits[:max_splits]

    return splits[::-1]


def validate_trade_date(date: dt.datetime) -> bool:
    """
    Validate if the trade date is one of the allowed dates
    (March 1st, June 1st, September 1st, or December 1st).

    Parameters
    ----------
    date : dt.datetime
        Date to validate.

    Returns
    -------
    bool
        True if date is valid, False otherwise.
    """
    allowed_months = [3, 6, 9, 12]
    return date.month in allowed_months and date.day == 1


def find_last_trading_date() -> Optional[dt.datetime]:
    """
    Find last trading date, which will be used for stock selection.

    Returns
    -------
    dt.datime
        Trading date.
    """

    today = dt.datetime.today()
    trade_dates = [
        dt.datetime(today.year - 1, 12, 1),
        dt.datetime(today.year, 3, 1),
        dt.datetime(today.year, 6, 1),
        dt.datetime(today.year, 9, 1),
        dt.datetime(today.year, 12, 1),
    ]
    past_dates = [date for date in trade_dates if date <= today]

    if past_dates:
        return max(past_dates)
    else:
        return None
