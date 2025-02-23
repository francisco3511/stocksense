import datetime as dt
from typing import Optional

import polars as pl


def round_params(params):
    return [round(v, 3) if isinstance(v, float) else v for v in params.values()]


def format_xgboost_params(solution: dict, seed: Optional[int] = None) -> dict:
    """
    Format model parameters.

    Parameters
    ----------
    solution : dict
        Parameter solution encoded as a dictionary.
    seed : Optional[int]
        Random seed. If None, no seed is set (allows randomness).
    """
    params = {
        "objective": "binary:logistic",
        "learning_rate": round(solution["learning_rate"], 3),
        "n_estimators": round(solution["n_estimators"]),
        "max_depth": round(solution["max_depth"]),
        "min_child_weight": round(solution["min_child_weight"], 3),
        "gamma": round(solution["gamma"], 3),
        "subsample": round(solution["subsample"], 3),
        "colsample_bytree": round(solution["colsample_bytree"], 3),
        "reg_alpha": round(solution["reg_alpha"], 3),
        "reg_lambda": round(solution["reg_lambda"], 3),
        "scale_pos_weight": round(solution["scale_pos_weight"], 3),
        "eval_metric": "logloss",
        "tree_method": "hist",
        "nthread": -1,
    }

    if seed is not None:
        params["random_state"] = seed

    return params


def get_train_val_splits(
    data: pl.DataFrame,
    max_train_years: int = 10,
    max_splits: int = 1,
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Generate training/validation splits using yearly blocks of validation data.
    """
    quarters = (
        data.select(pl.col("tdq"))
        .unique()
        .sort("tdq", descending=True)
        .get_column("tdq")
        .to_list()
    )

    # Get validation periods (4Q blocks)
    val_years = [quarters[i:i+4] for i in range(0, len(quarters), 4)][:max_splits]

    splits = []
    for val_quarters in val_years:
        last_val_quarter = max(val_quarters)
        val_idx = quarters.index(last_val_quarter)

        # Add 7 to skip: 4Q validation period + 3Q gap (total 4Q gap)
        train_start_idx = val_idx + 7
        train_end_idx = min(train_start_idx + max_train_years * 4, len(quarters))
        train_quarters = quarters[train_start_idx:train_end_idx]

        if len(train_quarters) < 20:
            continue

        train = data.filter(pl.col("tdq").is_in(train_quarters))
        val = data.filter(pl.col("tdq").is_in(val_quarters))
        splits.append((train, val))

    return splits


def get_train_bounds(
    trade_date: dt.datetime,
    max_train_years: int = 10,
) -> tuple[dt.datetime, dt.datetime]:
    """
    Get the training bounds for the model.

    Parameters
    ----------
    trade_date : dt.datetime
        Trade date.
    max_train_years : int
        Maximum number of years for training.
    """
    start_date = trade_date.replace(year=trade_date.year - 1 - max_train_years)
    end_date = trade_date.replace(year=trade_date.year - 1)
    return start_date, end_date


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


def get_dataset_imbalance_scale(data: pl.DataFrame, target: str) -> float:
    """
    Compute dataset class imbalance scale.

    Parameters
    ----------
    data : pl.DataFrame
        Training dataset
    target : str
        Target variable to compute class imbalance scale for

    Returns
    -------
    float
        Class imbalance scale (neg_count/pos_count) if significant imbalance exists,
        otherwise 1.0
    """
    neg_count = len(data.filter(pl.col(target) == 0))
    pos_count = len(data.filter(pl.col(target) == 1))

    if pos_count == 0:
        return 1.0

    return round(neg_count / pos_count, 2)
