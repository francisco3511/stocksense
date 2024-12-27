import pandera as pa
import polars as pl
from loguru import logger

from .schemas import FinancialSchema, InsiderSchema, MarketSchema, StockSchema, validate_df


class DataValidator:
    """Data validation handler for database operations."""

    _schemas = {
        "stock": StockSchema,
        "financial": FinancialSchema,
        "market": MarketSchema,
        "insider": InsiderSchema,
    }

    @classmethod
    def validate(cls, table_name: str, data: pl.DataFrame) -> pl.DataFrame:
        """
        Validate data against schema before database insertion.

        Parameters
        ----------
        table_name : str
            Name of the table the data is being inserted into
        data : pl.DataFrame
            Data to validate

        Returns
        -------
        pl.DataFrame
            Validated data

        Raises
        ------
        pa.errors.SchemaError
            If data fails validation
        KeyError
            If table_name doesn't have a defined schema
        """
        try:
            schema = cls._schemas.get(table_name)
            if not schema:
                logger.warning(f"No schema defined for table {table_name}")
                return data

            validated_data = validate_df(schema, data)
            return validated_data

        except pa.errors.SchemaError as e:
            logger.error(f"Validation failed for {table_name}: {str(e)}")
            raise
