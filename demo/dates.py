"""Provides methods for working with dates in Python."""

from datetime import date, datetime, time, timedelta, timezone
from typing import Union

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (date_format, dayofmonth, from_unixtime,
                                   month, pandas_udf, to_date, year)
from pyspark.sql.types import IntegerType


def get_iso_weeknumber(dt: Union[date, datetime]) -> int:
    """Return the ISO week number for a given date or datetime.

    Parameters
    ----------
    dt : Union[date, datetime]
        Date object to return the ISO week number for

    Returns
    -------
    int
        The ISO week number
    """
    if type(dt) is not datetime and type(dt) is not date:
        return None

    return dt.isocalendar()[1]


@pandas_udf(IntegerType())
def iso_week_of_year(dt: pd.Series) -> pd.Series:
    """ISO week of the year as part of ISO 8601 standard.

    Week 1 is the week with the first Thursday of the gregorian calendar in it. Further information
    is available in [Wikipedia](https://en.wikipedia.org/wiki/ISO_week_date).

    Parameters
    ----------
    dt : pd.Series
        The series containing the date or datetime values

    Returns
    -------
    pd.Series
        A new series with the same number of records containing the week number for the year
    """
    return dt.apply(lambda x: get_iso_weeknumber(x) if not pd.isnull(x) else pd.NA)


def create_date_dimension(spark: SparkSession, start: datetime, end: datetime) -> DataFrame:
    """Create a date dimension.

    Generates a date dimension for dates between the start and end dates provided.

    Parameters
    ----------
    spark : SparkSession
        Session instance to use to create the date dimension
    start : Union[date, datetime]
        The first day of the date dimension
    end : Union[date, datetime]
        The last day of the date dimensions

    Returns
    -------
    DataFrame
        A new DataFrame containing the date dimension data

    Raises
    ------
    ValueError
        If an invalid SparkSession instance is provided
    ValueError
        If the start value is not a valid datetime.datetime
    ValueError
        If the end value is not a valid datetime.datetime
    ValueError
        If the provided start date occurs after the end date
    """
    if spark is None or type(spark) is not SparkSession:
        raise ValueError("A valid SparkSession instance must be provided")

    if type(start) is not datetime:
        raise ValueError("Start date must be a datetime.datetime object")

    if type(end) is not datetime:
        raise ValueError("End date must be a datetime.datetime object")

    if start >= end:
        raise ValueError("Start date must be before the end date")

    if start.tzinfo is None:
        start = datetime.combine(start.date(), time(0, 0, 0), tzinfo=timezone.utc)

    if end.tzinfo is None:
        end = datetime.combine(end.date(), time(0, 0, 0), tzinfo=timezone.utc)

    end = end + timedelta(days=1)

    return (
        spark.range(start=start.timestamp(), end=end.timestamp(), step=24 * 60 * 60)
             .withColumn("date", to_date(from_unixtime("id")))
             .withColumn("date_key", date_format("date", "yyyyMMdd").cast("int"))
             .withColumn("day", dayofmonth("date"))
             .withColumn("day_name", date_format("date", "EEEE"))
             .withColumn("day_short_name", date_format("date", "EEE"))
             .withColumn("month", month("date"))
             .withColumn("month_name", date_format("date", "MMMM"))
             .withColumn("month_short_name", date_format("date", "MMM"))
             .withColumn("year", year("date"))
             .withColumn("week_number", iso_week_of_year("date"))
    )
