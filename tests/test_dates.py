"""Provides tests for the date module."""

import importlib.resources as resources
import unittest
from datetime import date, datetime

from demo.dates import create_date_dimension, get_iso_weeknumber
from pyspark.sql.functions import col
from pyspark.sql.types import (DateType, IntegerType, StringType, StructField,
                               StructType)

from tests.spark import PySparkTestCase


class ISOCalendarTests(unittest.TestCase):
    """Tests to validate methods performing ISO calendar functionality."""

    def test_simple_date(self):
        """Check for the week number of a given date."""
        dt = date(2021, 3, 27)
        expected_weeknum = 12

        result: int = get_iso_weeknumber(dt)

        self.assertEqual(result, expected_weeknum)

    def test_for_week_53(self):
        """The 1st Jan 2021 was a Friday and so the first full week with a Thursday in was the following week, so ensure the ISO week is recognized as 53."""
        dt = date(2021, 1, 1)
        expected_weeknum = 53

        result: int = get_iso_weeknumber(dt)

        self.assertEqual(result, expected_weeknum)

    def test_for_week_1(self):
        """The 1st Jan 2020 was a Tuesday and so contained a Thursday, so ensure the ISO week is recognized as 1."""
        dt = date(2020, 1, 1)
        expected_weeknum = 1

        result: int = get_iso_weeknumber(dt)

        self.assertEqual(result, expected_weeknum)

    def test_for_datetime(self):
        """Check for the week number of a given date with a time element."""
        dt = datetime(2021, 3, 27, 12, 10, 52)
        expected_weeknum = 12

        result: int = get_iso_weeknumber(dt)

        self.assertEqual(result, expected_weeknum)

    def test_for_non_date_value(self):
        """Check that a None value is returned for non-date objects."""
        result = get_iso_weeknumber('2021-03-27')
        self.assertIsNone(result)


class DateDimensionTests(PySparkTestCase):
    """Tests to validate the creation of date dimension data frames."""

    @staticmethod
    def get_dimension_schema() -> StructType:
        """Get the date dimension schema definition for testing.

        Returns
        -------
        StructType
            The schema of the date dimension under stest
        """
        return StructType([
            StructField("id", StringType()),
            StructField("date", DateType()),
            StructField("date_key", IntegerType()),
            StructField("day", IntegerType()),
            StructField("day_name", StringType()),
            StructField("day_name_short", StringType()),
            StructField("month", IntegerType()),
            StructField("month_name", StringType()),
            StructField("month_name_short", StringType()),
            StructField("year", IntegerType()),
            StructField("week_number", IntegerType())
        ])

    def get_input_file(self, file_name: str) -> str:
        """Get an input file from the package resources.

        Parameters
        ----------
        file_name : str
            The name of the file to load

        Returns
        -------
        str
            The path to the input file
        """
        package: str = "tests.resources.dates"

        if not resources.is_resource(package, file_name):
            raise FileExistsError(f"The file '{file_name}' does not exist as a resource")

        with resources.open_binary(package, file_name) as r:
            return r.name

    def test_simple_creation(self):
        """Validate the creation of a simple date dimension."""
        expected = self.spark.read.csv(
            self.get_input_file("simple_date_dim.csv"),
            header=True,
            schema=DateDimensionTests.get_dimension_schema())

        result = create_date_dimension(self.spark, datetime(2021, 1, 1), datetime(2021, 1, 5))

        self.assert_dataframes_equal(expected, result, display_differences=True)

    def test_simple_range_week53(self):
        """Confirm that week 53 contains 7 days when it crosses over a year."""
        week_53_day_count = (
            create_date_dimension(self.spark, datetime(2020, 12, 1), datetime(2021, 1, 31))
            .filter(col("week_number") == 53)
            .count()
        )

        self.assertEqual(7, week_53_day_count)
