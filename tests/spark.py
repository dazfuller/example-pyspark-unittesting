import unittest

from pyspark.conf import SparkConf
from pyspark.sql import DataFrame, SparkSession


class PySparkTestCase(unittest.TestCase):
    """A class for testing against a Spark session.

    A Spark session is created before tests are executed, and is stopped once all tests have
    completed. Tests should therefore cover the same are of functionallity.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Set up environment prior to executing any tests."""
        conf = SparkConf()
        conf.setAppName("demo-unittests")
        conf.setMaster("local[*]")
        conf.set("spark.sql.session.timezone", "UTC")

        cls.spark = SparkSession.builder.config(conf=conf).getOrCreate()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clear down environment after all tests have completed."""
        cls.spark.stop()

    def tearDown(self) -> None:
        """Perform any clean up after each test has completed."""
        self.spark.catalog.clearCache()

    def assert_dataframes_equal(self, expected: DataFrame, actual: DataFrame, display_differences: bool = False) -> None:
        """Assert if two data frames are equal.

        Implemented using the DataFrame.subtract method. If the data frames are not equal then
        the differences are display if instructed to do so.

        Parameters
        ----------
        expected : DataFrame
            DataFrame containing the expected data.
        actual : DataFrame
            DataFrame containing the actual data from executing the code under test.
        display_differences : bool, optional
            When true will display the difference between the data frames. False by default.
        """
        diff: DataFrame = actual.subtract(expected).cache()
        diff_count: int = diff.count()

        try:
            if diff_count != 0 and display_differences:
                diff.show()
            self.assertEqual(0, diff_count)
        finally:
            diff.unpersist
