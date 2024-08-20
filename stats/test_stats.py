"""Test module for stats librairie."""

import unittest
from random import shuffle

from stats import count, lower_quartile, max, mean, median, median_quartile, min, std, upper_quartile, iqr, d_range


class TestMean(unittest.TestCase):
    """Test module for mean function."""

    def test_empty(self):
        """Test with empty list."""
        with self.assertRaises(ValueError):
            mean([])

    def test_same(self):
        """Test with list filled with same values."""
        self.assertEqual(mean([5, 5, 5, 5, 5, 5, 5, 5]), 5)

    def test_basic(self):
        """Test with basic list."""
        self.assertEqual(mean([1, 2, 3, 4, 5]), 3)

    def test_nan(self):
        """Test with float("nan") values in list."""
        self.assertEqual(mean([1, 2, 3, float("nan"), 4, 5]), 3)


class TestMedian(unittest.TestCase):
    """Test module for median function."""

    def test_empty(self):
        """Test with empty list."""
        with self.assertRaises(ValueError):
            median([])

    def test_same(self):
        """Test with list filled with same values."""
        self.assertEqual(median([6, 6, 6, 6, 6, 6, 6, 6]), 6)

    def test_odd(self):
        """Test with n is odd."""
        self.assertEqual(median([1, 2, 4, 5, 6]), 4)

    def test_even(self):
        """Test with n is even."""
        self.assertEqual(median([1, 2, 3, 4]), 2.5)

    def test_nan(self):
        """Test with float("nan") values in list."""
        self.assertEqual(mean([1, 2, 3, float("nan"), 4]), 2.5)


class TestCount(unittest.TestCase):
    """Test module for count function."""

    def test_empty(self):
        """Test with empty list."""
        self.assertEqual(count([]), 0)

    def test_basic(self):
        """Test with basic list."""
        self.assertEqual(count([1, 2, 3, 4, 5]), 5)

    def test_nan(self):
        """Test with float("nan") values in list."""
        self.assertEqual(count([1, 2, float("nan"), 3, 4, float("nan"), 5]), 5)


class TestMax(unittest.TestCase):
    """Test module for max function."""

    def test_empty(self):
        """Test with empty list."""
        with self.assertRaises(ValueError):
            max([])

    def test_sorted(self):
        """Test with sorted list."""
        self.assertEqual(max([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 10)

    def test_nan(self):
        """Test with float("nan") values in list."""
        self.assertEqual(max([1, 2, 3, 4, float("nan"), 5, 6, 7, 8, float("nan"), 9, 10]), 10)

    def test_shuffle(self):
        """Test with unsorted list."""
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        shuffle(lst)
        self.assertEqual(max(lst), 10)


class TestMin(unittest.TestCase):
    """Test module for min function."""

    def test_empty(self):
        """Test with empty list."""
        with self.assertRaises(ValueError):
            min([])

    def test_sorted(self):
        """Test with sorted list."""
        self.assertEqual(min([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 1)

    def test_nan(self):
        """Test with float("nan") values in list."""
        self.assertEqual(min([1, 2, 3, 4, float("nan"), 5, 6, 7, 8, float("nan"), 9, 10]), 1)

    def test_shuffle(self):
        """Test with unsorted list."""
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]
        shuffle(lst)
        self.assertEqual(min(lst), 1)


class TestStd(unittest.TestCase):
    """Test module for standard deviation function."""

    def test_empty(self):
        """Test with empty list."""
        with self.assertRaises(ValueError):
            std([])

    def test_basic(self):
        """Test with basic list."""
        self.assertEqual(std([75, 450, 18, 597, 27474, 48575]), 18805.551214261766)

    def test_nan(self):
        """Test with float("nan") values in list."""
        self.assertEqual(std([75, 450, 18, 597, float("nan"), 27474, 48575, float("nan")]), 18805.551214261766)


class TestQuartile(unittest.TestCase):
    """Test module for Q1,Q2,Q3 functions."""

    def test_empty_q1(self):
        """Test with empty list."""
        with self.assertRaises(ValueError):
            lower_quartile([])

    def test_empty_q2(self):
        """Test with empty list."""
        with self.assertRaises(ValueError):
            median_quartile([])

    def test_empty_q3(self):
        """Test with empty list."""
        with self.assertRaises(ValueError):
            upper_quartile([])

    def test_quartile_easy(self):
        """Test with basic list."""
        data = [4, 6, 7, 8, 10, 23, 34]
        self.assertEqual(lower_quartile(data), 6)
        self.assertEqual(median_quartile(data), 8)
        self.assertEqual(upper_quartile(data), 23)

    def test_quartile_hard(self):
        """Test with list thats more difficult to compute."""
        data = [23, 13, 37, 16, 26, 35, 26, 35]
        self.assertEqual(lower_quartile(data), 15.25)
        self.assertEqual(median_quartile(data), 26)
        self.assertEqual(upper_quartile(data), 35)

    def test_quartile_nan(self):
        """Test with float("nan") values in list."""
        data = [23, 13, 37, float("nan"), 16, 26, 35, float("nan"), 26, 35]
        self.assertEqual(lower_quartile(data), 15.25)
        self.assertEqual(median_quartile(data), 26)
        self.assertEqual(upper_quartile(data), 35)


class TestIQR(unittest.TestCase):
    """Test module for IQR function."""

    def test_empty_iqr(self):
        """Test with empty list."""
        with self.assertRaises(ValueError):
            iqr([])

    def test_iqr_basic(self):
        """Test with basic list."""
        data = [4, 6, 7, 8, 10, 23, 34]
        self.assertEqual(iqr(data), 17)

    def test_iqr_hard(self):
        """Test with float("nan") values in list."""
        data = [23, 13, 37, 16, 26, 35, 26, 35]
        self.assertEqual(iqr(data), 19.75)

    def test_iqr_nan(self):
        """Test with float("nan") values in list."""
        data = [23, 13, 37, float("nan"), 16, 26, 35, float("nan"), 26, 35]
        self.assertEqual(iqr(data), 19.75)


class TestRange(unittest.TestCase):
    """Test module for d_range function."""

    def test_empty_d_range(self):
        """Test with empty list."""
        with self.assertRaises(ValueError):
            d_range([])

    def test_d_range_basic(self):
        """Test with basic list."""
        data = [4, 6, 7, 8, 10, 23, 34]
        self.assertEqual(d_range(data), 30)

    def test_d_range_hard(self):
        """Test with float("nan") values in list."""
        data = [23, 13, 37, 16, 26, 35, 26, 35]
        self.assertEqual(d_range(data), 24)

    def test_d_range_nan(self):
        """Test with float("nan") values in list."""
        data = [23, 13, 37, float("nan"), 16, 26, 35, float("nan"), 26, 35]
        self.assertEqual(d_range(data), 24)


if __name__ == "__main__":
    unittest.main()
