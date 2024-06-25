import unittest
from random import shuffle

from stats import count, lower_quartile, max, mean, median, median_quartile, min, std, upper_quartile


class TestMean(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(ValueError):
            mean([])

    def test_same(self):
        self.assertEqual(mean([5, 5, 5, 5, 5, 5, 5, 5]), 5)

    def test_basic(self):
        self.assertEqual(mean([1, 2, 3, 4, 5]), 3)

    def test_none(self):
        self.assertEqual(mean([1, 2, 3, None, 4, 5]), 3)


class TestMedian(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(ValueError):
            median([])

    def test_same(self):
        self.assertEqual(median([6, 6, 6, 6, 6, 6, 6, 6]), 6)

    def test_odd(self):
        self.assertEqual(median([1, 2, 4, 5, 6]), 4)

    def test_even(self):
        self.assertEqual(median([1, 2, 3, 4]), 2.5)

    def test_none(self):
        self.assertEqual(mean([1, 2, 3, None, 4]), 2.5)


class TestCount(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(count([]), 0)

    def test_basic(self):
        self.assertEqual(count([1, 2, 3, 4, 5]), 5)

    def test_none(self):
        self.assertEqual(count([1, 2, None, 3, 4, None, 5]), 5)


class TestMax(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(ValueError):
            max([])

    def test_sorted(self):
        self.assertEqual(max([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 10)

    def test_none(self):
        self.assertEqual(max([1, 2, 3, 4, None, 5, 6, 7, 8, None, 9, 10]), 10)

    def test_shuffle(self):
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        shuffle(lst)
        self.assertEqual(max(lst), 10)


class TestMin(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(ValueError):
            min([])

    def test_sorted(self):
        self.assertEqual(min([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 1)

    def test_none(self):
        self.assertEqual(min([1, 2, 3, 4, None, 5, 6, 7, 8, None, 9, 10]), 1)

    def test_shuffle(self):
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]
        shuffle(lst)
        self.assertEqual(min(lst), 1)


class TestStd(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(ValueError):
            std([])

    def test_basic(self):
        self.assertEqual(std([75, 450, 18, 597, 27474, 48575]), 353648756.47222215)

    def test_none(self):
        self.assertEqual(std([75, 450, 18, 597, None, 27474, 48575, None]), 353648756.47222215)


class TestQuartile(unittest.TestCase):
    def test_empty_q1(self):
        with self.assertRaises(ValueError):
            lower_quartile([])

    def test_empty_q2(self):
        with self.assertRaises(ValueError):
            median_quartile([])

    def test_empty_q3(self):
        with self.assertRaises(ValueError):
            upper_quartile([])

    def test_quartile_easy(self):
        data = [4, 6, 7, 8, 10, 23, 34]
        self.assertEqual(lower_quartile(data), 6)
        self.assertEqual(median_quartile(data), 8)
        self.assertEqual(upper_quartile(data), 23)

    def test_quartile_hard(self):
        data = [23, 13, 37, 16, 26, 35, 26, 35]
        self.assertEqual(lower_quartile(data), 15.25)
        self.assertEqual(median_quartile(data), 26)
        self.assertEqual(upper_quartile(data), 35)

    def test_quartile_none(self):
        data = [23, 13, 37, None, 16, 26, 35, None, 26, 35]
        self.assertEqual(lower_quartile(data), 15.25)
        self.assertEqual(median_quartile(data), 26)
        self.assertEqual(upper_quartile(data), 35)


if __name__ == "__main__":
    unittest.main()
