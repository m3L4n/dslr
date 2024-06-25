"""Stats lib implemented for dslr."""

from typing import Any


def remove_null(data: list[Any]) -> list[Any]:
    """Remove all None entries in a list.

    remove_null(data: list[Any]) -> list[Any]
    """
    return [x for x in data if x]


def mean(data: list[float]) -> float:
    """Return the mean of data.

    mean(data: list[float]) -> float
    """
    data = remove_null(data)
    if len(data) == 0:
        raise ValueError("List empty")
    return sum(data) / len(data)


def median(data: list[float]) -> float | int:
    """Return the median of data.

    median(data: list[float]) -> float
    """
    if len(data) == 0:
        raise ValueError("List empty")
    data = sorted(remove_null(data))
    data_len = len(data)
    data = sorted(data)
    if data_len % 2 != 0:
        return data[(data_len - 1) // 2]
    med1 = data[(data_len - 1) // 2]
    med2 = data[((data_len - 1) // 2) + 1]
    return (med1 + med2) / 2


def count(data: list[float]) -> float:
    """Return the number of non null element in list.

    count(data: list[float]) -> int
    """
    data = remove_null(data)
    return len(data)


def max(data: list[Any]) -> Any:
    """Return the bigger value in the list.

    max(data: list[Any]) -> Any
    """
    data = remove_null(data)
    if len(data) == 0:
        raise ValueError("List empty")
    max = float("-inf")
    for x in data:
        if x > max:
            max = x
    return max


def min(data: list[Any]) -> Any:
    """Return the bigger value in the list.

    min(data: list[Any]) -> Any
    """
    data = remove_null(data)
    if len(data) == 0:
        raise ValueError("List empty")
    max = float("inf")
    for x in data:
        if x < max:
            max = x
    return max


def std(data: list[float]) -> float:
    """Return the standard deviation of data.

    std(data: list[float]) -> float
    """
    data = remove_null(data)
    if len(data) == 0:
        raise ValueError("List empty")
    sample_mean = mean(data)
    score = [(x - sample_mean) ** 2 for x in data]
    total_score = sum(score)
    return total_score / (len(data))


def lower_quartile(data: list[float]) -> float:
    """Return the Q1 of data.

    lower_quartile(data: list[float]) -> float
    """
    data = sorted(remove_null(data))
    if len(data) == 0:
        raise ValueError("List empty")
    data_len = len(data)
    if (data_len + 1) % 4 == 0:
        return data[data_len // 4]
    first_term = data[(data_len // 4) - 1]
    second_term = data[data_len // 4]
    third_term = data[(data_len // 4) + 1]
    return first_term + 0.25 * (second_term - third_term)


def median_quartile(data: list[float]) -> float:
    """Return the Q2 of data.

    median_quartile(data: list[float]) -> float
    """
    data = sorted(remove_null(data))
    if len(data) == 0:
        raise ValueError("List empty")
    data_len = len(data)
    if (data_len + 1) % 2 == 0:
        return data[data_len // 2]
    fourth_term = data[(data_len // 2) - 1]
    fifth_term = data[data_len // 2]
    return fourth_term + 0.50 * (fifth_term - fourth_term)


def upper_quartile(data: list[float]) -> float:
    """Return the Q3 of data.

    upper_quartile(data: list[float]) -> float
    """
    data = sorted(remove_null(data))
    if len(data) == 0:
        raise ValueError("List empty")
    data_len = len(data)
    if (3 * (data_len + 1)) % 4 == 0:
        return data[(3 * data_len) // 4]
    sixth_term = data[(3 * data_len) // 4 - 1]
    seventh_term = data[(3 * data_len) // 4]
    return sixth_term + 0.75 * (seventh_term - sixth_term)
