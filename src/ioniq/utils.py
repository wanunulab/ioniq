#!/usr/bin/env python
"""
Utility functions and classes for internal and external use
"""

import numpy as np


class Singleton(type):
    """
    Generic singleton metaclass.

    This metaclass ensures the same return every time the class is instanced.
    """

    def __init__(self, *args, **kwargs):
        """

        :param args: Arguments for the class
        :param kwargs: Keyword arguments for the class
        """
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Returns the Singleton instance.

        It creates the instance if it doesn't exist the instance does not exist.
        If the instance exists, it is returned as-is.

        :param args: Arguments for the class constructor.
        :param kwargs: Keyword arguments for the class constructor.
        :return: The Singleton instance.
        :rtype: object
        """
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kwargs)
        return self.__instance


def split_voltage_steps(voltage: np.ndarray, n_remove=0, as_tuples=False):
    """
    Split the voltage signal into steps.

    This function splits the signal at the points where the voltage changes.

    :param voltage: The voltage signal.
    :type voltage: numpy.ndarray
    :param n_remove: Number of points to remove from the start of each split, defaults to 0.
    :type n_remove: int
    :param as_tuples: Whether to return the splits as tuples of start and end indices, defaults=False.
    :type as_tuples: bool
    :raises ValueError: If `n_remove` is negative or larger than the start of the voltage changes.
    :return: Start and end indices of the splits.
    :rtype: tuple of numpy.ndarray or list of tuple if as_tuples=True
    """
    # Check if the current or voltage arrays are empty
    if not voltage.size:
        return []
    # Check if the n_remove argument is negative
    if n_remove < 0:
        raise ValueError("n_remove must be non-negative")
    # Find the indices at which the voltage level changes
    split_indices = np.where(voltage[:-1] != voltage[1:])[0] + 1
    # Add the start and end indices of the current array to the split indices
    split_indices = np.concatenate([[0], split_indices, [len(voltage)]])
    # Calculate the start and end indices of the splits
    start_indices = split_indices[:-1] + n_remove
    end_indices = split_indices[1:]
    # Check if n_remove is not too large
    if np.any(end_indices <= start_indices):
        raise ValueError("n_remove is too large")
    if not as_tuples:
        return start_indices, end_indices
    # Optional: return list of tuples of start and end
    return [(start_ind, end_ind) for start_ind, end_ind in zip(start_indices, end_indices)]
