# Copyright (c) 2020 Civic Knowledge. This file is licensed under the terms of the
# MIT license included in this distribution as LICENSE

import numpy as np


def vectors_length(v):
    """Return the lengths of an array, where each row is a vector"""
    return np.sqrt(np.sum(np.square(v), axis=1, keepdims=True).astype(float))


def vectors_normalize(v):
    """Normalize an array of vectors, in a 2-d array. """
    return v/vectors_length(v)

def vector_length(v):
    """Return the length of a single vector"""
    return np.sqrt(np.sum(np.square(v)))

def vector_normalize(v):
    """Normalize a single vector"""
    return v/vector_length(v)


def _rand_round(v):
    """Randomly round a value up or down to the nearest integer, with probabilities based on the
    value. Generally, np.mean([rand_round(v) for _ in range(N)]) == v, for large enough N
    """
    if v == 0:
        return v

    f, c = np.floor(v), np.ceil(v)

    try:
        return np.random.choice([f, c], p=[c - v, v - f])
    except:
        print(v, f, c)
        raise

rand_round = np.vectorize(_rand_round)