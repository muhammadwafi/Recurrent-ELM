#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# src » activation_function.py
# ==============================================
# @Author    : Muhammad Wafi <mwafi@mwprolabs.com>
# @Support   : [https://mwprolabs.com]
# @Created   : 04-07-2022
# @Modified  : 04-07-2022 23:17:22 pm
# ----------------------------------------------
# @Copyright (c) 2022 MWprolabs https://mwprolabs.com
#
###

import numpy as np


def linear(h_init: np.ndarray) -> np.ndarray:
    return h_init


def sigmoid(h_init: np.ndarray) -> np.ndarray:
    result = 1 / (1 + np.exp(-h_init))
    return result


def hyperbolic_tanh(h_init: np.ndarray) -> np.ndarray:
    exponent = (1 + np.exp(-h_init))
    result = np.divide((1 - exponent), (1 + exponent))
    return result


def gaussian(x: np.ndarray, weights: np.ndarray, bias: list) -> np.ndarray:
    if not isinstance(x, (np.ndarray, np.generic)):
        raise TypeError("x must be a numpy ndarray!")
    if not isinstance(weights, (np.ndarray, np.generic)):
        raise TypeError("weights must be a numpy ndarray")
    if not isinstance(bias, list):
        raise TypeError("bias must be a list!")
    result = np.exp(((-1) * bias) * np.subtract(x, weights.T))
    return result


def cosine(h_init: np.ndarray) -> np.ndarray:
    result = np.cos(h_init)
    return result


def relu(h_init: np.ndarray) -> np.ndarray:
    return np.maximum(0, h_init)


def leaky_relu(h_init: np.ndarray) -> np.ndarray:
    return np.maximum(0.01 * h_init, h_init)
