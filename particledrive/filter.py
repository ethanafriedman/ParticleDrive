"""Particle filter for a car on a 2D map.

Each particle is a possible pose (x, y, heading). The three core steps are
predict -> weight -> resample. This module contains no web/UI code.
"""
from __future__ import annotations

import math
import numpy as np

WIDTH, HEIGHT = 900, 560
LANDMARKS = np.array([
    [90, 90], [300, 80], [540, 95], [790, 90],
    [110, 285], [460, 265], [790, 290],
    [90, 475], [340, 470], [605, 455], [805, 475],
], dtype=float)


def wrap(angle):
    """Keep angles between -pi and +pi."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def move(pose, distance, turn):
    """Move one pose or an (N, 3) array of poses."""
    result = np.array(pose, dtype=float, copy=True)
    result[..., 2] = wrap(result[..., 2] + turn)
    result[..., 0] += distance * np.cos(result[..., 2])
    result[..., 1] += distance * np.sin(result[..., 2])
    return result


def predict(particles, distance, turn, rng, motion_noise=0.12, turn_noise=0.045):
    """Apply noisy odometry to every particle."""
    count = len(particles)
    noisy_distance = distance + rng.normal(0, motion_noise * max(1, abs(distance)), count)
    noisy_turn = turn + rng.normal(0, turn_noise, count)
    result = particles.copy()
    result[:, 2] = wrap(result[:, 2] + noisy_turn)
    result[:, 0] += noisy_distance * np.cos(result[:, 2])
    result[:, 1] += noisy_distance * np.sin(result[:, 2])
    return result


def measure(pose, rng, noise):
    """Noisy distance from the real car to every fixed landmark."""
    distances = np.linalg.norm(LANDMARKS - np.asarray(pose)[:2], axis=1)
    return distances + rng.normal(0, noise, len(LANDMARKS))


def weights_for_measurements(particles, observations, noise):
    """Likelihood of each particle given distance readings.

    Use log likelihoods to avoid multiplying many tiny probabilities.
    """
    predicted = np.linalg.norm(particles[:, None, :2] - LANDMARKS[None, :, :], axis=2)
    residual = (predicted - observations[None, :]) / max(noise, 0.5)
    log_weights = -0.5 * np.sum(residual ** 2, axis=1)
    log_weights -= log_weights.max()  # Numerically stable exponentiation
    weights = np.exp(log_weights)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(particles), 1 / len(particles))
    return weights / total


def systematic_resample(particles, weights, rng):
    """Resample with one random offset, then equally spaced thresholds."""
    n = len(particles)
    thresholds = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    indices = np.searchsorted(cumulative, thresholds, side='left')
    return particles[np.clip(indices, 0, n - 1)].copy()


def estimate(particles, weights=None):
    """Average x/y and use a circular mean for heading."""
    if weights is None:
        weights = np.full(len(particles), 1 / len(particles))
    x = np.average(particles[:, 0], weights=weights)
    y = np.average(particles[:, 1], weights=weights)
    angle = math.atan2(np.sum(weights * np.sin(particles[:, 2])),
                       np.sum(weights * np.cos(particles[:, 2])))
    return np.array([x, y, angle])


def make_particles(count, rng, center=None):
    """Initially broad uncertainty; optionally center around a known pose."""
    if center is None:
        xy = np.column_stack((rng.uniform(20, WIDTH - 20, count),
                              rng.uniform(20, HEIGHT - 20, count)))
        theta = rng.uniform(-np.pi, np.pi, count)
    else:
        xy = rng.normal(np.asarray(center)[:2], [95, 95], (count, 2))
        theta = wrap(rng.normal(center[2], .7, count))
    return np.column_stack((xy, theta))
