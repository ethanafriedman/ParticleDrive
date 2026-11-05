"""Simulation orchestration: an actual car, odometry, and a particle estimate."""
from __future__ import annotations

import math
import numpy as np
from .filter import (WIDTH, HEIGHT, LANDMARKS, make_particles, move, predict,
                     measure, weights_for_measurements, systematic_resample, estimate, wrap)

# Waypoints trace a looping road route. Purely simulated, not a driving model.
WAYPOINTS = np.array([(125, 145), (755, 145), (755, 405), (150, 405)])


class Simulation:
    def __init__(self, count=700, noise=14, sensors=True, seed=7):
        self.rng = np.random.default_rng(seed)
        self.count = int(np.clip(count, 100, 2000))
        self.noise = float(np.clip(noise, 1, 70))
        self.sensors = bool(sensors)
        self.true = np.array([125., 145., 0.])
        self.dead = self.true.copy()
        self.particles = make_particles(self.count, self.rng, self.true)
        self.best = estimate(self.particles)
        self.destination = 1
        self.tick = 0
        self.history = []
        self.observations = []

    def step(self):
        """Move the real car, simulate noisy wheel odometry and sensor readings."""
        target = WAYPOINTS[self.destination]
        delta = target - self.true[:2]
        if np.linalg.norm(delta) < 17:
            self.destination = (self.destination + 1) % len(WAYPOINTS)
            target = WAYPOINTS[self.destination]
            delta = target - self.true[:2]
        desired = math.atan2(delta[1], delta[0])
        turn = float(np.clip(wrap(desired - self.true[2]), -.13, .13))
        distance = 4.8
        self.true = move(self.true, distance, turn)

        # Wheel odometry is a shared noisy input to dead reckoning and filter.
        odometry_distance = distance + self.rng.normal(0, .45)
        odometry_turn = turn + self.rng.normal(0, .025)
        self.dead = move(self.dead, odometry_distance, odometry_turn)
        self.particles = predict(self.particles, odometry_distance, odometry_turn, self.rng)

        if self.sensors:
            readings = measure(self.true, self.rng, self.noise)
            self.observations = readings.tolist()
            weights = weights_for_measurements(self.particles, readings, self.noise)
            self.particles = systematic_resample(self.particles, weights, self.rng)
            # A few fresh proposals help recover after unexpected relocation.
            n_new = max(2, self.count // 30)
            self.particles[:n_new] = make_particles(n_new, self.rng)
        else:
            self.observations = []

        self.best = estimate(self.particles)
        self.tick += 1
        err = float(np.linalg.norm(self.best[:2] - self.true[:2]))
        drift = float(np.linalg.norm(self.dead[:2] - self.true[:2]))
        self.history.append([self.tick, round(err, 2), round(drift, 2)])
        self.history = self.history[-240:]
        return self.snapshot()

    def kidnap(self):
        """Teleport the real car without telling the filter or odometry."""
        spots = [(220., 390., 0.), (660., 145., math.pi), (730., 345., -math.pi / 2)]
        self.true = np.array(spots[(self.tick // 17) % len(spots)])
        self.destination = int(np.argmin(np.linalg.norm(WAYPOINTS - self.true[:2], axis=1)))
        self.observations = []
        return self.snapshot()

    def snapshot(self):
        return {
            'true': self.true.tolist(), 'estimated': self.best.tolist(),
            'dead': self.dead.tolist(), 'particles': self.particles[:, :2].round(1).tolist(),
            'landmarks': LANDMARKS.tolist(), 'waypoints': WAYPOINTS.tolist(),
            'history': self.history, 'observations': self.observations,
            'tick': self.tick, 'count': self.count, 'noise': self.noise,
            'sensors': self.sensors,
            'error': round(float(np.linalg.norm(self.best[:2] - self.true[:2])), 1),
            'drift': round(float(np.linalg.norm(self.dead[:2] - self.true[:2])), 1)
        }
