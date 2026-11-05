import math
import numpy as np
from particledrive.filter import (make_particles, predict, move, estimate,
    weights_for_measurements, measure, systematic_resample, wrap)
from particledrive.simulation import Simulation


def test_move_in_straight_line():
    assert np.allclose(move([10, 20, 0], 5, 0), [15, 20, 0])


def test_wrap_angle():
    assert np.isclose(wrap(3 * math.pi), -math.pi)


def test_particle_count_preserved():
    rng = np.random.default_rng(5)
    particles = make_particles(500, rng)
    predicted = predict(particles, 3, 0.1, rng)
    weighted = np.full(500, 1 / 500)
    assert systematic_resample(predicted, weighted, rng).shape == (500, 3)


def test_measurement_prefers_correct_position():
    rng = np.random.default_rng(2)
    pose = np.array([120., 140., 0.])
    observation = measure(pose, rng, 0)
    particles = np.array([[120., 140., 0.], [700., 440., 0.]])
    weights = weights_for_measurements(particles, observation, 10)
    assert weights[0] > weights[1]
    assert np.isclose(sum(weights), 1)


def test_heading_average_wraps_correctly():
    poses = np.array([[0, 0, 3.13], [0, 0, -3.13]])
    assert abs(estimate(poses)[2]) > 3


def test_simulation_steps_and_records_error():
    sim = Simulation(count=200)
    state = sim.step()
    assert state['tick'] == 1
    assert len(state['particles']) == 200
    assert len(state['history']) == 1
    assert state['error'] >= 0


def test_sensor_dropout():
    sim = Simulation(count=200, sensors=False)
    assert sim.step()['observations'] == []


def test_kidnap_changes_true_position_not_particles():
    sim = Simulation(count=200)
    particles_before = sim.particles.copy()
    sim.kidnap()
    assert np.array_equal(sim.particles, particles_before)
    assert not np.allclose(sim.true[:2], [125, 145])
