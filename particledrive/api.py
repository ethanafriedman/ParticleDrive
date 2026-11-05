"""Small FastAPI layer; localization logic lives in filter.py."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
from .simulation import Simulation

app = FastAPI(title="ParticleDrive", description="2D particle-filter localization demo")
simulation = Simulation()

class Settings(BaseModel):
    particles: int = Field(default=700, ge=100, le=2000)
    noise: float = Field(default=14, ge=1, le=70)
    sensors: bool = True

@app.get('/api/state')
def state():
    return simulation.snapshot()

@app.post('/api/step')
def step():
    return simulation.step()

@app.post('/api/reset')
def reset(settings: Settings):
    global simulation
    simulation = Simulation(settings.particles, settings.noise, settings.sensors)
    return simulation.snapshot()

@app.post('/api/kidnap')
def kidnap():
    return simulation.kidnap()

@app.post('/api/sensors')
def toggle_sensor(enabled: bool = True):
    simulation.sensors = enabled
    return simulation.snapshot()

app.mount('/', StaticFiles(directory=Path(__file__).resolve().parents[1] / 'web', html=True), name='web')
