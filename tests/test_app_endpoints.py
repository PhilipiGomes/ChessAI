import json
import os
import time
import importlib

# import the Flask app from the top-level app.py
flask_app = importlib.import_module('app').app


def test_train_status():
    client = flask_app.test_client()
    r = client.get('/api/train_status')
    assert r.status_code == 200
    d = r.get_json()
    assert 'running' in d


def test_models_list():
    client = flask_app.test_client()
    r = client.get('/api/models')
    assert r.status_code == 200
    d = r.get_json()
    assert 'models' in d


def test_new_game_and_ai_move():
    client = flask_app.test_client()
    # start new pve game with small delay
    r = client.post('/api/new_game', json={'mode': 'pve', 'ai_depth': 1, 'ai_delay': 0.01})
    assert r.status_code == 200
    data = r.get_json()
    assert 'fen' in data
    # request an AI move
    r2 = client.post('/api/ai_move', json={})
    # may succeed or return game over; ensure response format
    assert r2.status_code in (200, 400)


def test_eve_start_stop():
    client = flask_app.test_client()
    r = client.post('/api/eve_start', json={'ai1_depth': 1, 'ai2_depth': 1, 'ai_delay': 0.01})
    assert r.status_code == 200
    # give it a brief moment to start
    time.sleep(0.05)
    r2 = client.post('/api/eve_stop')
    assert r2.status_code == 200
