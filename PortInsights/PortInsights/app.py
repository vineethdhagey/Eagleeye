from flask import Flask, jsonify
import random

app = Flask(__name__)

# Mock endpoint for congestion info (simulating MarineTraffic JSON)
@app.route("/api/congestion/<portid>")
def mock_congestion(portid):
    data = [
        {
            "PORTID": portid,
            "PORTNAME": "Klaipeda",
            "TIME_ANCH": 1.5,   # median waiting time at anchorage (days)
            "TIME_PORT": 2.3,   # median waiting time at port (days)
            "VESSELS": 12       # number of vessels
        }
    ]
    return jsonify(data)  # Always return a list like MarineTraffic does

def _simulate_vessel_movement(vessel):
    # Slightly adjust LAT and LON by ±0.01 degrees
    lat_drift = random.uniform(-0.01, 0.01)
    lon_drift = random.uniform(-0.01, 0.01)
    vessel = vessel.copy()
    vessel["LAT"] = round(vessel["LAT"] + lat_drift, 5)
    vessel["LON"] = round(vessel["LON"] + lon_drift, 5)
    # Optionally adjust SOG slightly (except for stationary vessels)
    if vessel["SOG"] > 0:
        sog_drift = random.uniform(-0.2, 0.2)
        vessel["SOG"] = max(0, round(vessel["SOG"] + sog_drift, 2))
    return vessel

# Mock endpoint for vessels info (simulating MarineTraffic JSON)
@app.route("/api/vessels/<portid>")
def mock_vessels(portid):
    vessels = [
        {
            "MMSI": "123456789",
            "SHIPNAME": "Baltic Trader",
            "LAT": 55.71,
            "LON": 21.12,
            "SOG": 12.3,
            "STATUS": "Under way"
        },
        {
            "MMSI": "987654321",
            "SHIPNAME": "Nordic Carrier",
            "LAT": 55.70,
            "LON": 21.15,
            "SOG": 0.0,
            "STATUS": "At anchor"
        }
    ]
    moved_vessels = [_simulate_vessel_movement(v) for v in vessels]
    return jsonify(moved_vessels)

# Optionally, an endpoint for all vessels (not filtered by portid)
@app.route("/api/vessels/")
def mock_vessels_all():
    vessels = [
        {
            "MMSI": "123456789",
            "SHIPNAME": "Baltic Trader",
            "LAT": 55.71,
            "LON": 21.12,
            "SOG": 12.3,
            "STATUS": "Under way"
        },
        {
            "MMSI": "987654321",
            "SHIPNAME": "Nordic Carrier",
            "LAT": 55.70,
            "LON": 21.15,
            "SOG": 0.0,
            "STATUS": "At anchor"
        }
    ]
    moved_vessels = [_simulate_vessel_movement(v) for v in vessels]
    return jsonify(moved_vessels)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)