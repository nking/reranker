import json

with open("pipeline.log") as f:
    parity_rows = [
        json.loads(line.split("[PARITY_JSON] ", 1)[1])
        for line in f
        if "[PARITY_JSON] " in line
    ]
