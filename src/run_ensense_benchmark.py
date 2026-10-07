import subprocess
import time
import pandas as pd
import os

MODEL_PATH = "/home/ayush/depth1-sensitivity/models/trained/senspb_depth3_model.json"
OUTPUT_CSV = "/home/ayush/depth1-sensitivity/results/csv/ensense_milp_tool_readings.csv"
PRECISION = 5000
OUTPUT_GAP = ["0.49", "0.51"]

feature_queries = [
    [0], [1], [2], [3], [4], [5], [6], [7],
    [0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [0,7],
    [0,1,2], [1,2,3], [2,3,4], [3,4,5],
    [0,1,2,3], [4,5,6,7],
]

rows = []
for query_id, features in enumerate(feature_queries):
    features_str = [str(f) for f in features]
    cmd = ["python", "sensitive.py", MODEL_PATH,
           "--solver", "milp",
           "--output_gap"] + OUTPUT_GAP + [
           "--precision", str(PRECISION),
           "--features"] + features_str

    print(f"[{query_id}] Running sensitive_features={features} ...")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    runtime = time.time() - start

    stdout = result.stdout
    status = "GLOBALLY SENSITIVE" if "Sensitive sample" in stdout else "ROBUST (Globally Insensitive)"

    row = {
        "query_id": query_id,
        "sensitive_features": str(features),
        "num_sensitive_features": len(features),
        "status": status,
        "runtime_ms": round(runtime * 1000, 4),
        "raw_stdout": stdout.strip()
    }
    rows.append(row)
    print(f"    -> status={status}, runtime={row['runtime_ms']} ms")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(rows)} readings to {OUTPUT_CSV}")