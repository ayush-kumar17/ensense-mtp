#!/bin/bash
set -e

MODEL="$1"
LGAP="$2"
UGAP="$3"
PRECISION="$4"
LOG="$5"
DATA="$6"

if [ -z "$MODEL" ] || [ -z "$LGAP" ] || [ -z "$UGAP" ] || [ -z "$PRECISION" ] || [ -z "$LOG" ] || [ -z "$DATA" ]; then
    echo "Usage: $0 <model_path> <lgap> <ugap> <precision> <output_log_path> <data_file>"
    exit 1
fi

feature_sets=(
    "0" "1" "2" "3" "4" "5" "6" "7"
    "0 1" "1 2" "2 3" "3 4" "4 5" "5 6" "6 7" "0 7"
    "0 1 2" "1 2 3" "2 3 4" "3 4 5"
    "0 1 2 3" "4 5 6 7"
)

mkdir -p "$(dirname "$LOG")"
> "$LOG"

cd ~/ensense/src

for i in "${!feature_sets[@]}"; do
    fs="${feature_sets[$i]}"
    echo "=== Query $i: features $fs ===" | tee -a "$LOG"
    python sensitive.py "$MODEL" --solver milp --all_opt --prob --output_gap "$LGAP" "$UGAP" --precision "$PRECISION" --features $fs --data_file "$DATA" 2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

echo "Done. Full log saved to $LOG"
