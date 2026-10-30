#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
REWARD_FILE="/logs/verifier/reward.txt"
echo "0.000" > "$REWARD_FILE"

# Dynamically locate verifier.py regardless of mount point
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFIER_SCRIPT="$SCRIPT_DIR/verifier.py"

if [[ ! -f "$VERIFIER_SCRIPT" ]]; then
    # Fallback paths used by different Harbor runner configurations
    if [[ -f "/root/tests/verifier.py" ]]; then
        VERIFIER_SCRIPT="/root/tests/verifier.py"
    elif [[ -f "/tests/verifier.py" ]]; then
        VERIFIER_SCRIPT="/tests/verifier.py"
    fi
fi

cd /workspace

# Rebuild workspace with agent changes
if ! su agent -c "cargo build --release"; then
    echo "0.000" > "$REWARD_FILE"
    exit 0
fi

# Execute verifier script and capture score
SCORE=$(python3 "$VERIFIER_SCRIPT" 2>/tmp/verifier.log || echo "0.000")

# Write final score strictly formatted as float
if [[ "$SCORE" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "$SCORE" > "$REWARD_FILE"
else
    echo "0.000" > "$REWARD_FILE"
fi
