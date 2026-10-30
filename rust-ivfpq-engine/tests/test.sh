#!/bin/bash
set -euo pipefail

export PATH="/usr/local/cargo/bin:$PATH"

mkdir -p /logs/verifier
REWARD_FILE="/logs/verifier/reward.txt"
echo "0.000" > "$REWARD_FILE"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFIER_SCRIPT="$SCRIPT_DIR/verifier.py"

if [[ ! -f "$VERIFIER_SCRIPT" ]]; then
    if [[ -f "/root/tests/verifier.py" ]]; then
        VERIFIER_SCRIPT="/root/tests/verifier.py"
    elif [[ -f "/tests/verifier.py" ]]; then
        VERIFIER_SCRIPT="/tests/verifier.py"
    fi
fi

cd /workspace

if ! su agent -c "PATH=/usr/local/cargo/bin:$PATH cargo build --release"; then
    echo "0.000" > "$REWARD_FILE"
    exit 0
fi

SCORE=$(python3 "$VERIFIER_SCRIPT" 2>/tmp/verifier.log || echo "0.000")

if [[ "$SCORE" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "$SCORE" > "$REWARD_FILE"
else
    echo "0.000" > "$REWARD_FILE"
fi
