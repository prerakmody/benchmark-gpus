#!/bin/bash
set -e

# Default Arguments
VOLUME_D=64
VOLUME_H=64
VOLUME_W=64
EPOCHS=2
ITERS=20
INFERENCE_ITERS=20
BATCH_SIZE=2
DEPTH=3
FILTERS=16
CLEANUP=false

# Parse Args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --volume_size) VOLUME_D="$2"; VOLUME_H="$3"; VOLUME_W="$4"; shift 4 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --inference_iters) INFERENCE_ITERS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --cleanup) CLEANUP=true; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

echo "=== 3D GPU Benchmark Setup ==="

# 1. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
else
    echo "uv already installed."
fi

# 2. Create Venv and install
echo "Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv
fi
source .venv/bin/activate

echo "Installing dependencies..."
uv pip install torch numpy tqdm

# 3. Run Benchmark
echo "=== Running Benchmark ==="
python benchmark_3d.py \
    --volume_size $VOLUME_D $VOLUME_H $VOLUME_W \
    --epochs $EPOCHS \
    --iters $ITERS \
    --inference_iters $INFERENCE_ITERS \
    --batch_size $BATCH_SIZE \
    --depth $DEPTH \
    --filters $FILTERS

# 4. Optional Cleanup
if [ "$CLEANUP" = true ]; then
    echo "Cleaning up..."
    rm -rf .venv
    echo "Done."
fi
