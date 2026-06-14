#!/usr/bin/env bash
# create_specter2_kb.sh
#
# Creates the scifact_specter2 KB using allenai/specter2_base (768-dim).
#
# Steps:
#   1. Patch embedding_model in config.yml to allenai/specter2_base
#   2. Restart the server (kills the running instance, starts fresh)
#   3. Run ingest_corpus.py --corpus specter2 from perspicacite_eval
#   4. Restore original embedding_model in config.yml
#   5. Restart server again with the original model
#
# Prerequisites:
#   - perspicacite_eval repo at ../perspicacite_eval (or set EVAL_REPO env var)
#   - corpus.jsonl present at perspicacite_eval/data/scifact/corpus.jsonl
#   - config.yml present in the Perspicacite-AI repo root
#   - A running Perspicacite server (the script will restart it)
#
# Usage:
#   cd /path/to/Perspicacite-AI
#   bash scripts/create_specter2_kb.sh
#
# To keep the server running with specter2_base permanently, comment out the
# "restore" section at the bottom and do not restart after ingest.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$REPO_DIR/config.yml"
EVAL_REPO="${EVAL_REPO:-$(dirname "$REPO_DIR")/perspicacite_eval}"
INGEST_SCRIPT="$EVAL_REPO/scripts/ingest_corpus.py"
SPECTER2_KB="${SCIFACT_SPECTER2_KB:-scifact_specter2}"
SERVER_PORT="${PERSPICACITE_PORT:-8000}"
SERVER_URL="${PERSPICACITE_URL:-http://localhost:$SERVER_PORT}"
PIDFILE="/tmp/perspicacite_server.pid"
LOG_FILE="/tmp/perspicacite_specter2_ingest.log"

# ---- Helpers ----------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }

server_running() {
    curl -sf "$SERVER_URL/api/health" >/dev/null 2>&1
}

stop_server() {
    if [ -f "$PIDFILE" ]; then
        PID="$(cat "$PIDFILE")"
        if kill -0 "$PID" 2>/dev/null; then
            echo "  Stopping server (PID $PID)…"
            kill "$PID"
            local tries=0
            while kill -0 "$PID" 2>/dev/null && [ $tries -lt 20 ]; do
                sleep 0.5; tries=$((tries + 1))
            done
            kill -0 "$PID" 2>/dev/null && kill -9 "$PID" || true
        fi
        rm -f "$PIDFILE"
    else
        # Fallback: kill any uvicorn/perspicacite on the port
        pkill -f "perspicacite.*serve" 2>/dev/null || true
        pkill -f "uvicorn.*perspicacite" 2>/dev/null || true
        sleep 1
    fi
}

start_server() {
    echo "  Starting server (config: $CONFIG, port: $SERVER_PORT)…"
    cd "$REPO_DIR"
    uv run perspicacite -c "$CONFIG" serve &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$PIDFILE"
    echo "  Waiting for server to be ready (PID $SERVER_PID)…"
    local tries=0
    until server_running || [ $tries -gt 40 ]; do
        sleep 2; tries=$((tries + 1))
    done
    server_running || die "Server did not become healthy after 80 s"
    echo "  Server is up."
}

patch_config() {
    local model="$1"
    echo "  Patching config.yml: embedding_model → $model"
    # Replace the embedding_model value (handles quotes)
    sed -i.bak "s|embedding_model:.*|embedding_model: \"$model\"|" "$CONFIG"
}

restore_config() {
    if [ -f "$CONFIG.bak" ]; then
        echo "  Restoring config.yml from backup…"
        mv "$CONFIG.bak" "$CONFIG"
    fi
}

# ---- Preflight checks -------------------------------------------------------

echo "=== SciFact SPECTER2 KB creation ==="
echo "  Repo:      $REPO_DIR"
echo "  Config:    $CONFIG"
echo "  Eval repo: $EVAL_REPO"
echo "  KB name:   $SPECTER2_KB"
echo "  Log:       $LOG_FILE"
echo ""

[ -f "$CONFIG" ] || die "config.yml not found at $CONFIG"
[ -f "$INGEST_SCRIPT" ] || die "ingest_corpus.py not found at $INGEST_SCRIPT"
[ -f "$EVAL_REPO/data/scifact/corpus.jsonl" ] || \
    die "corpus.jsonl not found; run perspicacite_eval/scripts/download_scifact.py first"

# Confirm SPECTER2 can load
echo "Step 0: Verifying allenai/specter2_base loads…"
cd "$REPO_DIR"
uv run python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('allenai/specter2_base')
dim = m.get_embedding_dimension() if hasattr(m, 'get_embedding_dimension') else m.get_sentence_embedding_dimension()
assert dim == 768, f'Unexpected dim {dim}'
print(f'  allenai/specter2_base loaded OK (dim={dim})')
" 2>&1 | grep -v "^Warning\|Loading weights\|FutureWarning\|it/s\|Downloading" || \
    die "allenai/specter2_base failed to load — check sentence-transformers installation"

# ---- Step 1: Patch config ---------------------------------------------------

echo "Step 1: Patching config.yml with allenai/specter2_base…"
ORIGINAL_MODEL="$(grep -oP '(?<=embedding_model: )["'"'"']?[^"'"'"'\n]+["'"'"']?' "$CONFIG" | tr -d '"'"'")"
echo "  Original model: $ORIGINAL_MODEL"
patch_config "allenai/specter2_base"

# ---- Step 2: Restart server -------------------------------------------------

echo "Step 2: Restarting server with specter2_base…"
stop_server
start_server

# ---- Step 3: Ingest ---------------------------------------------------------

echo "Step 3: Ingesting SciFact abstracts into $SPECTER2_KB…"
cd "$EVAL_REPO"
uv run python scripts/ingest_corpus.py \
    --corpus specter2 \
    --kb-name "$SPECTER2_KB" \
    2>&1 | tee "$LOG_FILE"

echo "  Ingest finished. Check $LOG_FILE for details."

# ---- Step 4: Restore config -------------------------------------------------

echo "Step 4: Restoring original embedding_model ($ORIGINAL_MODEL)…"
stop_server
restore_config

echo "Step 5: Restarting server with original model…"
cd "$REPO_DIR"
start_server

echo ""
echo "=== Done ==="
echo "  KB '$SPECTER2_KB' is ready with allenai/specter2_base (768-dim) embeddings."
echo "  Server is running with the original embedding_model ($ORIGINAL_MODEL)."
echo "  NOTE: the two KBs use different embedding models and CANNOT be queried together"
echo "  via multi-KB mode (check_embedding_compat will reject them)."
