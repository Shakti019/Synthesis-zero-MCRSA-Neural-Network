#!/bin/bash
set -e

echo "=========================================================="
echo " SYNTHESIS-ZERO: Single Container Development Environment "
echo "=========================================================="

cd /workspace

# If node_modules is missing or empty in container volume, install dependencies
if [ ! -d "/workspace/frontend/node_modules/next" ]; then
  echo "[Init] Installing Next.js dependencies in container volume..."
  cd /workspace/frontend && npm install
  cd /workspace
fi

# Graceful shutdown handler
cleanup() {
  echo ""
  echo "[Synthesis-Zero] Terminating services..."
  kill -TERM "$PYTHON_PID" "$NEXT_PID" 2>/dev/null || true
  wait "$PYTHON_PID" 2>/dev/null || true
  wait "$NEXT_PID" 2>/dev/null || true
  exit 0
}
trap cleanup SIGTERM SIGINT

# 1. Start Python AI / GNN Backend (Auto-reloads on file change)
echo "[1/2] Starting Python Quantum GNN Engine (Port 5000) with auto-reload..."
PORT=5000 FLASK_DEBUG=1 python -u app.py &
PYTHON_PID=$!

# 2. Start Next.js Development Server (Hot-reloads on file change)
echo "[2/2] Starting Next.js Dev Server (Port 3000) with Hot-Reload..."
cd /workspace/frontend
npm run dev -- -H 0.0.0.0 -p 3000 &
NEXT_PID=$!

echo ""
echo "=========================================================="
echo " BOTH SERVICES LIVE IN SINGLE CONTAINER!"
echo " - Next.js Solar Lab & Mobile Node: http://localhost:3000"
echo " - Python GNN Backend API:          http://localhost:5000"
echo " Live file sync & hot-reloading active on both apps."
echo "=========================================================="

wait -n "$PYTHON_PID" "$NEXT_PID"

