#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# ─── Mode selection ─────────────────────────────────────────────────────────
MODE="${1:-app}"
shift 2>/dev/null || true

case "$MODE" in
    backend)
        echo "Starting FastAPI backend on port 8199..."
        uvicorn backend.main:app --host 0.0.0.0 --port 8199 --reload
        ;;
    app)
        # Start backend in background, then launch macOS app
        echo "Starting FastAPI backend on port 8199 (background)..."
        uvicorn backend.main:app --host 0.0.0.0 --port 8199 &
        BACKEND_PID=$!
        echo "Backend PID: $BACKEND_PID"

        # Wait for backend to be ready
        echo "Waiting for backend to start..."
        for i in $(seq 1 30); do
            if curl -s http://localhost:8199/api/health > /dev/null 2>&1; then
                echo "Backend is ready!"
                break
            fi
            sleep 1
        done

        # Build and launch the macOS app
        APP_PATH="$SCRIPT_DIR/GemmaSAMApp"
        BUILD_DIR="$HOME/Library/Developer/Xcode/DerivedData/GemmaSAMApp-hgnisgageztrfcdgojljsclsvdtl/Build/Products/Debug/GemmaSAMApp.app"

        echo "Building macOS app..."
        (cd "$APP_PATH" && xcodebuild -project GemmaSAMApp.xcodeproj -scheme GemmaSAMApp -configuration Debug build > /dev/null 2>&1)

        echo "Launching GemmaSAMApp..."
        open "$BUILD_DIR"

        echo ""
        echo "Both services running:"
        echo "   Backend: http://localhost:8199"
        echo "   macOS App: GemmaSAMApp"
        echo ""
        echo "Press Ctrl+C to stop both..."

        # Wait for Ctrl+C
        wait $BACKEND_PID
        ;;
    *)
        echo "Usage: $0 {app|backend}"
        echo ""
        echo "  app     - Launch backend + macOS app (default)"
        echo "  backend - Start only the FastAPI backend"
        exit 1
        ;;
esac