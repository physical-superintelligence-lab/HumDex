#!/bin/bash

# Quick publish script for macOS/Linux
echo ""
echo "========================================"
echo "  Video2Act Website Publishing Script"
echo "========================================"
echo ""

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "[Step 1/3] Committing changes to source code..."
    git add .
    read -p "Enter commit message: " commit_msg
    git commit -m "$commit_msg"
    git push origin main
    echo "[OK] Source code pushed to main branch"
    echo ""
else
    echo "[Step 1/3] No changes to commit, skipping..."
    echo ""
fi

echo "[Step 2/3] Building website..."
npm run build
if [ $? -ne 0 ]; then
    echo "[ERROR] Build failed!"
    exit 1
fi
echo "[OK] Build completed"
echo ""

echo "[Step 3/3] Deploying to GitHub Pages..."
npx gh-pages -d dist
if [ $? -ne 0 ]; then
    echo "[ERROR] Deployment failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "  Successfully deployed!"
echo "  Visit: https://video2act.github.io"
echo "========================================"
echo ""
