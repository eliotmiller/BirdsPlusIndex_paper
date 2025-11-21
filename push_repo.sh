#!/usr/bin/env bash
set -euo pipefail

REMOTE_URL="https://github.com/eliotmiller/BirdsPlusIndex_paper.git"
LFS_THRESHOLD=$((80 * 1024 * 1024))   # 80 MB

# ---------------------------------------
# 1. Initialize Git repo if needed
# ---------------------------------------
if [ ! -d ".git" ]; then
    echo "→ Initializing Git repository..."
    git init
    git remote add origin "$REMOTE_URL"
    git checkout -b main
else
    echo "→ Git repo already initialized."
fi

# ---------------------------------------
# 2. Enable Git LFS
# ---------------------------------------
echo "→ Initializing Git LFS..."
git lfs install || true

# ---------------------------------------
# 3. Create .gitignore
# ---------------------------------------
echo "→ Creating .gitignore..."
echo ".DS_Store" > .gitignore
git add .gitignore
git commit -m "Add .gitignore" || echo "(Nothing to commit)"

# ---------------------------------------
# Helper: LFS-track file if needed
# ---------------------------------------
track_large_file() {
    local file="$1"
    local size
    size=$(stat -f%z "$file")

    if (( size > LFS_THRESHOLD )); then
        echo "  • Tracking large file with Git LFS: $file"
        git lfs track "$file"
    fi
}

# ---------------------------------------
# 4. Add Python/, R/, outputs/
# ---------------------------------------
echo "→ Adding Python/, R/, and outputs/... "

for dir in Python R outputs; do
    if [ -d "$dir" ]; then
        echo "  • Processing $dir/"
        while IFS= read -r -d '' file; do
            track_large_file "$file"
        done < <(find "$dir" -type f -print0)

        git add "$dir"
    fi
done

git add .gitattributes || true
git commit -m "Add Python, R, and outputs" || echo "(Nothing to commit)"
git push origin main || true

# ---------------------------------------
# 5. Commit + push each subdirectory in data/ separately
# ---------------------------------------
echo "→ Processing data/ subdirectories..."

if [ ! -d data ]; then
    echo "⚠ No data/ directory found. Skipping."
    exit 0
fi

for sub in data/*; do
    if [ -d "$sub" ]; then
        name=$(basename "$sub")
        echo "  → Processing subdirectory: $name"

        # Track large files inside this subdirectory
        while IFS= read -r -d '' file; do
            track_large_file "$file"
        done < <(find "$sub" -type f -print0)

        git add "$sub"
        git add .gitattributes || true

        git commit -m "Add data subdirectory: $name" || echo "(Nothing to commit)"
        echo "  • Pushing $name ..."
        git push origin main || true
    fi
done

echo "→ DONE. All data subdirectories committed and pushed separately."
