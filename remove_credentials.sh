#!/bin/bash

# Create a backup of the repository first
BACKUP_DIR="../tapo-cam-backup-$(date +%Y%m%d_%H%M%S)"
echo "Creating backup at: $BACKUP_DIR"
cp -r . "$BACKUP_DIR"

# Use git filter-branch to remove credentials
echo "Removing credentials from git history..."

# First, let's identify all files that might contain credentials
FILES_WITH_CREDS=$(git log --all --name-only --pretty=format: | sort | uniq | grep -E '\.(rs|toml|md|txt)$')

# Create a filter to remove specific credential strings
git filter-branch --force --index-filter \
    "git rm --cached --ignore-unmatch -f -r . >/dev/null 2>&1; \
     git reset --hard HEAD" \
    --prune-empty --tag-name-filter cat -- --all

# Now specifically remove credential strings from file contents
git filter-branch --tree-filter "
    # Remove hardcoded credentials from main.rs
    if [ -f src/main.rs ]; then
        sed -i \"s/kverndal1@gmail.com/TAPO_EMAIL/g\" src/main.rs
        sed -i \"s/Kvernd@l1/TAPO_PASSWORD/g\" src/main.rs
        sed -i \"s/rtsp:\\/\\/kverndal1@gmail.com:Kvernd@l1@192.168.10.185\\/stream1/rtsp:\\/\\/TAPO_EMAIL:TAPO_PASSWORD@192.168.10.185\\/stream1/g\" src/main.rs
    fi
    
    # Check other files too
    for file in *.rs *.toml *.md *.txt; do
        if [ -f \"\$file\" ]; then
            sed -i \"s/kverndal1@gmail.com/TAPO_EMAIL/g\" \"\$file\"
            sed -i \"s/Kvernd@l1/TAPO_PASSWORD/g\" \"\$file\"
        fi
    done
" --prune-empty --tag-name-filter cat -- --all

# Clean up
git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "Done! Credentials removed from history."
echo "Original backup saved at: $BACKUP_DIR"
echo ""
echo "IMPORTANT: You must force push to update remote:"
echo "  git push origin --force --all"
echo "  git push origin --force --tags"