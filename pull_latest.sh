#!/bin/bash

# Script to stash local changes and pull the latest updates from the remote repository
# Usage: bash pull_latest.sh [branch_name]
# If branch_name is not provided, it will pull from the current branch

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Determine which branch to pull from
TARGET_BRANCH=${1:-$CURRENT_BRANCH}
echo "Target branch for pulling: $TARGET_BRANCH"

# Check if there are any local changes
STASHED=false
if ! git diff-index --quiet HEAD --; then
    echo "Detected local changes..."
    
    # Add all changes
    echo "Adding all changes..."
    git add .
    
    # Commit changes with generic message
    echo "Committing changes..."
    git commit -m "Stashing changes before pull"
    echo "Changes committed successfully."
    
    # Stash any remaining changes (if commit failed for some reason)
    if ! git diff-index --quiet HEAD --; then
        echo "Stashing remaining changes..."
        git stash push -m "Automatic stash before pulling latest changes"
        STASHED=true
    fi
else
    echo "No local changes detected."
fi

# Fetch all branches from remote
echo "Fetching updates from remote..."
git fetch --all

# Pull the latest changes from the target branch
echo "Pulling latest changes from origin/$TARGET_BRANCH..."
git pull origin $TARGET_BRANCH

# Only show stash instructions if we actually stashed something
if [ "$STASHED" = true ]; then
    echo ""
    echo "You have stashed changes. You can apply them with:"
    echo "  git stash apply"
    echo "Or you can view the stash list with:"
    echo "  git stash list"
    echo ""
    echo "To discard stashed changes, use:"
    echo "  git stash drop"
else
    # Check if there are any existing stashes from previous runs
    STASH_COUNT=$(git stash list | wc -l)
    if [ $STASH_COUNT -gt 0 ]; then
        echo ""
        echo "Note: You have $STASH_COUNT existing stashed change(s) from previous operations."
        echo "View them with: git stash list"
        echo "Apply the most recent with: git stash apply"
    fi
fi

echo "Done! Latest changes have been pulled from origin/$TARGET_BRANCH."
 