#!/bin/bash

# Script to reset to the remote branch and pull the latest updates
# Hard coded to use origin/main_cleaned

# Set target branch to main_cleaned
TARGET_BRANCH="main_cleaned"
echo "Resetting to origin/$TARGET_BRANCH..."

# Reset hard to the remote branch
git reset --hard origin/$TARGET_BRANCH

# Fetch all branches from remote
echo "Fetching updates from remote..."
git fetch --all

# Pull the latest changes to ensure we're up to date
echo "Pulling latest changes from origin/$TARGET_BRANCH..."
git pull origin $TARGET_BRANCH

echo "Done! Local repository has been reset to origin/$TARGET_BRANCH."
 