@echo off
setlocal enabledelayedexpansion

:: Change to the repository root directory
cd /d %~dp0

:: Pull the latest changes from the remote repository
git pull

echo.
echo Git pull completed successfully.
echo Your local repository is now up to date with the remote.
echo.

pause
