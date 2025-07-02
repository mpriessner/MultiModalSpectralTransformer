@echo off
setlocal enabledelayedexpansion

:: Change to the repository root directory
cd /d %~dp0

:: Default commit message
set commit_msg=Update configuration paths and project files

:: Check if a commit message was provided as an argument
if not "%~1"=="" (
    set commit_msg=%~1
)

:: Add all changes
git add .

:: Commit with the message
git commit -m "%commit_msg%"

:: Push to the remote repository
git push

echo.
echo Git operations completed successfully.
echo Added all changes, committed with message: "%commit_msg%", and pushed to remote.
echo.

pause
