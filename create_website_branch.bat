@echo off
echo ===================================================
echo Create Website V1 Clean Branch
echo ===================================================
echo.

echo Changing to repository root directory...
cd /d %~dp0

echo.
echo Checking current branch...
for /f "tokens=*" %%a in ('git branch --show-current') do set CURRENT_BRANCH=%%a
echo Current branch: %CURRENT_BRANCH%

echo.
echo Checking for uncommitted changes...
git status --porcelain > temp_status.txt
set /p STATUS=<temp_status.txt
del temp_status.txt

if not "%STATUS%"=="" (
    echo.
    echo You have uncommitted changes in your repository.
    set /p COMMIT="Would you like to commit these changes before creating the branch? (y/n): "
    if /i "%COMMIT%"=="y" (
        echo.
        set /p COMMIT_MSG="Enter commit message (or press Enter for default message): "
        if "%COMMIT_MSG%"=="" set COMMIT_MSG="Website cleanup before branching"
        
        echo Committing changes with message: %COMMIT_MSG%
        git add .
        git commit -m "%COMMIT_MSG%"
        
        if errorlevel 1 (
            echo Error committing changes. Aborting branch creation.
            goto :end
        )
    ) else (
        echo Continuing without committing. Changes will be carried to the new branch.
    )
)

echo.
echo Creating new branch 'website-v1-clean'...
git branch -m %CURRENT_BRANCH% website-v1-clean
if errorlevel 1 (
    echo.
    echo Error creating branch. Trying alternative method...
    git checkout -b website-v1-clean
)

echo.
echo Current branch status:
git branch

echo.
echo Successfully created and switched to branch 'website-v1-clean'
echo This branch will be used for website fixes and improvements.
echo.
echo Next steps:
echo 1. Make your changes to fix the website
echo 2. Commit your changes regularly
echo 3. Push the branch to remote when ready with: git push -u origin website-v1-clean
echo.

:end
echo.
echo Press any key to exit...
pause > nul
