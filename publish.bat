@echo off
REM Quick publish script for Windows
echo.
echo ========================================
echo   Video2Act Website Publishing Script
echo ========================================
echo.

REM Check if there are uncommitted changes
git diff-index --quiet HEAD --
if errorlevel 1 (
    echo [Step 1/3] Committing changes to source code...
    git add .
    set /p commit_msg="Enter commit message: "
    git commit -m "%commit_msg%"
    git push origin main
    echo [OK] Source code pushed to main branch
    echo.
) else (
    echo [Step 1/3] No changes to commit, skipping...
    echo.
)

echo [Step 2/3] Building website...
call npm run build
if errorlevel 1 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)
echo [OK] Build completed
echo.

echo [Step 3/3] Deploying to GitHub Pages...
call npx gh-pages -d dist
if errorlevel 1 (
    echo [ERROR] Deployment failed!
    pause
    exit /b 1
)
echo.
echo ========================================
echo   Successfully deployed!
echo   Visit: https://video2act.github.io
echo ========================================
echo.
pause
