@echo off
echo ========================================
echo   Building website...
echo ========================================
call npm run build
if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Build completed!
echo   Now manually deploy to gh-pages branch
echo ========================================
echo.
echo The compiled files are in the 'dist' folder.
echo.
echo Next steps:
echo 1. cd dist
echo 2. git init
echo 3. git add -A
echo 4. git commit -m "Deploy"
echo 5. git push -f https://github.com/Video2Act/video2act.github.io.git HEAD:gh-pages
echo.
echo Or run: npm run manual-deploy
echo.
pause
