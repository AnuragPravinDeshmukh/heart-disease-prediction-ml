@echo off
title Heart Disease Prediction ML App
echo ========================================================
echo   Starting Heart Disease Prediction Streamlit App...
echo ========================================================
python -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo An error occurred while launching Streamlit.
    pause
)
