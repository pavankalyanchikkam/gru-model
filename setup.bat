@echo off
echo 🚀 Setting up PrognosAI Streamlit Application...

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Creating directories...
if not exist models mkdir models
if not exist test_data mkdir test_data
if not exist utils mkdir utils
if not exist assets mkdir assets

echo.
echo ✅ Setup complete!
echo.
echo Next steps:
echo 1. Copy your trained models to the 'models' folder
echo 2. Run 'streamlit run main.py'
echo 3. Open browser to: http://localhost:8501
pauses