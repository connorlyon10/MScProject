#!/bin/bash
# Move to project root
cd "$(dirname "$0")/../.."

# Activate venv
source .venv/bin/activate

# Run Streamlit
streamlit run src/app_VAD.py
