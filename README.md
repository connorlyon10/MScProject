# Speech Processing in Real-Time with CNNs

## Overview
This project is for my MSc Data Science final project. The project has resulted in machine learning models which can predict whether or not speech is present in audio, and the number of speakers present in speech.

This README contains the structure for the repo and basic set-up tips for running the models.

## Setting up the virtual environment (venv)

1. Download and install **Python 3.9.13** from the official website: https://www.python.org/downloads/release/python-3913/

2. Open the terminal.

3. Navigate to the project root folder (`cxb1114/`):
```ps
cd path/to/cxb1114
```
4. Create a virtual environment called `.venv` using Python 3.9.13:
```ps
python3.9 -m venv .venv
```
(If python3.9 is not in your PATH on Windows, use the full path to the Python executable, e.g.  `C:\Python39\python.exe -m venv .venv`)

5. Activate the virtual environment.

Windows:

```ps
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
source .venv/bin/activate
```

6. Installed required python packages:
```ps
pip install -r requirements.txt
```


## Repository Structure 
- Practical work, experiments and model training can be found in `notebooks/`.
- The model was trained on over 10GB of data which was in `data/` but is too large to host on GitLab.
- Source code, including model files, helper functions and classes can be found in `src/`.
- `thesis/` is the LaTeX source for the final project report.
- The apps which run the models are found in `app/`. More about this in "Setting up the apps" below.
```cxb1114/
|-- app/
|   |-- macOS/
|   |   `-- run_VAD.sh
|   |
|   |-- Win/
|   |   `-- run_VAD.bat
|   |
|
|-- data/
|   |-- data.wav
|   `-- ...
|
|-- notebooks/
|   |-- DataWrangling.ipynb
|   |-- HPTuning.ipynb
|   |-- ModelTraining1_5Class.ipynb
|   |-- ModelTraining2_4Class.ipynb
|   |-- ModelTraining3_VAD.ipynb
|   `-- Visuals.ipynb
|
|-- src/
|   |-- app_Count.py
|   |-- app_VAD.py
|   |-- data.py
|   |-- model.py
|   |
|   |-- utils/
|   |   |-- helpers.py
|   |   |-- SpectrogramExtractor.py
|   |   `-- predict.py
|   |
|   `-- model/
|       |-- ConvCount_5_OptunaResults_F1_234.pt
|       |-- ConvCount_full.pt
|       |-- ConvCount_postVAD.pt
|       `-- ConvCount_VAD.pt
|
|-- thesis/
|   `-- Dissertation.pdf
|
|-- demo.pptx
`-- requirements.txt
```


## Setting up the apps
There are two VAD scripts, for both Windows and macOS. These can be accessed in `app/`. When ran (double click) the scripts will open an app in your browser (localhost).
#### Note for macOS Users
The files must be made executable first:

 `chmod +x app/macOS/*.sh`.
 
The macOS scripts couldn't be tested by the author; please report any issues and I'll assist in a solution.
