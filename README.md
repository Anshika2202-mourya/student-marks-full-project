Student Marks Prediction - Full Project

Files:
- data/student_marks.csv : sample dataset
- train_model.py : trains RandomForest and saves models/model.pkl
- models/model.pkl : saved model (created below)
- app.py : Streamlit web app for inference
- qt/marks.ui : Qt Designer UI
- qt/main_qt.py : PyQt loader for the UI
- requirements.txt, Dockerfile, README.md

Quickstart:
1) python -m venv venv
2) source venv/bin/activate  # or .\venv\Scripts\activate on Windows
3) pip install -r requirements.txt
4) python train_model.py   # creates models/model.pkl
5) streamlit run app.py    # web app
6) python qt/main_qt.py    # desktop GUI (after training)
