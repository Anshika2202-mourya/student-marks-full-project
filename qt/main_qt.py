import sys, os
from PyQt5 import QtWidgets, uic
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
UI_PATH = os.path.join(os.path.dirname(__file__), 'marks.ui')

class MarksApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(UI_PATH, self)
        self.predictButton.clicked.connect(self.predict)
        self.load_model()

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            self.resultLabel.setText('Model not found. Run train_model.py to create models/model.pkl')
            self.model = None
            return
        data = joblib.load(MODEL_PATH)
        self.model = data['model']

    def predict(self):
        if self.model is None:
            return
        try:
            hours = float(self.hoursInput.text())
            attendance = float(self.attendanceInput.text())
            assignments = float(self.assignmentsInput.text())
            gpa = float(self.gpaInput.text())
            pred = self.model.predict([[hours, attendance, assignments, gpa]])[0]
            self.resultLabel.setText(f'Predicted Marks: {pred:.2f}')
        except Exception as e:
            self.resultLabel.setText('Error: ' + str(e))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MarksApp()
    w.show()
    sys.exit(app.exec_())
