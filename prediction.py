import joblib

def predict(data):
    clf=joblib.load('output_models/rf_model.sav')
    return clf.predict(data)