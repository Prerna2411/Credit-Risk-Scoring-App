import joblib

def load_model(path):
    return joblib.load(path)


def handle_missing_values(df):

def predict_risk(model,input_data):
    probability=model.predict_proba(input_data)[0][1]
    return round(probability *100,2)


def explain_prediction(model,input_data,explainer):
    shap_values=explainer.shap_values(input_data)
    return shap_values


def format_prediction_output(probability):
    if probability >= 50:
        return f"High risk of default:{probability}%"
    else:
        return f"Low risk of default:{probability} %"
    

