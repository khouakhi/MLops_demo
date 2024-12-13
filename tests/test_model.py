import joblib
import pandas as pd

def test_model_load():
    """
    Test if the model file loads successfully.
    """
    try:
        model = joblib.load("v1/model_v1.pkl")
    except Exception as e:
        assert False, f"Model failed to load: {e}"

def test_valid_predictions():
    """
    Test if the model produces valid predictions for given input data.
    """
    model = joblib.load("v1/model_v1.pkl")
    
    # input data
    test_data = pd.DataFrame({
        "age": [16, 17, 18],           # Age of the students
        "address": [1, 0, 1],         # 1 = Urban, 0 = Rural (converted to numerical)
        "absences": [3, 0, 10]        # Number of absences
    })
    
    predictions = model.predict(test_data)
    
    # Check that the number of predictions matches the input size
    assert len(predictions) == len(test_data), "Prediction count mismatch!"
    
    # Check that predictions are non-negative
    assert all(pred >= 0 for pred in predictions), "Predictions should be non-negative!"
