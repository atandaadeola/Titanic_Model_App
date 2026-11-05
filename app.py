import gradio as gr
import pandas as pd
import joblib

# Load the trained model once when the app starts
random_forest = joblib.load("Models/forest_model.joblib")

# PILLAR 1: THE FUNCTION
def predict_survival(name, Pclass, Sex, Age, SibSp, PaCh, Fare):
    input_data = pd.DataFrame({
        "Pclass": [Pclass],
        "Sex": [Sex],
        "Age": [Age],
        "SibSp": [SibSp],
        "PaCh": [PaCh],
        "Fare": [Fare]
    })
    
    prediction = random_forest.predict(input_data)[0]
    probabilities = random_forest.predict_proba(input_data)[0]
    
    if prediction == 0:
        result = f"Woops {name.title()}, you Died"
    else:
        result = f"Cheers {name.title()}, you Survived"
        
    # gr.Label expects a dict of {class name: probability}
    probs = {"Died": float(probabilities[0]), "Survived": float(probabilities[1])}
    return result, probs

# PILLAR 2: THE INPUTS (one per function parameter, same order)
inputs = [
    gr.Textbox(label="Name", placeholder="e.g. Adeola Atanda"),
    gr.Radio([1, 2, 3], label="Passenger class (1 = First)", value=3),
    gr.Radio([("male", 0), ("female", 1)], label="Sex", value=0),
    gr.Slider(0, 80, value=30, step=1, label="Age"),
    gr.Number(label="Siblings/Spouses aboard", value=0, precision=0),
    gr.Number(label="Parents/Children aboard", value=0, precision=0),
    gr.Number(label="Fare paid ($)", value=30),
]

# PILLAR 3: THE OUTPUTS (one per return value, same order)
outputs = [
    gr.Textbox(label="Prediction"),
    gr.Label(label="Probability"),
]

# Wire the three pillars together
demo = gr.Interface(
    fn=predict_survival,
    inputs=inputs,
    outputs=outputs,
    title="Titanic Survival Predictor",
    description="Enter a Passenger's details and see if our random forest thinks they'd survive.",
    examples=[
        ["Musa Adamu", 3, 0, 20, 0, 0, 8],
        ["Tomi Jesu", 1, 1, 17, 0, 1, 150],
    ],
)

demo.launch()
