from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()


class InputData(BaseModel):
    Age: int
    Gender: str
    Height: float
    Weight: float
    Fitness_Level: str
    Fitness_Goal: str
    Medical_History: str


# Load your scikit-learn models
try:
    meal_model = pickle.load(open('meal_plan.pkl', 'rb'))
    calories_model = pickle.load(open('calories.pkl', 'rb'))
    workout_model = pickle.load(open('workout_plan.pkl', 'rb'))
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")


@app.post("/predict_all")
def predict_all(input_data: InputData):
    try:
        # Create a pandas DataFrame with named columns
        input_df = pd.DataFrame({
            'Age': [input_data.Age],
            'Gender': [input_data.Gender],
            'Height': [input_data.Height],
            'Weight': [input_data.Weight],
            'Fitness_Level': [input_data.Fitness_Level],
            'Fitness_Goal': [input_data.Fitness_Goal],
            'Medical_History': [input_data.Medical_History],
        })

        # Make predictions using the loaded models
        meal_plan = meal_model.predict(input_df)
        calories = calories_model.predict(input_df)
        workout_plan = workout_model.predict(input_df)

        # Directly return the predictions
        return {
            "meal_plan": meal_plan[0],
            "calories": calories[0],
            "workout_plan": workout_plan[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error:{str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)