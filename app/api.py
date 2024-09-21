import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

def retrieve_prediction(shop_id: int, item_id: int):
    input_dir = Path("../data/model")
    df = pd.read_parquet(input_dir.joinpath("task2.parquet"))
    id = f"{shop_id}-{item_id}"
    return df[df["id"] == id]["prediction"].tolist()[0]

app = FastAPI()

# Request model for input parameters
class SalesPredictionRequest(BaseModel):
    shop_id: int
    item_id: int

# Route to predict sales based on shop_id and item_id
@app.post("/predict_sales/")
async def get_sales_prediction(request: SalesPredictionRequest):
    predicted_sales = retrieve_prediction(request.shop_id, request.item_id)
    return {
        "shop_id": request.shop_id,
        "item_id": request.item_id,
        "predicted_sales": predicted_sales
    }