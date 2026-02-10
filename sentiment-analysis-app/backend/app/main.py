from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import TextIn, SentimentOut
from app.model.predict import get_prediction

app = FastAPI(title="Sentiment AI API")

origins = [
    "https://sentiment-analyzer-ai-theta.vercel.app", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Sentiment API running"}

@app.post("/predict", response_model=SentimentOut)
async def predict_sentiment(data: TextIn):
    result = get_prediction(data.text)
    return {"sentiment": result}
