from fastapi import FastAPI
from modules import sentiment_analysis

app = FastAPI()

scorer = sentiment_analysis.SentimentAnalysis()


@app.get("/ping")
def ping():
    return {
        "pong",
    }


@app.post("/score")
def score(text):
    s = scorer.predict(text)
    return s
