from contextlib import asynccontextmanager
from fastapi import FastAPI
from modules import sentiment_analysis
import runhouse as rh

SCORER = None


@asynccontextmanager
async def lifespan(app):
    global SCORER
    cluster = rh.ondemand_cluster(
        name="fastapi-runhouse-example",
        instance_type="CPU:2+",
        provider="aws",
        region="us-east-2"
    )
    cluster.up_if_not()
    RemoteScorer = rh.module(sentiment_analysis.SentimentAnalysis).to(cluster)
    SCORER = RemoteScorer()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
def ping():
    return {
        "pong",
    }


@app.post("/score")
def score(text):
    s = SCORER.predict(text)
    return s
