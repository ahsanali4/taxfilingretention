import os
import signal
from contextlib import asynccontextmanager
from typing import List

import fastapi
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from taxfilingretention.core.model_training import Classifier
from taxfilingretention.core.preprocessing import Preprocessor
from taxfilingretention.models.models import UserData
from taxfilingretention.utils.config import Settings
from taxfilingretention.utils.logger import LogManager

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load preprocessor and classifier models at startup
    app.state.preprocessor = Preprocessor.load_preprocessor(settings.PREPROCESSOR)
    app.state.classifier = Classifier.load_model(settings.MODEL)
    print("Models loaded and attached to app state.")

    # Yield control back to FastAPI; the models remain in app.state during runtime.
    yield

    # Optionally, perform any cleanup here during shutdown.
    print("Shutting down and cleaning up resources.")


# Initialize FastAPI
app = FastAPI(
    title="Tax Filing Prediction API",
    description="Predicts whether a user will complete their tax filing.",
    version="1.0",
    lifespan=lifespan,
)


logger = LogManager(__name__).get_logger()

# setting up CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health", tags=["Health Check"])
async def health_check():
    # return application version and health information
    return {"status": "ok"}


@app.post("/predict")
def predict(data: List[UserData]):
    try:
        # Convert incoming data to DataFrame
        input_df = pd.DataFrame([d.dict() for d in data])

        # Use the pre-loaded preprocessor and classifier
        processed_input = app.state.preprocessor.transform_new_data(input_df)
        predictions = app.state.classifier.predict(processed_input)
        probabilities = app.state.classifier.predict_proba(processed_input)

        return {"predictions": predictions.tolist(), "probabilities": probabilities.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/", tags=["system"])
async def root_entry():
    return {"message": "****** REST API Tax Filing Retention System ******"}


def shutdown():
    os.kill(os.getpid(), signal.SIGTERM)
    return fastapi.Response(status_code=200, content="Server shutting down...")


@app.on_event("shutdown")
def on_shutdown():
    print("Server shutting down...")


if __name__ == "__main__":
    uvicorn.run("app:app", port=settings.PORT, reload=True, log_level=0)
