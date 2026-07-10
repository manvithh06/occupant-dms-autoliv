from fastapi import FastAPI

app = FastAPI(title="Occupant DMS API")

@app.get("/")
def root():
    return {"status": "Occupant DMS backend running"}

@app.get("/health")
def health():
    return {"status": "ok"}