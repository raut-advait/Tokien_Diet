from fastapi import FastAPI

app = FastAPI(title="Token-Diet Dynamic Context Compressor API")

@app.get("/")
async def root():
    return {"message": "Token-Diet API running"}
