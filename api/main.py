from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Playground API", version="0.1.0")

class ChatRequest(BaseModel):
    message: str
    model: str = "default"

class ChatResponse(BaseModel):
    response: str
    model_used: str

@app.get("/")
async def root():
    return {"message": "AI Playground API is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_available": ["huggingface", "local"],
        "api_keys_configured": bool(os.getenv('HF_TOKEN'))
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for AI interactions"""
    if not os.getenv('HF_TOKEN'):
        raise HTTPException(status_code=400, detail="No API key configured")
    
    # This is where you'd call your AI model
    response_text = f"AI Response to: {request.message}"
    
    return ChatResponse(
        response=response_text,
        model_used=request.model
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
