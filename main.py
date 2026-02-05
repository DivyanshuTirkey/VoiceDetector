
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from typing import Optional
from detector import Detector
from utils import decode_audio, trim_audio
from limiter import limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Register Rate Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

detector = Detector()

# Request Model
class VoiceDetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.post("/api/voice-detection")
@limiter.limit("10/minute")
async def detect_voice(
    request: Request,
    body: VoiceDetectionRequest,
    x_api_key: Optional[str] = Header(None)
):
    # 1. API Key Validation
    # In a real scenario, compare with a stored secret. 
    # For now, we accept any key or check against env if desired.
    # The spec says "Your API must validate an API Key... Requests without a valid API key must be rejected."
    # Let's assume a simple check for existence or a predefined key in env.
    
    server_api_key = os.getenv("SERVER_API_KEY", "sk_test_123456789") # Default for testing
    
    if x_api_key != server_api_key:
        return {
            "status": "error",
            "message": "Invalid API key"
        }

    # 2. Input Validation
    if body.audioFormat.lower() != "mp3":
         return {
            "status": "error",
            "message": "Only mp3 format is supported"
        }

    try:
        # 3. Decode Audio
        audio_bytes = decode_audio(body.audioBase64)
        
        # 3.5 Trim Audio (Standard Timeframe: 30s)
        audio_bytes = trim_audio(audio_bytes, max_duration_ms=30000)
        
        # 4. Analyze
        result = detector.analyze_audio(audio_bytes)

        
        # 5. Format Response
        return {
            "status": "success",
            "language": body.language,
            "classification": result["classification"],
            "confidenceScore": result["confidenceScore"],
            "explanation": result["explanation"]
        }

    except ValueError as e:
         return {
            "status": "error",
            "message": str(e)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Internal Server Error: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
