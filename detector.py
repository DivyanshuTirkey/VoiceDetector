
import os
import json
import time
import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=API_KEY)

class DetectionResult(BaseModel):
    is_ai_generated: bool = Field(..., description="True if the voice is AI-generated, False otherwise.")
    confidence_score: float = Field(..., description="Confidence score between 0.0 and 1.0.")
    explanation: str = Field(..., description="Short explanation of the decision.")

class Detector:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-flash-latest')
        self.max_retries = 3

    def analyze_audio(self, audio_bytes: bytes) -> dict:
        """
        Analyzes audio bytes using Gemini 1.5 Flash to detect if it's AI-generated.
        """
        
        # Write bytes to a temporary file for upload (Gemini requires file upload for audio usually, 
        # or we might be able to pass data directly if supported, but file is safer for 'flash')
        # Actually, for the API, inline data is supported for some modalities, but let's check.
        # It's safer to use the 'parts' construction with mime_type if passing directly 
        # is supported, but often native file API is best. 
        # However, for speed and statelessness, let's try passing as blob if possible, 
        # otherwise use temp file. 
        # 'gemini-1.5-flash' supports inline data for audio.
        
        prompt = """
        You are an expert audio forensics analyst. 
        Analyze the following audio clip to determine if the voice is AI-generated (Synthetic) or Human.
        
        Step-by-step reasoning process:
1. **Prosody & Rhythm**: Natural human speech has irregular pauses, breaths, filler words (um/uh). AI often uniform. Flag AI only if *highly* robotic (no breaths, perfect regularity).
2. **Pitch & Timbre**: Humans show micro-variations, breathiness. AI overly smooth. Count as AI evidence *only* if metallic/unnatural + no organic wobble.
3. **Articulation**: Humans have subtle slurs, regional accents. AI crisp but sometimes blends phonemes oddly. Neutral unless clear synthesis error.
4. **Timing**: Humans vary speed with emphasis. AI predictable. Flag only extreme uniformity.
5. **Spectral Artifacts**: Check for synthesis clues (flat regions, repeating patterns). Human audio often has natural noise floor. This is strongest AI indicator.
6. **Human Variability Check**: Presence of breaths, hesitations, background noise, or emotional shifts = strong HUMAN evidence.
7. **Decision Rule**: Count AI indicators (must be 3+ strong ones). If <3 or any ambiguity, classify HUMAN. Bias toward HUMAN—real humans are messy.      
        Based on this analysis, provide a classification.
        
        Return the result strictly in JSON format matching the following schema:
        {
            "is_ai_generated": boolean,
            "confidence_score": float (0.0 to 1.0),
            "explanation": "string (Max 15 words, crisp and direct reason)"
        }
        """

        for attempt in range(self.max_retries):
            try:
                # pass audio as inline data
                response = self.model.generate_content([
                    prompt,
                    {"mime_type": "audio/mp3", "data": audio_bytes}
                ])
                
                # Cleanup potential markdown code blocks
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text[7:-3].strip()
                elif text.startswith("```"):
                     text = text[3:-3].strip()

                # Parse and Validate
                data = json.loads(text)
                validated = DetectionResult(**data)
                
                return {
                    "classification": "AI_GENERATED" if validated.is_ai_generated else "HUMAN",
                    "confidenceScore": validated.confidence_score,
                    "explanation": validated.explanation
                }

            except (json.JSONDecodeError, ValidationError) as e:
                print(f"Attempt {attempt+1} failed: JSON/Validation Error: {e}")
                time.sleep(1) # wait before retry
            except Exception as e:
                print(f"Attempt {attempt+1} failed: API Error: {e}")
                time.sleep(1)

        # Fallback if all retries fail
        return {
            "classification": "HUMAN", # Default to human to avoid false positives if unsure? Or Error?
            "confidenceScore": 0.0,
            "explanation": "Analysis failed due to repeated errors. Defaulting to Human classification."
        }
