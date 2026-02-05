
import base64
import io

def decode_audio(base64_string: str) -> bytes:
    """
    Decodes a Base64 encoded audio string to bytes.
    """
    try:
        # Check if header exists (e.g., data:audio/mp3;base64,) and strip it
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        
        return base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Invalid Base64 audio data: {str(e)}")

from pydub import AudioSegment

def trim_audio(audio_bytes: bytes, max_duration_ms: int = 30000) -> bytes:
    """
    Trims audio to a maximum duration (default 30 seconds).
    Returns audio bytes in MP3 format.
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        
        if len(audio) > max_duration_ms:
            audio = audio[:max_duration_ms]
            
        # Export back to bytes
        buffer = io.BytesIO()
        audio.export(buffer, format="mp3")
        return buffer.getvalue()
    except Exception as e:
        # In case of any error during trimming (e.g. malformed or pydub issue), 
        # log it and return original to keep flow robust? 
        # Or raise error? Let's raise for now as we want valid MP3s.
        raise ValueError(f"Error trimming audio: {str(e)}")

