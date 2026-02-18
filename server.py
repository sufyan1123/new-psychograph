"""
FastAPI Backend Server for PsychoGraph Instagram Chat Analyzer
This provides a web API that the dashboard can call to analyze files
without needing to run Python scripts manually from command line.

Usage:
  1. pip install fastapi uvicorn anthropic
  2. export ANTHROPIC_API_KEY="your-key-here"
  3. python server.py
  4. Open browser to http://127.0.0.1:8000
"""

import json
import os
import tempfile
import shutil
from typing import Optional
from pathlib import Path

# ── FastAPI imports for building the web server ────────────────────────────────
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Import all the analysis functions from our analyzer module ─────────────────
from analyzer import (
    load_instagram_export,
    identify_patient,
    parse_thread,
    format_for_claude,
    trim_to_token_limit,
    analyze_defense_mechanisms,
    analyze_kpis,
    qualitative_summary,
)

# ════════════════════════════════════════════════════════════════
#  FASTAPI APP SETUP
# ════════════════════════════════════════════════════════════════

# Create the FastAPI application instance
app = FastAPI(
    title="PsychoGraph API",
    description="Instagram chat analysis with Claude AI",
    version="1.0.0"
)

# ── CORS Middleware: allows the HTML dashboard to make API calls ───────────────
# Without this, browsers block cross-origin requests for security
# In production, you'd restrict allow_origins to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Allow all origins (fine for local dev)
    allow_credentials=True,        # Allow cookies/auth headers
    allow_methods=["*"],           # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],           # Allow all request headers
)


# ════════════════════════════════════════════════════════════════
#  ROUTE 1: Serve the dashboard HTML at the root URL
# ════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """
    When users visit http://127.0.0.1:8000/ in their browser,
    serve the dashboard HTML file directly.
    
    This eliminates the need to open the HTML file separately —
    everything runs through one URL.
    """
    dashboard_path = Path(__file__).parent / "dashboard.html"
    
    # Check if the dashboard file exists in the same folder as this script
    if not dashboard_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"dashboard.html not found. Make sure it's in {Path(__file__).parent}"
        )
    
    # Read and return the HTML content
    with open(dashboard_path, "r", encoding="utf-8") as f:
        return f.read()


# ════════════════════════════════════════════════════════════════
#  ROUTE 2: Health check endpoint
# ════════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    """
    Simple endpoint to verify the server is running.
    Returns a JSON object with the server status.
    
    Visit http://127.0.0.1:8000/health to test.
    """
    return {"status": "ok", "message": "PsychoGraph API is running"}


# ════════════════════════════════════════════════════════════════
#  ROUTE 3: Main analysis endpoint
# ════════════════════════════════════════════════════════════════

@app.post("/analyze")
async def analyze_instagram_export(file: UploadFile = File(...)):
    """
    Main API endpoint that receives an Instagram export file (or folder),
    runs the full Claude analysis pipeline, and returns JSON results.
    
    The dashboard calls this endpoint when the user uploads a file.
    
    Accepts:
      - Single message_X.json file from one conversation
      - A .zip file containing the full Instagram export folder
      
    Returns:
      JSON object matching the analysis_results.json format
    """
    
    # ── Step 1: Validate the uploaded file ─────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    filename = file.filename.lower()
    
    # Accept either JSON or ZIP files
    if not (filename.endswith(".json") or filename.endswith(".zip")):
        raise HTTPException(
            status_code=400,
            detail="Please upload either a .json file or a .zip of your Instagram export"
        )
    
    # ── Step 2: Read the uploaded file into memory ─────────────────────────────
    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    # ── Step 3: Handle the file based on its type ──────────────────────────────
    temp_dir = None
    try:
        if filename.endswith(".json"):
            # Single JSON file — parse it directly
            data = handle_json_file(file_content, filename)
            
        elif filename.endswith(".zip"):
            # ZIP archive — extract it to a temp folder and process
            temp_dir = tempfile.mkdtemp()
            data = handle_zip_file(file_content, temp_dir)
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # ── Step 4: Run the full analysis pipeline ─────────────────────────────
        results = run_full_analysis(data)
        
        return JSONResponse(content=results)
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is (they have proper error messages)
        raise
    
    except Exception as e:
        # Catch any unexpected errors and return a 500 error
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
    
    finally:
        # ── Step 5: Clean up temporary files ───────────────────────────────────
        # If we created a temp directory for ZIP extraction, delete it
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def handle_json_file(file_content: bytes, filename: str) -> dict:
    """
    Parse a single Instagram message JSON file.
    Handles both UTF-8 and Latin-1 encodings.
    """
    # Try UTF-8 first (most common)
    try:
        text = file_content.decode("utf-8")
        data = json.loads(text)
        return {"single_file": data, "filename": filename}
    
    except UnicodeDecodeError:
        # Fall back to Latin-1 if UTF-8 fails (Instagram sometimes uses this)
        try:
            text = file_content.decode("latin-1")
            data = json.loads(text)
            return {"single_file": data, "filename": filename}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON encoding: {str(e)}")
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")


def handle_zip_file(file_content: bytes, temp_dir: str) -> dict:
    """
    Extract a ZIP file to a temporary directory and scan for Instagram
    message files. Returns a dict that can be passed to the analyzer.
    """
    import zipfile
    
    zip_path = os.path.join(temp_dir, "upload.zip")
    
    # Write the uploaded ZIP to disk temporarily
    with open(zip_path, "wb") as f:
        f.write(file_content)
    
    # Extract all files from the ZIP
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    
    # Return the path to the extracted folder so analyzer can scan it
    return {"folder": temp_dir}


def run_full_analysis(data: dict) -> dict:
    """
    Main analysis orchestrator. Takes either a single JSON file
    or a folder path, runs the full Claude analysis pipeline,
    and returns the results in dashboard format.
    """
    
    # ── Case 1: Single JSON file uploaded ──────────────────────────────────────
    if "single_file" in data:
        raw_json = data["single_file"]
        filename = data.get("filename", "unknown")
        
        # Check if this is a raw Instagram message file or already-analyzed results
        if "participants" in raw_json and "messages" in raw_json:
            # This is a raw Instagram conversation file — analyze it
            threads = {"uploaded_conversation": raw_json}
        
        elif "patient_name" in raw_json and "conversations" in raw_json:
            # This is already an analysis_results.json file — just return it
            return raw_json
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Unrecognized JSON format. Expected Instagram message export or analysis_results.json"
            )
    
    # ── Case 2: ZIP folder uploaded ────────────────────────────────────────────
    elif "folder" in data:
        folder_path = data["folder"]
        # Use the existing load_instagram_export function to scan the folder
        threads = load_instagram_export(folder_path)
    
    else:
        raise HTTPException(status_code=400, detail="Invalid data structure")
    
    # ── Extract patient name from first thread ─────────────────────────────────
    if not threads:
        raise HTTPException(status_code=400, detail="No Instagram message data found in upload")
    
    first_thread = next(iter(threads.values()))
    patient_name = identify_patient(first_thread.get("participants", []))
    
    # ── Build results container ────────────────────────────────────────────────
    results = {
        "patient_name": patient_name,
        "conversations": {}
    }
    
    # ── Loop through each conversation thread and analyze it ───────────────────
    for thread_key, thread_data in threads.items():
        
        # Use the human-readable title if available
        participant_label = thread_data.get("title", thread_key)
        
        # Parse the raw messages into clean format
        messages = parse_thread(thread_data, patient_name)
        
        # Skip threads where patient sent no messages
        patient_messages = [m for m in messages if m.get("is_patient", False)]
        if not patient_messages:
            continue
        
        # Format the conversation for Claude and trim if too long
        conversation_text = format_for_claude(messages)
        conversation_text = trim_to_token_limit(conversation_text)
        
        # ── Run all three Claude analyses ──────────────────────────────────────
        try:
            defense_data = analyze_defense_mechanisms(conversation_text, participant_label)
            kpi_data     = analyze_kpis(conversation_text, participant_label)
            summary_data = qualitative_summary(conversation_text, participant_label)
            
            # Store the results for this conversation
            results["conversations"][participant_label] = {
                "message_count":       len(messages),
                "defense_mechanisms":  defense_data,
                "kpis":                kpi_data,
                "qualitative_summary": summary_data
            }
        
        except Exception as e:
            # If one conversation fails, store the error but continue with others
            results["conversations"][participant_label] = {
                "error": str(e)
            }
    
    return results


# ════════════════════════════════════════════════════════════════
#  SERVER STARTUP
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    # Print startup message with instructions
    print("\n" + "="*60)
    print("🧠 PsychoGraph Server Starting...")
    print("="*60)
    print("\n📍 Open your browser to: http://127.0.0.1:8000")
    print("📊 Dashboard will load automatically")
    print("🔑 Make sure ANTHROPIC_API_KEY is set in your environment\n")
    print("Press CTRL+C to stop the server\n")
    
    # Start the Uvicorn web server
    # host="0.0.0.0" makes it accessible from other devices on your network
    # port=8000 is the default development port
    # reload=True auto-restarts the server when you edit this file (useful for dev)
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
