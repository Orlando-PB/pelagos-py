import os
import uvicorn
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- Configuration ---
HOST = "127.0.0.1"
PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "pipeline_config.yaml")
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")
# ---------------------

app = FastAPI(title="Autonomy Toolbox Dashboard")

class ConfigPayload(BaseModel):
    yaml_content: str

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    with open(INDEX_HTML_PATH, "r") as f:
        return f.read()

@app.get("/api/config")
def get_config():
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        with open(DEFAULT_CONFIG_PATH, "w") as f:
            f.write("pipeline:\n  name: New Pipeline\n  description: ''\n  visualisation: false\nsteps: []")
            
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        return {"yaml_content": f.read()}

@app.post("/api/config")
def save_config(payload: ConfigPayload):
    with open(DEFAULT_CONFIG_PATH, "w") as f:
        f.write(payload.yaml_content)
    return {"status": "success"}

@app.get("/api/available-steps")
def get_available_steps():
    try:
        from toolbox.steps.base_step import REGISTERED_STEPS
        return {"steps": list(REGISTERED_STEPS.keys())}
    except Exception as e:
        return {"steps": [], "error": str(e)}

@app.get("/api/browse")
def browse_file():
    # Spawning a subprocess bypasses the macOS main-thread restriction for GUIs
    script = (
        "import tkinter as tk, tkinter.filedialog as fd; "
        "root=tk.Tk(); root.withdraw(); root.attributes('-topmost', True); "
        "print(fd.askopenfilename())"
    )
    try:
        result = subprocess.check_output(["python3", "-c", script], text=True).strip()
        return {"path": result}
    except Exception:
        return {"path": ""}

@app.post("/api/run")
def run_pipeline():
    try:
        from toolbox.pipeline import Pipeline
        pipeline = Pipeline(config_path=DEFAULT_CONFIG_PATH)
        pipeline.run()
        return {"status": "success", "message": "Pipeline executed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/api/available-qc")
def get_available_qc():
    try:
        from toolbox.steps.base_test import REGISTERED_QC
        return {"tests": list(REGISTERED_QC.keys())}
    except Exception as e:
        return {"tests": [], "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)