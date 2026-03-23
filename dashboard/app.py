import os
import uvicorn
import subprocess
import sys
import shutil
import asyncio
import yaml
from collections import deque
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- Configuration ---
HOST = "127.0.0.1"
PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "pipeline_config.yaml")
TEMPLATE_CONFIG_PATH = os.path.join(BASE_DIR, "default_pipeline.yaml")
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")


PIPELINE_ORDER = [
    "Load OG1",
    "Find Profiles Beta"
]

# ---------------------

app = FastAPI(title="Autonomy Toolbox Dashboard")
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


try:
    from toolbox.steps.base_step import REGISTERED_STEPS
    from toolbox.steps.base_test import REGISTERED_QC
    print("Toolbox modules pre-loaded successfully.")
except Exception as e:
    print(f"Warning: Could not pre-load toolbox modules: {e}")
# ------------------------

# Holds the last 2000 lines of terminal output
log_buffer = deque(maxlen=2000) 

class ConfigPayload(BaseModel):
    yaml_content: str

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    with open(INDEX_HTML_PATH, "r") as f:
        return f.read()

@app.post("/api/config")
def save_config(payload: ConfigPayload):
    with open(DEFAULT_CONFIG_PATH, "w") as f:
        f.write(payload.yaml_content)
    return {"status": "success"}

@app.get("/api/available-steps")
def get_available_steps():
    return {"steps": []} 

@app.get("/api/default-config")
def get_default_config():
    """Dynamically builds the default YAML template by reading step schemas."""
    from toolbox.steps.base_step import REGISTERED_STEPS
    
    steps_config = []
    
    for step_name in PIPELINE_ORDER:
        if step_name in REGISTERED_STEPS:
            cls = REGISTERED_STEPS[step_name]
            schema = cls.get_schema()
            
            # Extract just the default values for the YAML
            default_params = {k: v.get("default") for k, v in schema.items()}
            
            steps_config.append({
                "name": step_name,
                "parameters": default_params,
                "diagnostics": False
            })
            
    default_yaml_dict = {
        "pipeline": {
            "name": "Modern Glider Pipeline",
            "description": "Dynamic pipeline generated from step schemas",
            "visualisation": False
        },
        "steps": steps_config
    }
    
    # Dump to a formatted YAML string
    yaml_str = yaml.dump(default_yaml_dict, sort_keys=False, default_flow_style=False)
    return {"yaml_content": yaml_str}

@app.get("/api/config")
def get_config():
    """Returns the user's saved config, or generates a fresh one from schemas if missing."""
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        # Generate the default config dynamically and save it
        default_content = get_default_config()["yaml_content"]
        with open(DEFAULT_CONFIG_PATH, "w") as f:
            f.write(default_content)
            
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        return {"yaml_content": f.read()}

@app.get("/api/browse")
def browse_file():
    if sys.platform == "darwin":
        script = 'POSIX path of (choose file of type {"netcdf", "nc", "public.data"})'
        try:
            result = subprocess.check_output(["osascript", "-e", script], text=True).strip()
            return {"path": result}
        except subprocess.CalledProcessError:
            return {"path": ""}
    else:
        script = (
            "import tkinter as tk, tkinter.filedialog as fd; "
            "root=tk.Tk(); root.withdraw(); root.attributes('-topmost', True); "
            "print(fd.askopenfilename(filetypes=[('NetCDF files', '*.nc')]))"
        )
        try:
            result = subprocess.check_output(["python3", "-c", script], text=True).strip()
            return {"path": result}
        except Exception:
            return {"path": ""}

@app.get("/api/logs")
def get_logs():
    """Endpoint for the frontend to poll terminal output."""
    return {"logs": list(log_buffer)}

@app.post("/api/run")
async def run_pipeline():
    global log_buffer
    log_buffer.clear()
    
    # Run pipeline in a separate process so Tkinter gets its own main thread
    # We use forward slashes for paths to ensure cross-platform compatibility in the script string
    config_path_clean = DEFAULT_CONFIG_PATH.replace("\\", "/")
    script = f"from toolbox.pipeline import Pipeline; Pipeline('{config_path_clean}').run()"
    
    try:
        # Launch the subprocess asynchronously
        env = os.environ.copy()
        env["AUTONOMY_WEB_MODE"] = "1"
                
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env
        )

        # Stream output directly into the log buffer while the process runs
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            # Decode and strip newline, handle weird characters safely
            log_buffer.append(line.decode('utf-8', errors='replace').rstrip('\n'))

        # Wait for the user to close any plots/windows and for the process to exit
        await process.wait()

        if process.returncode != 0:
            raise HTTPException(status_code=500, detail="Pipeline encountered an error. Check logs.")
            
        return {"status": "success", "message": "Pipeline executed successfully."}
        
    except Exception as e:
        log_buffer.append(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/available-qc")
def get_available_qc():
    try:
        from toolbox.steps.base_test import REGISTERED_QC
        return {"tests": list(REGISTERED_QC.keys())}
    except Exception as e:
        return {"tests": [
            "impossible date test", "impossible location test", 
            "position on land test", "impossible speed test", 
            "range test", "gross range test", "stuck value test", 
            "spike test", "valid profile test", "flag full profile", 
            "PAR irregularity test"
        ]}

if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)