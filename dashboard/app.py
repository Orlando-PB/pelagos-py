import os
import uvicorn
import subprocess
import sys
import asyncio
import yaml
from collections import deque
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HOST = "127.0.0.1"
PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Reroute data folder to the standard user data directory
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_CONFIG_PATH = os.path.join(DATA_DIR, "pipeline_config.yaml")
TEMPLATE_CONFIG_PATH = os.path.join(BASE_DIR, "default_pipeline.yaml")
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")

PIPELINE_ORDER = [
    "Load OG1",
    "Apply QC",
    "Find Profiles Beta",
    "Interpolate Data",
    "Data Export"
]

app = FastAPI(title="Autonomy Toolbox Dashboard")
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..", "src")))

try:
    import toolbox.steps
    from toolbox.steps import STEP_CLASSES, QC_CLASSES
    print("Toolbox modules discovered and pre-loaded successfully.")
except Exception as e:
    print(f"Failed to load toolbox modules: {e}")

log_buffer = deque(maxlen=2000) 
active_process = None

diagnostic_state = {
    "is_paused": False,
    "data": None,
    "user_action": None,
    "new_parameters": None
}

class ConfigPayload(BaseModel):
    yaml_content: str

class DiagActionPayload(BaseModel):
    action: str
    parameters: dict

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    with open(INDEX_HTML_PATH, "r") as f:
        return f.read()

@app.post("/api/config")
def save_config(payload: ConfigPayload):
    with open(DEFAULT_CONFIG_PATH, "w") as f:
        f.write(payload.yaml_content)
    return {"status": "success"}

@app.post("/api/reset")
def reset_config():
    if os.path.exists(DEFAULT_CONFIG_PATH):
        os.remove(DEFAULT_CONFIG_PATH)
    default_content = get_default_config()["yaml_content"]
    with open(DEFAULT_CONFIG_PATH, "w") as f:
        f.write(default_content)
    return {"status": "success", "yaml_content": default_content}

@app.get("/api/available-steps")
def get_available_steps():
    return {"steps": []} 

@app.get("/api/default-config")
def get_default_config():
    from toolbox.steps import STEP_CLASSES
    
    steps_config = []
    
    for step_name in PIPELINE_ORDER:
        if step_name in STEP_CLASSES:
            cls = STEP_CLASSES[step_name]
            schema = cls.get_schema()
            
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
    
    yaml_str = yaml.dump(default_yaml_dict, sort_keys=False, default_flow_style=False)
    return {"yaml_content": yaml_str}

@app.get("/api/config")
def get_config():
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        default_content = get_default_config()["yaml_content"]
        with open(DEFAULT_CONFIG_PATH, "w") as f:
            f.write(default_content)
            
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        return {"yaml_content": f.read()}
@app.get("/api/validate")
def validate_config():
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        return {"status": "error", "message": "No config file found."}
        
    try:
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return {"status": "error", "message": f"Invalid YAML format: {e}"}

    def check_files(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                # If the parameter name implies it's a file or path
                if 'path' in k.lower() or 'file' in k.lower():
                    if not v: # Catch empty strings or None
                        raise ValueError(f"Missing required path for '{k}'. Please select a file.")
                    elif isinstance(v, str) and not os.path.exists(v):
                        raise ValueError(f"Required file not found on system: {v}")
                check_files(v)
        elif isinstance(obj, list):
            for item in obj:
                check_files(item)
    
    try:
        check_files(config.get("steps", []))
        from toolbox.pipeline import Pipeline
        Pipeline(DEFAULT_CONFIG_PATH) 
    except Exception as e:
        return {"status": "error", "message": str(e)}

    return {"status": "success"}
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
    return {"logs": list(log_buffer)}

@app.post("/api/cancel")
def cancel_pipeline():
    global active_process, diagnostic_state
    if active_process:
        try:
            active_process.terminate()
        except Exception:
            pass
    diagnostic_state["user_action"] = "cancel"
    return {"status": "success"}

@app.post("/api/run")
async def run_pipeline():
    global log_buffer, diagnostic_state, active_process
    log_buffer.clear()
    diagnostic_state = {"is_paused": False, "data": None, "user_action": None, "new_parameters": None}
    
    config_path_clean = DEFAULT_CONFIG_PATH.replace("\\", "/")
    script = f"from toolbox.pipeline import Pipeline; Pipeline('{config_path_clean}').run()"
    
    try:
        env = os.environ.copy()
        env["AUTONOMY_WEB_MODE"] = "1"
        
        src_path = os.path.abspath(os.path.join(BASE_DIR, "..", "src"))
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
                
        active_process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env
        )

        while True:
            line = await active_process.stdout.readline()
            if not line:
                break
            log_buffer.append(line.decode('utf-8', errors='replace').rstrip('\n'))

        await active_process.wait()

        if active_process.returncode != 0 and active_process.returncode != -15:
            raise HTTPException(status_code=500, detail="Pipeline encountered an error. Check logs.")
            
        return {"status": "success", "message": "Pipeline execution finished."}
        
    except Exception as e:
        log_buffer.append(str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        active_process = None

@app.get("/api/available-qc")
def get_available_qc():
    try:
        from toolbox.steps import QC_CLASSES
        return {"tests": list(QC_CLASSES.keys())}
    except Exception as e:
        return {"tests": [
            "impossible date test", "impossible location test", 
            "position on land test", "impossible speed test", 
            "range test", "gross range test", "stuck value test", 
            "spike test", "valid profile test", "flag full profile", 
            "PAR irregularity test"
        ]}
    
@app.post("/api/internal/pause")
async def pause_pipeline(payload: dict):
    global diagnostic_state
    diagnostic_state["is_paused"] = True
    diagnostic_state["data"] = payload
    diagnostic_state["user_action"] = None
    return {"status": "ok"}

@app.get("/api/internal/status")
def get_pipeline_status():
    return {
        "status": diagnostic_state["user_action"],
        "parameters": diagnostic_state["new_parameters"]
    }

@app.post("/api/internal/ack")
def ack_pipeline():
    global diagnostic_state
    diagnostic_state["is_paused"] = False
    diagnostic_state["user_action"] = None
    return {"status": "ok"}

@app.get("/api/diagnostic")
def get_diagnostic():
    if diagnostic_state["is_paused"]:
        return {"status": "paused", "data": diagnostic_state["data"]}
    return {"status": "running"}

@app.post("/api/diagnostic/action")
def set_diagnostic_action(payload: DiagActionPayload):
    global diagnostic_state
    diagnostic_state["user_action"] = payload.action
    diagnostic_state["new_parameters"] = payload.parameters
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True, access_log=False)