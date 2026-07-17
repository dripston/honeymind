from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import subprocess
import os
import sys

router = APIRouter()

# Global flag to toggle deception on or off
# By default, we leave it ON, but the UI can change it.
global_state = {
    "HONEYMIND_ENABLED": True
}

@router.post("/toggle", summary="Toggle HoneyMind Defense")
async def toggle_defense(enabled: bool):
    global_state["HONEYMIND_ENABLED"] = enabled
    return {"status": "success", "enabled": enabled}

async def run_script_and_stream(script_path: str, cwd: str, args: list = None):
    """Generator that runs a script and yields its stdout line-by-line for SSE."""
    import asyncio
    cmd_args = [sys.executable, "-u", script_path]
    if args:
        cmd_args.extend(args)
        
    process = await asyncio.create_subprocess_exec(
        *cmd_args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )
    
    try:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            yield f"data: {line.decode('utf-8', errors='replace')}\n\n"
        yield f"data: [PROCESS_COMPLETE]\n\n"
    except asyncio.CancelledError:
        # Handle client disconnect (page reload)
        pass
    finally:
        try:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                process.terminate()
            await process.wait()
        except Exception:
            pass

@router.get("/train", summary="Stream Victim Model Training")
async def trigger_training():
    script_path = os.path.join(os.path.dirname(__file__), "..", "..", "victim_api", "ml", "train.py")
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return StreamingResponse(run_script_and_stream(script_path, cwd), media_type="text/event-stream")

@router.get("/attack/{attack_type}/{target_mode}", summary="Stream Specific Attack")
async def trigger_specific_attack(attack_type: str, target_mode: str):
    script_path = os.path.join(os.path.dirname(__file__), "..", "..", "attackers", "run_all_attackers.py")
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    
    victim_url = os.environ.get("VICTIM_URL", "http://127.0.0.1:8000")
    gateway_url = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8001")
    target_url = f"{victim_url}/api/v1/predict" if target_mode == "undefended" else f"{gateway_url}/api/v1/predict"
    args = ["--attack", attack_type, "--target", target_url, "--mode", target_mode]
    
    return StreamingResponse(
        run_script_and_stream(script_path, cwd, args),
        media_type="text/event-stream"
    )

@router.get("/eval/{target_mode}", summary="Stream Stolen Model Evaluation")
async def trigger_eval(target_mode: str, include: str = "knockoff,jbda,analytical,evolutionary"):
    script_path = os.path.join(os.path.dirname(__file__), "..", "..", "attackers", "run_all_attackers.py")
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    
    victim_url = os.environ.get("VICTIM_URL", "http://127.0.0.1:8000")
    gateway_url = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8001")
    target_url = f"{victim_url}/api/v1/predict" if target_mode == "undefended" else f"{gateway_url}/api/v1/predict"
    args = ["--attack", "eval", "--target", target_url, "--mode", target_mode, "--include", include]
    
    return StreamingResponse(
        run_script_and_stream(script_path, cwd, args),
        media_type="text/event-stream"
    )

@router.get("/files", summary="List Stolen Assets")
async def list_stolen_files():
    attackers_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "attackers"))
    files = []
    if os.path.exists(attackers_dir):
        for f in os.listdir(attackers_dir):
            if f.endswith(".csv") or f.endswith(".pkl") or f.endswith(".json"):
                files.append(f)
    return {"files": sorted(files)}

@router.get("/files/read", summary="Read Stolen Asset")
async def read_stolen_file(filename: str):
    import pandas as pd
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "attackers", filename))
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    if filename.endswith(".csv"):
        df = pd.read_csv(file_path).head(10)
        return {"type": "csv", "content": df.to_html(classes="table", index=False)}
    elif filename.endswith(".json"):
        import json
        with open(file_path, "r") as f:
            return {"type": "json", "content": json.dumps(json.load(f), indent=2)}
    else:
        return {"type": "other", "content": f"Binary file ({filename}) cannot be previewed."}

@router.get("/victim_logs", summary="Stream Victim API Access Logs")
async def stream_victim_logs():
    import asyncio
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "victim_api", "victim_access.log"))
    
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("Victim API Access Log Initialized...\n")

    async def log_generator():
        process = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command", f"Get-Content -Path '{log_path}' -Wait -Tail 0",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                yield f"data: {line.decode('utf-8')}\n\n"
        finally:
            process.terminate()

    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream"
    )

@router.get("/gateway_logs", summary="Stream Gateway Access Logs")
async def stream_gateway_logs():
    import asyncio
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gateway_access.log"))
    
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("HoneyMind AI Gateway Log Initialized...\n")

    async def log_generator():
        process = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command", f"Get-Content -Path '{log_path}' -Wait -Tail 0",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                yield f"data: {line.decode('utf-8')}\n\n"
        finally:
            process.terminate()

    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream"
    )
