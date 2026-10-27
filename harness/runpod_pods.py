"""Launch / query / stop RunPod vLLM pods for self-hosted work (T2 log-probs, T7 raw completions, base model).

python3 -m harness.runpod_pods launch --version 3.8            # one pod per model
python3 -m harness.runpod_pods launch --hf Qwen/Qwen3.5-9B-Base --name base9b
python3 -m harness.runpod_pods status
python3 -m harness.runpod_pods wait <pod_id>                    # block until /v1/models answers
python3 -m harness.runpod_pods stop <pod_id> | stop --all
Pods are recorded in data/pods.json. ALWAYS stop pods when done; they bill per hour.
"""
import argparse, json, os, sys, time
import requests
from . import config
from .client import _load_env
_load_env()
KEY = os.environ["RUNPOD_API_KEY"]
HF = os.environ.get("HF_TOKEN", "")
GQL = "https://api.runpod.io/graphql?api_key=" + KEY
PODS_FILE = config.ROOT / "data" / "pods.json"
GPU_PREFS = ["NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA H100 80GB HBM3"]
IMAGE = "vllm/vllm-openai:latest"

def gql(q, variables=None):
    r = requests.post(GQL, json={"query": q, "variables": variables or {}}, timeout=60)
    d = r.json()
    if "errors" in d: raise RuntimeError(d["errors"])
    return d["data"]

def _pods():
    return json.loads(PODS_FILE.read_text()) if PODS_FILE.exists() else {}
def _save(p):
    PODS_FILE.parent.mkdir(parents=True, exist_ok=True); PODS_FILE.write_text(json.dumps(p, indent=1))

def launch(hf_id, name, gpu=None, max_len=8192, disk=120, extra_args=""):
    args = f"--model {hf_id} --max-model-len {max_len} --gpu-memory-utilization 0.92 --dtype bfloat16 --served-model-name {hf_id} --port 8000 {extra_args}".strip()
    env = [{"key": "HF_TOKEN", "value": HF}, {"key": "HUGGING_FACE_HUB_TOKEN", "value": HF}, {"key": "VLLM_ARGS", "value": args}]
    gpus = [gpu] if gpu else GPU_PREFS
    last = None
    for g in gpus:
        q = """mutation($in: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: $in) { id desiredStatus imageName machineId costPerHr } }"""
        inp = {"cloudType": "SECURE", "gpuCount": 1, "gpuTypeId": g, "name": name, "imageName": IMAGE,
               "containerDiskInGb": disk, "volumeInGb": 0, "ports": "8000/http", "env": env,
               "dockerArgs": args, "minVcpuCount": 8, "minMemoryInGb": 60}
        try:
            d = gql(q, {"in": inp})["podFindAndDeployOnDemand"]
            if d:
                p = _pods(); p[d["id"]] = {"name": name, "hf": hf_id, "gpu": g, "cost_hr": d.get("costPerHr"), "started": time.time(), "url": f"https://{d['id']}-8000.proxy.runpod.net/v1"}
                _save(p); print(json.dumps({"id": d["id"], **p[d["id"]]})); return d["id"]
        except Exception as e:
            last = e; print(f"  {g}: {str(e)[:160]}", file=sys.stderr)
    raise RuntimeError(f"no GPU available: {last}")

def status():
    q = """query { myself { clientBalance pods { id name desiredStatus costPerHr runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } machine { gpuDisplayName } } } }"""
    d = gql(q)["myself"]
    print("balance:", d["clientBalance"])
    for p in d["pods"]:
        rt = p.get("runtime") or {}
        print(json.dumps({"id": p["id"], "name": p["name"], "status": p["desiredStatus"], "gpu": (p.get("machine") or {}).get("gpuDisplayName"), "cost_hr": p["costPerHr"], "uptime_s": rt.get("uptimeInSeconds")}))
    return d

def wait(pod_id, timeout=3600):
    url = f"https://{pod_id}-8000.proxy.runpod.net/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                print("ready:", r.json()); return True
        except requests.RequestException:
            pass
        time.sleep(20)
    return False

def stop(pod_id):
    q = """mutation($id: String!) { podTerminate(input: {podId: $id}) }"""
    gql(q, {"id": pod_id}); p = _pods(); p.pop(pod_id, None); _save(p); print("terminated", pod_id)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    l = sub.add_parser("launch"); l.add_argument("--version"); l.add_argument("--hf"); l.add_argument("--name"); l.add_argument("--gpu"); l.add_argument("--max-len", type=int, default=8192); l.add_argument("--extra", default="")
    sub.add_parser("status")
    w = sub.add_parser("wait"); w.add_argument("pod_id")
    s = sub.add_parser("stop"); s.add_argument("pod_id", nargs="?"); s.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.cmd == "launch":
        hf = a.hf or config.HF_IDS[a.version]; launch(hf, a.name or f"qwen{a.version}", a.gpu, a.max_len, extra_args=a.extra)
    elif a.cmd == "status": status()
    elif a.cmd == "wait": sys.exit(0 if wait(a.pod_id) else 1)
    elif a.cmd == "stop":
        if a.all:
            for pid in list(_pods()): stop(pid)
        else: stop(a.pod_id)
