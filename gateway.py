"""CPA Gateway - Reverse proxy with per-API-key model access control."""

import fnmatch
import json
import logging
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("cpa-gateway")

CONFIG_PATH = Path(__file__).parent / "gateway_config.yaml"

# Headers that must NOT be copied to avoid corruption
SKIP_REQ_HEADERS = {"host", "content-length", "transfer-encoding"}
SKIP_RESP_HEADERS = {"content-length", "transfer-encoding", "content-encoding", "connection"}


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def model_allowed(model_id: str, allowed: list[str]) -> bool:
    if not allowed:
        return True
    return any(fnmatch.fnmatch(model_id, p) for p in allowed)


def extract_api_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else ""


def forward_headers(request: Request) -> dict[str, str]:
    return {k: v for k, v in request.headers.items() if k.lower() not in SKIP_REQ_HEADERS}


def safe_resp_headers(resp: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in resp.headers.items() if k.lower() not in SKIP_RESP_HEADERS}


app = FastAPI(title="CPA Gateway")
client = httpx.AsyncClient(timeout=600.0)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy(request: Request, path: str):
    cfg = load_config()
    upstream = cfg["gateway"]["upstream"]

    # --- Gateway management routes ---
    if path == "v0/gateway/keys":
        return await handle_keys(request, cfg)

    api_key = extract_api_key(request)
    key_cfg = cfg.get("api-keys", {}).get(api_key)
    allowed = key_cfg.get("models", []) if key_cfg else []

    url = f"{upstream}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    headers = forward_headers(request)
    body = await request.body()

    # --- /v1/models: filter by allowed models ---
    if path == "v1/models" and allowed:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            data["data"] = [m for m in data["data"] if model_allowed(m["id"], allowed)]
            return JSONResponse(content=data)
        return Response(content=resp.content, status_code=resp.status_code)

    # --- Model access check for completions/messages ---
    if path in ("v1/chat/completions", "v1/messages") and allowed and body:
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                model = parsed.get("model", "")
                if model and not model_allowed(model, allowed):
                    return JSONResponse(status_code=403, content={
                        "error": {"type": "forbidden", "message": f"Model '{model}' not allowed for this API key."}
                    })
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    # --- Stream detection ---
    is_stream = False
    if body:
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                is_stream = parsed.get("stream", False)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    # --- Forward: streaming or regular ---
    if is_stream:
        req = client.build_request(request.method, url, headers=headers, content=body)
        resp = await client.send(req, stream=True)
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=safe_resp_headers(resp),
            media_type=resp.headers.get("content-type"),
            background=resp.aclose,
        )

    resp = await client.request(request.method, url, headers=headers, content=body)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=safe_resp_headers(resp),
        media_type=resp.headers.get("content-type"),
    )


async def handle_keys(request: Request, cfg: dict):
    if request.method == "GET":
        return JSONResponse({k: v for k, v in cfg.get("api-keys", {}).items()})
    if request.method == "PUT":
        body = await request.json()
        key = body["key"]
        if key not in cfg.get("api-keys", {}):
            return JSONResponse(status_code=404, content={"error": "key not found"})
        if "models" in body:
            cfg["api-keys"][key]["models"] = body["models"]
        if "label" in body:
            cfg["api-keys"][key]["label"] = body["label"]
        save_config(cfg)
        log.info(f"Updated key {key[:6]}***: models={body.get('models')}")
        return JSONResponse({"status": "ok"})
    return JSONResponse(status_code=405, content={"error": "method not allowed"})


if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    gw = cfg["gateway"]
    log.info(f"CPA Gateway on {gw['host']}:{gw['port']} -> {gw['upstream']}")
    uvicorn.run(app, host=gw["host"], port=gw["port"], log_level="info")
