"""
MITRE CWE API client (minimal, stdlib-only).

Example endpoint:
  https://cwe-api.mitre.org/api/v1/cwe/weakness/22
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


DEFAULT_BASE_URL = "https://cwe-api.mitre.org/api/v1/cwe/weakness"


@dataclass(frozen=True)
class MitreApiConfig:
    base_url: str = DEFAULT_BASE_URL
    timeout_sec: float = 30.0
    max_retries: int = 3
    backoff_sec: float = 1.0
    user_agent: str = "CLEAR/mitre-cwe-client"


def _http_get_json(url: str, *, cfg: MitreApiConfig) -> Tuple[Optional[Dict[str, Any]], int]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": cfg.user_agent,
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            data = resp.read()
            if not data:
                return None, status
            return json.loads(data.decode("utf-8", errors="replace")), status
    except urllib.error.HTTPError as e:
        try:
            body = e.read()
            if body:
                return json.loads(body.decode("utf-8", errors="replace")), int(e.code)
        except Exception:
            pass
        return None, int(getattr(e, "code", 500) or 500)
    except urllib.error.URLError:
        return None, 0


def fetch_cwe_weakness(cwe_number: str, *, cfg: Optional[MitreApiConfig] = None) -> Dict[str, Any]:
    cfg = cfg or MitreApiConfig()
    url = f"{cfg.base_url}/{str(cwe_number).strip()}"
    last: Optional[Dict[str, Any]] = None
    last_status = 0
    for attempt in range(cfg.max_retries):
        payload, status = _http_get_json(url, cfg=cfg)
        last, last_status = payload, status
        if status and status < 400:
            return payload or {}
        if status in (429,) or status >= 500 or status == 0:
            if attempt < cfg.max_retries - 1:
                time.sleep(cfg.backoff_sec * (2**attempt))
                continue
        break
    return {
        "error": True,
        "status": last_status,
        "url": url,
        "payload": last,
    }
