
from __future__ import annotations
from pathlib import Path
import hashlib, json, uuid, time, os

LICENSE_FILE = Path.home() / '.hyperfitpro_license.json'


def machine_fingerprint() -> str:
    raw = f"{uuid.getnode()}|{os.name}|HyperFitPro"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def current_license_status():
    if not LICENSE_FILE.exists():
        return {'status': 'evaluation', 'message': 'No license file found. Running in evaluation mode.', 'fingerprint': machine_fingerprint()}
    try:
        data = json.loads(LICENSE_FILE.read_text(encoding='utf-8'))
        return {'status': 'licensed', 'message': 'Local license file loaded.', 'fingerprint': machine_fingerprint(), 'data': data}
    except Exception as e:
        return {'status': 'evaluation', 'message': f'License file could not be read: {e}', 'fingerprint': machine_fingerprint()}
