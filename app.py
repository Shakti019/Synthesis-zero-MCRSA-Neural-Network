import base64
import io
import math
import socket
import time
from typing import Tuple
from flask import Flask, jsonify, render_template, request, send_file
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import qrcode

from discovery_engine import engine
from src.universal_crystal_engine import (
    MATERIAL_REGISTRY,
    get_material_catalog,
    resolve_material,
    get_material_redesign,
)
from src.vision_detector import detect_product_from_image, generate_molecular_schematic
from src.neural_redesign_engine import NeuralRedesignEngine

app = Flask(__name__, static_folder="static", template_folder="static")

# Ensure engine starts discovery loop on boot
engine.start_loop()
neural_redesign = NeuralRedesignEngine(engine)



def get_local_ip() -> str:
    """Retrieve host's primary local LAN IP address for phone connection."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def generate_qr_base64(data: str) -> str:
    """Generate a high-contrast QR code as a base64 data URI."""
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=2,
        )
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#090d16", back_color="#ffffff")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        print(f"QR generation error: {e}")
        return ""


# Connected Mobile Device State
connected_device = {
    "connected": False,
    "device_name": "Mobile Phone",
    "ip": None,
    "last_seen": 0,
}


def parse_device_name(ua: str) -> str:
    """Extract friendly device name from mobile User-Agent header."""
    ua_lower = (ua or "").lower()
    if "iphone" in ua_lower:
        return "Apple iPhone"
    if "ipad" in ua_lower:
        return "Apple iPad"
    if "samsung" in ua_lower or "sm-" in ua_lower:
        return "Samsung Galaxy"
    if "pixel" in ua_lower:
        return "Google Pixel"
    if "xiaomi" in ua_lower or "redmi" in ua_lower:
        return "Xiaomi Mobile"
    if "oneplus" in ua_lower:
        return "OnePlus Smartphone"
    if "android" in ua_lower:
        return "Android Smartphone"
    if "mobile" in ua_lower:
        return "Mobile Device"
    return "Mobile Companion Phone"


# State for Live Desktop & Mobile Sync (Empty on boot until image received)
latest_scan_counter = 0
latest_scan_data = {}
latest_raw_image_bytes = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/camera")
def camera_portal():
    ua = request.headers.get("User-Agent", "")
    connected_device["connected"] = True
    connected_device["device_name"] = parse_device_name(ua)
    connected_device["ip"] = request.remote_addr
    connected_device["last_seen"] = time.time()
    return render_template("camera.html")


@app.route("/api/device_ping", methods=["POST", "GET"])
def device_ping():
    ua = request.headers.get("User-Agent", "")
    if request.is_json:
        data = request.get_json(silent=True) or {}
        ua = data.get("user_agent") or ua
    connected_device["connected"] = True
    connected_device["device_name"] = parse_device_name(ua)
    connected_device["ip"] = request.remote_addr
    connected_device["last_seen"] = time.time()
    return jsonify({"status": "connected", "device": connected_device})


@app.route("/api/device_status")
def device_status():
    is_active = connected_device["connected"] and (time.time() - connected_device["last_seen"] < 300)
    return jsonify({
        "connected": is_active,
        "device_name": connected_device["device_name"],
        "ip": connected_device["ip"],
        "last_seen": connected_device["last_seen"],
    })


@app.route("/api/device_simulate_connect", methods=["POST"])
def device_simulate_connect():
    connected_device["connected"] = True
    connected_device["device_name"] = "Apple iPhone (Local Wi-Fi)"
    connected_device["ip"] = request.remote_addr
    connected_device["last_seen"] = time.time()
    return jsonify({"status": "simulated", "device": connected_device})


@app.route("/api/device_disconnect", methods=["POST"])
def device_disconnect():
    connected_device["connected"] = False
    connected_device["last_seen"] = 0
    return jsonify({"status": "disconnected"})


def get_public_portal_urls() -> Tuple[str, str]:
    """Dynamically determine the public portal URL for mobile camera scanner.
    Works seamlessly on Render.com, custom domains, and local Wi-Fi.
    """
    # 1. Render provides RENDER_EXTERNAL_URL (e.g. https://synthesis-zero.onrender.com)
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    if render_url:
        base = render_url.rstrip("/")
        return f"{base}/camera", base

    # 2. Check if request context has a live domain host (e.g. accessed via onrender.com)
    try:
        if request and request.host:
            host = request.host.lower()
            if "onrender.com" in host or not any(k in host for k in ("localhost", "127.0.0.1", "0.0.0.0")):
                scheme = "https" if request.is_secure or request.headers.get("X-Forwarded-Proto") == "https" else "http"
                base = f"{scheme}://{request.host}".rstrip("/")
                return f"{base}/camera", base
    except Exception:
        pass

    # 3. Fallback to local network IP for local Wi-Fi testing
    local_ip = get_local_ip()
    port = int(os.environ.get("PORT", 5000))
    return f"http://{local_ip}:{port}/camera", f"http://{local_ip}:{port}"


@app.route("/api/qr_code")
def get_qr_image():
    """Returns QR code image directly for immediate browser rendering."""
    mobile_url, _ = get_public_portal_urls()
    buf = io.BytesIO()
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    qr.add_data(mobile_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#090d16", back_color="#ffffff")
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/connect_info")
def get_connect_info():
    mobile_url, desktop_url = get_public_portal_urls()
    local_ip = get_local_ip()
    port = int(os.environ.get("PORT", 5000))
    qr_b64 = generate_qr_base64(mobile_url)
    return jsonify({
        "local_ip": local_ip,
        "port": port,
        "mobile_url": mobile_url,
        "desktop_url": desktop_url,
        "qr_image": qr_b64,
        "device_connected": connected_device["connected"] and (time.time() - connected_device["last_seen"] < 300),
        "device_name": connected_device["device_name"],
    })


@app.route("/api/material_library")
def get_library():
    return jsonify(get_material_catalog())


@app.route("/api/material/<material_id>")
def get_material_by_id(material_id):
    builder = MATERIAL_REGISTRY.get(material_id)
    if not builder:
        return jsonify({"error": f"Material '{material_id}' not found"}), 404
    return jsonify(builder())


@app.route("/api/latest_scan")
def get_latest_scan():
    return jsonify(latest_scan_data)


@app.route("/api/scan_material", methods=["POST"])
def scan_material():
    global latest_scan_counter, latest_scan_data, latest_raw_image_bytes
    image_bytes = None

    # Handle JSON payload (from camera stream base64)
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        b64 = payload.get("image") or payload.get("image_base64", "")
        if b64:
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(b64)
            except Exception:
                pass

    # Handle multipart file upload (from desktop/mobile file picker)
    elif "image" in request.files:
        f = request.files["image"]
        image_bytes = f.read()

    if image_bytes:
        latest_raw_image_bytes = image_bytes

    # Perform genuine deep learning vision detection with MobileNetV3
    detection = detect_product_from_image(image_bytes)
    detected_hint = detection["baseline_material_id"]
    material_info = resolve_material(detected_hint)

    # Generate compressed base64 thumbnail for live frontend display
    captured_image_uri = ""
    if image_bytes:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img.thumbnail((360, 360))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            captured_image_uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except Exception as e:
            print(f"Thumbnail generation error: {e}")

    # Generate authentic 2D molecular blueprint schematic
    atoms = material_info.get("structure_3d", {}).get("atoms", [])
    bonds = material_info.get("structure_3d", {}).get("bonds", [])
    schematic_uri = generate_molecular_schematic(atoms, bonds)

    # Determine caller source
    ua = request.headers.get("User-Agent", "").lower()
    source_label = "Smartphone Camera Stream" if ("mobile" in ua or "iphone" in ua or "android" in ua) else "Workstation Optical Upload"

    latest_scan_counter += 1
    result = {
        "status": "success",
        "scan_id": latest_scan_counter,
        "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source_label,
        "product_name": detection["product_name"],
        "top_category": detection["top_category"],
        "detection_confidence": round(detection["confidence"] * 100, 1),
        "top5_predictions": detection.get("top5_predictions", []),
        "model_architecture": detection["model_architecture"],
        "material_id": detected_hint,
        "formula": material_info.get("formula", ""),
        "density_g_cm3": material_info.get("properties", {}).get("density_g_cm3", 1.0),
        "structure_3d": material_info.get("structure_3d", {}),
        "captured_image_uri": captured_image_uri,
        "schematic_uri": schematic_uri,
    }
    latest_scan_data = result

    # Mark device active
    connected_device["connected"] = True
    connected_device["last_seen"] = time.time()

    return jsonify(result), 200


@app.route("/api/redesign_product", methods=["POST", "GET"])
def redesign_product():
    """Single-click Autonomous Material Redesign Pipeline (NO prompt box required).

    Computes:
    1. PyTorch EGNN forward pass on baseline atomic coordinates
    2. Deep learning atomic lattice reconfiguration via PyTorch Autograd
    3. Real formation energy and thermodynamic stability gain delta
    4. 2D orthographic molecular blueprint projection
    Zero fake text strings, zero hardcoded shortcuts.
    """
    global latest_raw_image_bytes, latest_scan_data
    payload = request.get_json(silent=True) or {}

    img_bytes = latest_raw_image_bytes
    b64_input = payload.get("image") or payload.get("image_base64")
    if b64_input:
        if "," in b64_input:
            b64_input = b64_input.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(b64_input)
            latest_raw_image_bytes = img_bytes
        except Exception:
            pass

    # Execute REAL PyTorch EGNN forward pass and Autograd atomic relaxation
    redesign_result = neural_redesign.execute_redesign(latest_scan_data)

    relaxed_atoms = redesign_result["redesigned_model"]["structure_3d"]["atoms"]
    relaxed_bonds = redesign_result["redesigned_model"]["structure_3d"]["bonds"]
    redesign_result["schematic_uri"] = generate_molecular_schematic(relaxed_atoms, relaxed_bonds)
    redesign_result["captured_image_uri"] = latest_scan_data.get("captured_image_uri", "")

    return jsonify(redesign_result), 200


@app.route("/api/finetune_cpu", methods=["POST"])
def trigger_finetune():
    """Triggers authentic CPU fine-tuning on universal dataset in background."""
    import threading
    from scripts.finetune_cpu import run_cpu_finetuning
    t = threading.Thread(target=run_cpu_finetuning, kwargs={"epochs": 8, "lr": 1e-4})
    t.daemon = True
    t.start()
    return jsonify({"status": "finetune_started", "device": "cpu", "epochs": 8}), 202




@app.route("/api/status")
def get_status():
    is_device_active = connected_device["connected"] and (time.time() - connected_device["last_seen"] < 300)
    return jsonify({
        'is_running': engine.is_running,
        'total_steps': engine.total_steps,
        'device_connected': is_device_active,
        'device_name': connected_device["device_name"],
        'total_discoveries': len(engine.discovered_vault),
        'current_formula': engine.current_formula,
        'current_loss': round(engine.current_loss, 6),
        'prediction_uncertainty': round(engine.current_uncertainty, 6),
        'learning_rate': engine.optimizer.param_groups[0]['lr'],
        'device': str(engine.device),
        'last_checkpoint_time': engine.last_checkpoint_time,
        'worker_thread_alive': bool(
            engine.worker_thread and engine.worker_thread.is_alive()
        ),
        'last_training_time': engine.last_training_time,
        'last_loop_error': engine.last_loop_error,
        'training_phase': engine.training_phase,
        'pretraining_completed': engine.pretraining_completed,
        'pretraining_epochs_completed': engine.pretraining_epochs_completed,
        'pretraining_epochs': engine.pretraining_epochs,
        'training_records': len(engine.train_dataset),
        'candidate_pool_exhausted': not bool(engine.candidate_pool),
        'search_status': (
            'pretraining' if not engine.pretraining_completed
            else 'exhausted' if not engine.candidate_pool
            else 'active'
        ),
        'latest_scan_id': latest_scan_counter,
    })


@app.route("/api/discoveries")
def get_discoveries():
    limit = request.args.get('limit', default=100, type=int)
    return jsonify(engine.discovered_vault[:limit])


@app.route("/api/neural_graph")
def get_neural_graph():
    return jsonify(engine.get_neural_graph_data())


@app.route("/api/telemetry")
def get_telemetry():
    return jsonify(engine.get_telemetry())


@app.route("/api/lattice_3d/<formula>")
def get_lattice_3d(formula):
    matching = next((item for item in engine.discovered_vault if item['formula'] == formula), None)
    if matching:
        return jsonify(matching)
    # Check universal crystal resolver
    resolved = resolve_material(formula)
    return jsonify(resolved)


@app.route("/api/control/start", methods=['POST'])
def start_engine():
    engine.start_loop()
    return jsonify({'status': 'started', 'is_running': engine.is_running})


@app.route("/api/control/pause", methods=['POST'])
def pause_engine():
    engine.pause_loop()
    return jsonify({'status': 'paused', 'is_running': engine.is_running})


@app.route("/api/control/checkpoint", methods=['POST'])
def force_checkpoint():
    engine.save_checkpoint()
    return jsonify({'status': 'checkpoint_saved', 'time': engine.last_checkpoint_time})


@app.route("/api/feedback", methods=['POST'])
def record_feedback():
    payload = request.get_json(silent=True) or {}
    try:
        record = engine.record_external_feedback(
            payload.get('formula'),
            payload.get('observed_energy_eV'),
            payload.get('source'),
        )
    except (TypeError, ValueError) as error:
        return jsonify({'error': str(error)}), 400
    return jsonify({'status': 'feedback_recorded', 'record': record}), 201


@app.route("/api/ingest/materials-project", methods=['POST'])
def ingest_materials_project():
    payload = request.get_json(silent=True)
    records = payload if isinstance(payload, list) else (payload or {}).get('records')
    if not isinstance(records, list):
        return jsonify({'error': 'records must be a JSON array'}), 400
    results = engine.ingest_materials_project_records(records)
    return jsonify({
        'accepted': sum(result.status == 'accepted' for result in results),
        'rejected': sum(result.status == 'rejected' for result in results),
        'duplicates': sum(result.status == 'duplicate' for result in results),
        'results': [
            {
                'status': result.status,
                'fingerprint': result.fingerprint,
                'reason': result.reason,
            }
            for result in results
        ],
    }), 202


@app.route("/api/materials-project/sync", methods=['POST'])
def sync_materials_project():
    try:
        status = engine.sync_materials_project_once()
    except RuntimeError as error:
        return jsonify({'error': str(error)}), 503
    return jsonify(status), 202


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    ip = get_local_ip()
    print("=" * 65)
    print("  SYNTHESIS-ZERO - Autonomous Material Intelligence")
    print(f"  -> Desktop Workstation : http://localhost:{port}")
    print(f"  -> Phone Camera Scanner: http://{ip}:{port}/camera")
    print("=" * 65)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
