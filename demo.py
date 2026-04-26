import gradio as gr
import numpy as np
import torch
from pathlib import Path
import time
import random

from infer import TerraMindTinyShipClassifier, apply_lora

# ---------------- CONFIG ----------------
DATA_DIR = Path("data/processed_val_chips")
MODEL_PATH = "terramind_tiny_ship_classifier.pth"
THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL ----------------
model = TerraMindTinyShipClassifier()
model = apply_lora(model)

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state, strict=False)
model.to(device)
model.eval()

chip_files = sorted(list(DATA_DIR.glob("*.npy")))

# ---------------- PROCESS ----------------
def process_chip(path):
    chip = np.load(path).astype(np.float32)

    for c in range(chip.shape[0]):
        p2, p98 = np.percentile(chip[c], [2, 98])
        chip[c] = np.clip((chip[c] - p2) / (p98 - p2 + 1e-6), 0, 1)

    tensor = torch.from_numpy(chip).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).item()

    pred = int(prob >= THRESHOLD)
    img = (chip[0] * 255).astype(np.uint8)

    return img, pred

# ---------------- AIS SIMULATION ----------------
base_lat = 15.0
base_lon = 80.0

def generate_fake_ais(step):
    global base_lat, base_lon

    # small movement instead of random jump
    base_lat += random.uniform(-0.2, 0.2)
    base_lon += random.uniform(-0.2, 0.2)

    vessel_id = f"IN-{random.randint(1000,9999)}"
    speed = round(6 + random.random()*10, 1)
    heading = random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

    return f"""VESSEL ID: {vessel_id}  
LAT: {round(base_lat,2)}°N  
LON: {round(base_lon,2)}°E  
SPEED: {speed} knots  
HEADING: {heading}  
STATUS: AIS VERIFIED"""

# ---------------- STREAM ----------------
def run_demo():
    shown = 0

    for path in chip_files:
        img, pred = process_chip(path)

        if pred == 0:
            continue

        shown += 1

        if shown < 7:
            ais = generate_fake_ais(shown)
            alert = "Normal vessel"
            is_alert = False
        else:
            ais = """NO AIS TRANSMISSION  
LAST SIGNAL: UNKNOWN  
TRACK: UNREGISTERED"""
            alert = "🚨 GHOST VESSEL DETECTED"
            is_alert = True

            yield img, "Ship Detected", ais, alert, is_alert
            return

        yield img, "Ship Detected", ais, alert, is_alert
        time.sleep(1.5)

# ---------------- UI UPDATE ----------------
def update_ui(img, pred, ais, alert, is_alert):
    return img, pred, ais, alert, gr.update(elem_classes=["alert"] if is_alert else [])

# ---------------- UI ----------------
with gr.Blocks() as demo:

    container = gr.Column()

    with container:
        image = gr.Image(label="Satellite Feed", elem_id="image_box")

        with gr.Row(elem_id="bottom_panel"):

            with gr.Column(elem_id="left_panel"):
                pred_text = gr.Textbox(label="Detection")
                alert_text = gr.Textbox(label="System Status")

            with gr.Column(elem_id="ais_box"):
                ais_text = gr.Textbox(label="AIS Verification")

    start_btn = gr.Button("🚀 Start Surveillance", elem_id="start_btn")

    start_btn.click(
        fn=run_demo,
        outputs=[image, pred_text, ais_text, alert_text, gr.State()]
    ).then(
        fn=update_ui,
        inputs=[image, pred_text, ais_text, alert_text, gr.State()],
        outputs=[image, pred_text, ais_text, alert_text, container]
    )

# ---------------- LAUNCH ----------------
demo.launch(
    inbrowser=True,
    css="""
    body {
        background-color: #0a0f1c;
        color: #00ffcc;
        font-family: monospace;
        padding-bottom: 100px;
    }

    #image_box img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    #image_box {
        height: 60vh;
    }

    #bottom_panel {
        height: 25vh;
    }

    #left_panel {
        width: 30%;
    }

    #ais_box {
        width: 70%;
        padding: 20px;
        border: 2px solid #00ffcc;
        font-size: 18px;
    }

    /* 🔴 STRONG ALERT VISUAL */
    .alert #ais_box {
        background: #4b0000;
        border: 3px solid red;
        box-shadow: 0 0 25px red;
        color: white;
        animation: blink 1s infinite;
    }

    @keyframes blink {
        0% {opacity: 1;}
        50% {opacity: 0.6;}
        100% {opacity: 1;}
    }

    #start_btn {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 60%;
    }
    """
)