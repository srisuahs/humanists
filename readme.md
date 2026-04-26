# TerraMind-Tiny: Orbital Ship Identification & AIS Verification

### The Problem
International waters are plagued by "ghost ships" that intentionally spoof or block their **AIS (Automatic Identification System)** data to evade tracking. Additionally, many smaller vessels lack AIS beacons entirely. 

Our model identifies these ships via satellite SAR imagery and verifies them against AIS logs from 10–15 minutes prior. By utilizing on-board data center compute, we can transfer inference results faster than raw data can be processed or downlinked. This allows authorities to identify and intercept vessels before the moment of interest has passed. This is vital intelligence for governments for stopping illegal activities and for traders to gain information about movement of goods like oil barrels and strategic assets.

**Note:** AIS information is not public but is commonly available for sale—a practice generally used by hedge funds and government facilities for the same purpose.

---

### About the Dataset
We utilize the **SARFish dataset** for high-resolution radar imagery.
* **Imagery Sources:** [Full SARFish Dataset (HuggingFace)](https://huggingface.co/datasets/ConnorLuckettDSTG/SARFish/tree/main/GRD).
* **Labels:** Ground truth labels are sourced from [John-J-Tanner's SARFish Repository](https://github.com/John-J-Tanner/Extract-SARFish-Data/tree/main/Labels).
* **Training Strategy:** After processing through our custom **dataset_chopper.py** to combat data imbalance, the resulting set contains **62 positive** and **62 negative** training examples.
* **Testing:** Benchmarking was performed on a random unseen scene from the complete SARFish GRD repository.
* **Modalities:** * **VH (Vertical-Horizontal):** Essential for metal-on-water detection; metal ship hulls create a high cross-polarized return against the dark ocean.
    * **VV (Vertical-Vertical):** Provides sea state context; helpful for filtering out waves and surface roughness that might otherwise cause false positives.
* **Core Backbone:** We use these two modalities with **TerraMind** to generate contextual input embeddings, which are then passed to our specialized detection head.

---

### Script Explanation & Usage

| Script | Function |
| :--- | :--- |
| `dataset_chopper.py` | Solves the massive class imbalance inherent in SAR data. It segments large swaths into tiles and balances the ship-to-ocean ratio to prevent model bias. |
| `train.py` | Fine-tunes the TerraMind-Tiny backbone with a **GELU-Dropout-Linear** detection head. It utilizes LoRA for efficiency and generates a quantized model for edge deployment. |
| `infer.py` | Tests the generated model weights, runs the benchmark samples, and prints final performance metrics. |

---

### Reproduction
**Requirements:** Python 3.11.x or 3.12.x (Tested on **3.12.10**). Later versions of Python are not stable with all geospatial libraries; Ensure all libraries in `requirements.txt` are installed.

1. **Immediate Test:** You can run `infer.py` directly to check the demo. The `processed_val_chips` folder already contains the first 24 sections of the scene and full labels for quick validation.(dataset size <10mb)
2. **Full Implementation Steps:**
    * Download a SAR product (e.g., `S1A_IW_GRDH...SAFE`).
    * Navigate to: `.../measurement` to find the VH and VV `.tiff` files.
    * Transfer these files to `submission -> data -> chips` (for training) or `submission -> data -> val_chips` (for validation).
    * Run `dataset_chopper.py`: This generates `.npy` formatted data and matched labels in the data folder.
    * Run `train.py`: Generates the `terramind_tiny_ship_classifier.pth` file.
    * Run `infer.py`: Executes the inference on the `processed_val_chips` folder to generate final metrics.No arguments required for inference as everything is formated to work with the first click if you are only looking for a demo.

---

### Metrics and Numbers
Our baseline is the **YOLOv8m** model, comparable in complexity and size to our fine-tuned TerraMind-Tiny model.

#### Hardware & Logic Comparison
| Metric | TerraMind (Ours) | YOLOv8m (Baseline) |
| :--- | :--- | :--- |
| **Total Parameters** | **5,466,497** | 15,774,898 |
| **Model Size** | **21.09 MB** | 30.21 MB |
| **FLOPs / Sample** | **~1.066 G** | 2.565 G |
| **Batch Inference Time** | **92.96 ms** | 122.41 ms |
| **Throughput (samples/sec)** | **2151.4** | 1633.8 |

The GPU that has been used for testing is **Nvidia RTX 5070ti mobile** and that is translated to a **entry-level NVIDIA Jetson nano** like so:

472 GFLOPs / 1.066 GFLOPs per frame = ~442 FPS

but In actual edge-deployment engineering, we estimate that real-world AI inference runs at about 15% to 20% efficiency of the theoretical max.

442 FPS * 0.15 = ~66 FPS

442 FPS * 0.20 = ~88 FPS

**Important FLOPs Note:** TerraMind FLOPs were calculated with unsupported operator warnings, making this a **lower-bound estimate**. It is useful for relative comparison but is not an absolute hardware count.

#### Final Results
* **YOLOv8m:** 63% Accuracy
* **TerraMind-Tiny (Ours):**

| Metric | Value |
| :--- | :--- |
| **Accuracy** | **73.61%** |
| **Precision** | **81.93%** |
| **Recall** | **78.94%** |
| **F1 Score** | **80.41%** |
| **True Positives** | 390 |
| **True Negatives** | 140 |
| **False Positives** | 86 |
| **False Negatives** | 104 |
| **Decision Threshold** | 0.5 |
| **Test Samples** | 720 |

---

### Model Size Analysis: How we hit 21MB
The 21MB footprint is a direct result of our **High-Efficiency Fine-Tuning Strategy**:
1. **LoRA Rank $r=1$:** We utilized the lowest possible LoRA rank. Instead of updating the full weight matrices, we injected a rank-1 decomposition into the Attention projections (`q_proj`, `v_proj`). This added only ~768 parameters per layer instead of ~150,000.
2. **Tiny Backbone:** The `terramind_v1_tiny` architecture was chosen for its native 5M parameter footprint, which is $3 \times$ smaller than the YOLOv8m baseline.
3. **Bottleneck Classification Head:** Our custom head compresses the final features into a 128-neuron bottleneck before output, keeping the total head size under 50KB.
4. **FP32 Storage:** The total size reflects 5.4M parameters at 4 bytes each, resulting in the optimized 21.09 MB weight file.

---


### Demo Execution

Run the demo:

```bash
python demo.py
````

This launches a Gradio web interface.

### Workflow

* Cycles through true positive detections
* Displays SAR chips with predicted ships
* Simulates AIS matching using latitude/longitude

### Usage

1. Open the Gradio link in your browser
2. Click **Start Simulation**
3. View detections and AIS-aligned positions




### Hackathon Q&A

**What problem are you solving?**
We are solving the "Dark Vessel" problem in maritime surveillance. Malicious actors, illegal fishers, and smugglers disable AIS tracking to disappear from global monitoring systems. Our customer is maritime law enforcement and port authorities. They would pay for this because it provides a real-time, unhackable "ground truth" layer that identifies ships that don't want to be found.

**What did you build?**
We built an edge-ready detection pipeline using a **TerraMind-Tiny** backbone fine-tuned via **LoRA ($r=1$)**. We utilized the SARFish dataset, specifically optimizing for VH/VV modalities. Our fine-tuning recipe includes a GELU-activated detection head and a custom data-balancing script (`dataset_chopper.py`) to handle the extreme empty-ocean bias found in satellite imagery.

**How did you measure it?**
We measured performance against a YOLOv8m baseline. Our model is **3x smaller** in parameters and achieved a **10.6% improvement in accuracy** (73.6% vs 63%), while maintaining significantly higher throughput for real-time processing.

**What's the orbital-compute story?**
This model is designed for **Space Data Center** integration. With a footprint of only **~21MB** and a throughput of **2,151 samples/sec**, it can process massive SAR swaths in real-time on-board the satellite. This fits within the strict power and memory limits of orbital edge hardware (like the NVIDIA Orin Nano), allowing the satellite to downlink only the detected ship coordinates rather than wasting bandwidth on empty ocean data.

**What doesn't work yet?**
Honesty is key: We currently do not have public, real-time access to AIS data streams, so the "verification" step relies on historical or purchased snapshots. Furthermore, the model's accuracy can still fluctuate in high-sea-state conditions (extreme storms) where wave clutter can occasionally mirror metal backscatter.

---

### The Future
The next priority is securing a way to integrate live, public AIS data and improving the model's generalizability across more varied SAR sensors beyond Sentinel-1.
