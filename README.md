# LLM-Distilled Semantic Knowledge for Task-Aware Object Selection on Edge Platforms

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8--nano-green.svg)](https://docs.ultralytics.com/)
[![Sentence-BERT](https://img.shields.io/badge/Sentence--BERT-all--MiniLM--L6--v2-orange.svg)](https://www.sbert.net/)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-purple.svg)](https://openrouter.ai/)
[![Target Hardware](https://img.shields.io/badge/Hardware-RISC--V%20%2B%20FPGA-red.svg)](https://dvcon-india.org/)

An edge-deployable, training-free framework for **task-aware object selection** in cluttered scenes. Unlike conventional detectors that treat all instances of a category identically, this system combines **YOLOv8-nano**, an **offline 47 KB LLM-distilled affordance knowledge bank**, and **OpenRouter dynamic reasoning** to identify, rank, and select the optimal object for any arbitrary human task query while safely rejecting scenes when no suitable object is present.

---

## 📑 Table of Contents
1. [Project Overview & Core Problem](#1-project-overview--core-problem)
2. [Comparison with Base Paper (Sawatzky et al., CVPR 2019)](#2-comparison-with-base-paper-sawatzky-et-al-cvpr-2019)
3. [System Architecture & 4-Stage Pipeline](#3-system-architecture--4-stage-pipeline)
4. [Offline Knowledge Bank Footprint Analysis](#4-offline-knowledge-bank-footprint-analysis)
5. [OpenRouter Dynamic Reasoning for Novel Tasks & Objects](#5-openrouter-dynamic-reasoning-for-novel-tasks--objects)
6. [Hardware Co-Design (RISC-V + FPGA Accelerator)](#6-hardware-co-design-risc-v--fpga-accelerator)
7. [Repository File Structure](#7-repository-file-structure)
8. [Installation & Setup](#8-installation--setup)
9. [How to Run the Project](#9-how-to-run-the-project)
10. [Presentation & Review Preparation Guide (Viva Q&A)](#10-presentation--review-preparation-guide-viva-qa)

---

## 1. Project Overview & Core Problem

### The Semantic Ambiguity Problem
Traditional computer vision frameworks (YOLO, Faster R-CNN, SSD) excel at **object categorization**—identifying that an image contains multiple "chairs", "bottles", or "cutlery". However, standard category detection cannot answer: **Which specific object is suitable for a given task?**

* **Example 1**: An assistive robot supporting an elderly person seeking comfortable seating faces a room with a cushioned couch, a wooden bar stool, a dining chair, and a bed. A standard detector tags them all as "chair/furniture". Selecting a hard stool can cause discomfort or injury.
* **Example 2**: A task asks to "step on something to reach a top shelf". A flimsy decorative chair or fragile glass coffee table is dangerous, while a sturdy wooden table or stepladder is suitable.
* **Example 3**: A user asks to "cut an apple" in a room without knives or scissors. A standard detector will output high-confidence detections for other objects; our system must know to **safely reject** the scene.

---

## 2. Comparison with Base Paper (Sawatzky et al., CVPR 2019)

* **Base Paper**: Johann Sawatzky, Yaser Souri, Christian Grund, Juergen Gall, *"What Object Should I Use? - Task Driven Object Detection"*, CVPR 2019 / arXiv:1904.03000.
* **Benchmark**: COCO-Tasks Dataset (~40,000 images, 14 everyday tasks).

| Architectural Dimension | Base Paper (Sawatzky et al., 2019) | Our Proposed Approach |
| :--- | :--- | :--- |
| **Core Algorithm** | Gated Graph Neural Network (GGNN) with iterative message passing | LLM-distilled semantic vector matching + weighted score fusion |
| **Visual Feature Extractor** | ResNet-101 CNN (heavyweight, ~44M params) | YOLOv8-nano (~3.2M params, CPU/edge friendly) |
| **Semantic Representation** | One-hot category vector + learned graph states | 384-dimensional dense Sentence-BERT vectors (`all-MiniLM-L6-v2`) |
| **Training Requirement** | **Supervised**: Requires training on ~30,000 task-annotated COCO images | **Zero-Shot / Training-Free**: No task training needed; runs out-of-the-box |
| **Inference Dependency** | Must process and aggregate all scene detections simultaneously in a graph | Evaluates each candidate independently with **O(1) array lookup** |
| **Novel / Open-Domain Query Support** | Strictly locked to the 14 training tasks | **Unlimited queries**: OpenRouter dynamically decomposes novel tasks |
| **Edge & Hardware Feasibility** | Impractical for edge (requires GPU, >150 W) | **Edge feasible**: CPU-only (~150 ms) or RISC-V + FPGA (<35 ms, 4.3 W) |
| **Memory Footprint** | >170 MB model weights | **35.5 KB (INT8) / 70.7 KB (FP16)** on-chip knowledge bank |

---

## 3. System Architecture & 4-Stage Pipeline

```
Raw Camera Image + Task Query ("sit comfortable")
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Lightweight Object Detection (YOLOv8-nano)    │
│ -> Detects bounding boxes bi, class ci, confidence Ci  │
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Direct Array Lookup & Semantic Scoring        │
│ -> O(1) embedding lookup from precomputed bank eoi     │
│ -> Real-time task encoding et (Sentence-BERT)          │
│ -> Cosine Similarity: Sim(oi, t) = (eoi . et) / (|eoi||et|)
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Exact Weighted Score Fusion                   │
│ -> Score(oi) = 0.7 * Sim(oi, t) + 0.3 * Conf(oi)       │
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Relevance Thresholding & Selection            │
│ -> Discards candidates if Sim(oi, t) < 0.12 or Score < 0.25 │
│ -> If valid candidates exist: Highlight top-1 (Green)  │
│ -> If all candidates fail: Reject Scene (Red Banner)   │
└────────────────────────────────────────────────────────┘
```

### The 4 Stages Explained:
1. **Stage 1 - Object Detection**: The raw image is passed through YOLOv8-nano. Objects detected with confidence above the base threshold are extracted along with their bounding box coordinates.
2. **Stage 2 - Semantic Scoring**: For each detected object, its 384-dimensional affordance vector is retrieved instantly from the offline knowledge bank in O(1) time. The task query is encoded via Sentence-BERT into unit vector et, and cosine similarity is computed.
3. **Stage 3 - Weighted Fusion**:
   ```
   Score(oi) = alpha * Sim(oi, t) + (1 - alpha) * Conf(oi)
   ```
   where alpha = 0.7 (70% semantic relevance weight, 30% detection confidence weight).
4. **Stage 4 - Relevance Filtering & Selection**: Candidates must pass semantic gating (Sim >= 0.12) and fused thresholding (Score >= 0.25). The top candidate is highlighted with green bounding boxes and score breakdowns. If no object qualifies, the system reports `NO_SUITABLE_OBJECT_FOUND` without false alarms.

---

## 4. Offline Knowledge Bank Footprint Analysis

To run on resource-constrained platforms (such as RISC-V CPUs with limited on-chip SRAM or FPGA Block RAM), the knowledge bank encodes **all 80 COCO categories** and **14 official COCO tasks** into compact embeddings.

File: `generate_knowledge_bank.py`

| Data Structure | Dimensions | Precision | Exact File Size |
| :--- | :--- | :--- | :--- |
| **80 Object Embeddings** | 80 x 384 | Float16 | 61,568 bytes (60.12 KB) |
| **14 Task Embeddings** | 14 x 384 | Float16 | 10,880 bytes (10.62 KB) |
| **80 Object Embeddings (Quantized)** | 80 x 384 | INT8 | 30,848 bytes (30.12 KB) |
| **14 Task Embeddings (Quantized)** | 14 x 384 | INT8 | 5,504 bytes (5.38 KB) |
| **Total INT8 Knowledge Bank** | **94 x 384** | **INT8** | **35.50 KB** |

**Footprint Verification**: The total INT8 bank requires only **35.50 KB**, strictly respecting the **47 KB BRAM constraint** specified in DVCon India hardware budgets.

---

## 5. OpenRouter Dynamic Reasoning for Novel Tasks & Objects

While the offline knowledge bank stores the 80 standard COCO classes, real-world deployment frequently encounters **untrained objects** and **open-domain task queries**. The system integrates **OpenRouter** (`openrouter_engine.py`) for live dynamic LLM distillation:

### A. Novel (Untrained) Objects Outside COCO-80
When an object term outside the 80 categories is provided (e.g. `ladder`, `cushion`, `frying pan`, `wooden stool`, `hammer`, `flashlight`, `sponge`, `teapot`, `screwdriver`):
1. OpenRouter generates a 1-sentence physical affordance description detailing material, shape, rigidity, and functional utility.
2. Sentence-BERT encodes the description into 384 dimensions.
3. The embedding is cached into `knowledge_bank/openrouter_cache.json` so the API is never called twice for the same object.

### B. Novel Complex Task Decomposition
When an arbitrary task query is entered (e.g. `"reheat leftover food"`, `"reach top shelf"`, `"study for exam"`):
* The LLM decomposes the user goal into required physical characteristics (e.g. thermal chamber, elevated rigid support, flat surface) before vector matching, drastically increasing cosine similarity with suitable tools.

### C. Active OpenRouter Models & Failover
* Default Model: `openrouter/auto` (automatically selects active free endpoints).
* Fallbacks: `google/gemma-4-26b-a4b-it:free`, `nvidia/nemotron-3.5-lightning:free`, `minimax/minimax-m3.1-flash:free`, `liquid/lfm-2.5-2.6b:free`.
* Offline Fallback: If offline or without an API key, the system uses the local offline knowledge bank without crashing.

---

## 6. Hardware Co-Design (RISC-V + FPGA Accelerator)

For the **DVCon India 2026 Design Contest**, the architecture is partitioned across a heterogeneous platform:

```
┌────────────────────────────────────────────────────────┐
│           RISC-V Vega Processor (1.4 GHz)              │
│  - Stage 1: YOLO-tiny / nano INT8 Detection   (25 ms)  │
│  - Stage 2: Feature Extraction & Setup         (5 ms)  │
│  - Stage 4: Candidate Sorting & Selection      (2 ms)  │
└──────────────────────────┬─────────────────────────────┘
                           │ 32-bit AXI4 Bus (100 MHz)
                           ▼
┌────────────────────────────────────────────────────────┐
│          Genesys-2 Kintex-7 FPGA Accelerator           │
│  - 4x Parallel Dot-Product Engines (DSP Slices)        │
│  - 512 DSP slices utilized (61% utilization)           │
│  - Stage 3: Cosine Similarity Matching         (3 ms)  │
│  - Knowledge Bank stored in Block RAM         (35.5 KB)│
└────────────────────────────────────────────────────────┘
```

* **Total End-to-End Latency**: 35 ms (~28 FPS real-time operation)
* **System Power Consumption**: 4.3 W (< 5.0 W battery budget for assistive mobile robots)
* **FPGA Resource Utilization**: 42% LUTs, 29% FFs, 40% BRAM, 61% DSP slices

---

## 7. Repository File Structure

```
├── app.py                                  # Interactive Streamlit Web Dashboard (Tabs for Scene & Novel Objects)
├── run_app.bat                             # One-click Windows launcher for Web App
├── task_aware_detector.py                  # Core pipeline module (Detection, O(1) Lookup, Fusion, Visualizer)
├── openrouter_engine.py                    # OpenRouter LLM dynamic distillation & caching engine
├── generate_knowledge_bank.py              # Generator for 80-class offline INT8/FP16 .npy knowledge bank
├── run_demo.py                             # Verification script running benchmark test queries
├── test_novel_objects_demo.py              # Zero-shot test on untrained objects (ladder, cushion, frying pan, etc.)
├── test_novel_terms_openrouter.py          # OpenRouter affordance expansion test script
│
├── knowledge_bank/                         # Offline Precomputed Knowledge Bank (~35.5 KB INT8)
│   ├── coco80_knowledge_bank_int8.npy      # 80 COCO classes INT8 quantized embeddings (30.1 KB)
│   ├── coco80_knowledge_bank_fp16.npy      # 80 COCO classes FP16 embeddings (60.1 KB)
│   ├── coco14_tasks_int8.npy               # 14 standard COCO-Tasks INT8 embeddings (5.4 KB)
│   ├── coco14_tasks_fp16.npy               # 14 standard COCO-Tasks FP16 embeddings (10.6 KB)
│   ├── coco80_classes.json                 # Class mappings and textual affordance descriptions
│   ├── coco14_tasks.json                   # 14 official task descriptions
│   └── openrouter_cache.json               # Dynamic cache for newly distilled novel objects
│
├── sample_living_room_raw.jpg              # Clean raw test photo: living room (no bounding boxes)
├── sample_kitchen_raw.jpg                  # Clean raw test photo: kitchen & dining (no bounding boxes)
├── decorate.jpg / decorate1.jpg            # Benchmark reference photos
├── yolov8n.pt                              # YOLOv8-nano pretrained PyTorch model (~6.5 MB)
│
├── Task_Aware_Object_Selection_Pipeline.ipynb  # Comprehensive Jupyter / Google Colab Notebook
├── DVCon_stage_2_submission/               # DVCon India Stage 2A Submission artifacts
│   ├── DVCon stage 2 submission.pdf        # Stage 2A Report
│   └── DVCon_stage2A_updated.ipynb         # Updated Stage 2A Notebook
│
├── AWS REVIEW 1  (1).pdf                   # College Project Review 1 Report
├── aws_review_ieee_format.pdf              # College Project Review 1 in IEEE Paper Format
├── Dvcon stage 1 submission.pdf            # DVCon India 2026 Stage 1 Design Report
└── base paper.pdf                          # Base research paper (Sawatzky et al., CVPR 2019)
```

---

## 8. Installation & Setup

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Windows, Linux, or macOS

### Clone the Repository
```bash
git clone https://github.com/Shailesh-A-hub/task-aware-object-selection.git
cd task-aware-object-selection
```

### Install Dependencies
```bash
pip install ultralytics sentence-transformers requests opencv-python matplotlib pandas streamlit
```

*(Optional) Set OpenRouter API Key in your terminal:*
```bash
# Windows Command Prompt:
set OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Linux / macOS:
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

---

## 9. How to Run the Project

### Option A: Launch Interactive Web Dashboard (Recommended for Presentation)
Double-click `run_app.bat` on Windows, or run:
```bash
streamlit run app.py
```
This opens `http://localhost:8501` in your browser with:
* **Tab 1: 🖼️ Image Scene Selection**:
  * Select clean raw photos (`sample_living_room_raw.jpg` or `sample_kitchen_raw.jpg`) or upload any photo from your phone.
  * Enter manual queries or pick from the 14 standard COCO tasks.
  * Adjust Alpha (`0.70`) and Threshold (`0.25`) sliders live.
  * View latency metrics, candidate ranking tables, and bounding box visualizations.
* **Tab 2: 🧪 Novel (Untrained) Objects Playground**:
  * Test objects completely outside COCO (e.g. `ladder`, `cushion`, `frying pan`, `hammer`, `sponge`).
  * Watch OpenRouter distill affordances and produce a ranked leaderboard with real-time bar charts.

### Option B: Run Command-Line Verification Scripts
```bash
# Run multi-query benchmark test on living room scene:
python run_demo.py

# Run zero-shot evaluation on untrained novel objects:
python test_novel_objects_demo.py
```

### Option C: Run in Google Colab / Jupyter
Open `Task_Aware_Object_Selection_Pipeline.ipynb` in Jupyter Notebook or upload to Google Colab.

---

## 10. Presentation & Review Preparation Guide (Viva Q&A)

Use this section to prepare for reviewer questions during project reviews and design contest evaluations:

### Q1: What makes your method "training-free"?
**Answer:** The base paper trained a 44M-parameter ResNet-101 and a Gated Graph Neural Network across 30,000 images using custom human preference annotations. Our approach is zero-shot: we use a general-purpose pretrained detector (YOLOv8n) and match pre-computed language embeddings of physical affordances. We require zero task-specific model weight updates or loss backpropagation.

### Q2: Why not just use CLIP (Contrastive Language-Image Pretraining)?
**Answer:** CLIP is a powerful vision-language model, but:
1. It is computationally expensive (>150 W GPU requirement), making it impossible to deploy on low-power mobile assistive robots.
2. Models like TaskCLIP struggle to localize multiple object candidates in real-time on CPU.
Our pipeline decouples lightweight object localization (YOLOv8n, ~3.2M params on CPU) from semantic similarity (Sentence-BERT in 15 ms / FPGA dot-product in 3 ms), consuming only 4.3 W.

### Q3: Why is weighted score fusion needed? Why not rely 100% on semantic similarity?
**Answer:** Formula: `Score = 0.7 * Sim(oi, t) + 0.3 * Conf(oi)`.
Semantic similarity measures functional suitability, but if the detector is uncertain (e.g. low visual confidence due to heavy occlusion or blur), relying solely on language matching could select a hallucinated or misclassified object. Weighting detection confidence ensures the selected object is both functionally suitable and visually confirmed.

### Q4: How do you handle scenes where NO suitable object exists?
**Answer:** We implement two gates:
1. **Semantic Gating**: Objects must have `Sim(oi, t) >= 0.12`.
2. **Relevance Threshold**: Final fused score must be `>= 0.25`.
If you ask for `"cut something"` in a living room containing only couches and books, all candidates fail the threshold, and the system outputs `NO_SUITABLE_OBJECT_FOUND` with a red banner instead of falsely selecting an inappropriate item.

### Q5: How does the system handle objects that were NEVER seen during training?
**Answer:** When an open-vocabulary object (like `ladder` or `frying pan`) is entered, OpenRouter dynamically distills a 1-sentence physical affordance description focusing on structural and functional properties. Sentence-BERT encodes this into a 384-dimensional vector, and it is cached for instant future reuse.

### Q6: How does this fit into the DVCon 2026 hardware constraints?
**Answer:**
* **BRAM Memory**: The entire 80-class + 14-task knowledge bank quantizes to **35.5 KB INT8**, fitting comfortably inside FPGA Block RAM (under the 47 KB budget).
* **Latency**: Offloading the 384-d cosine dot-products to 4 parallel DSP engines on the Kintex-7 FPGA executes in 3 ms. Total system latency is 35 ms, exceeding the 50 ms real-time requirement.
* **Power**: System draws 4.3 W, well within the 5.0 W battery budget.

---

## 👥 Authors & Team
* **Team**: RTL Rebels (Team 616)
* **Members**: Shailesh A., Pradeep P., Sanjay K. (VIT Vellore)
* **Advisor / Subject**: Fall Semester 2026-27 | DVCon India 2026 Design Contest
