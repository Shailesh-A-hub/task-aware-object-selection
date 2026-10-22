"""
Generates the comprehensive, publication-ready Jupyter Notebook:
Task_Aware_Object_Selection_Pipeline.ipynb
Fully aligned with the CPU-only framework and featuring:
- COCO-Tasks vs OpenRouter Comparison
- Zero-Shot Novel Object / Task Playground
- How to launch the Streamlit Web Dashboard in Colab
"""

import json

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# LLM-Distilled Semantic Knowledge for Task-Aware Object Selection on CPU\n",
                "### Final Review Submission | Academic Computer Vision Project\n",
                "**Authors:** Shailesh A., Pradeep P., Sanjay K. (VIT Vellore)\n",
                "\n",
                "---\n",
                "### Architecture Highlights:\n",
                "1. **Detection Engine**: YOLOv8-nano (~3.2M params, CPU lightweight).\n",
                "2. **Dual-Tier Semantic Engine**:\n",
                "   - **Tier 1 (Offline)**: 80 COCO classes + 14 standard tasks in an ultra-compact **35.5 KB INT8** knowledge bank for sub-1 ms O(1) array lookup.\n",
                "   - **Tier 2 (Dynamic Online)**: OpenRouter LLM integration for open-domain novel queries (e.g. *\"bake\"*, *\"phone call\"*) and open-vocabulary objects (*\"ladder\"*, *\"frying pan\"*, *\"cushion\"*).\n",
                "3. **Exact Weighted Score Fusion**: `Score = 0.7 * Sim(oi, t) + 0.3 * Conf(oi)`.\n",
                "4. **Dual Threshold Gating**: Discards candidates below 0.12 semantic similarity or 0.25 fused relevance score, safely handling negative scenes (*NO_SUITABLE_OBJECT_FOUND*).\n",
                "5. **Edge Feasible**: Runs on standard CPU hardware (~140-150 ms end-to-end latency, zero GPU requirement)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 1: Install required dependencies\n",
                "!pip install ultralytics sentence-transformers requests opencv-python matplotlib"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 2: Clone repository if running directly in Colab (optional)\n",
                "import os\n",
                "if not os.path.exists('task_aware_detector.py'):\n",
                "    !git clone https://github.com/Shailesh-A-hub/task-aware-object-selection.git .\n",
                "print(\"Repository files ready!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 3: Import modules and verify CPU environment\n",
                "import os\n",
                "import json\n",
                "import time\n",
                "import numpy as np\n",
                "import cv2\n",
                "import matplotlib.pyplot as plt\n",
                "from ultralytics import YOLO\n",
                "from sentence_transformers import SentenceTransformer\n",
                "import requests\n",
                "\n",
                "from task_aware_detector import TaskAwareDetector\n",
                "from openrouter_engine import OpenRouterEngine\n",
                "\n",
                "print(\"All libraries successfully imported!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Initialize Task-Aware Detector (CPU)\n",
                "Loads YOLOv8-nano and the 35.5 KB INT8 precomputed offline knowledge bank."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Set your OpenRouter API key (or leave empty to test offline mode)\n",
                "OPENROUTER_KEY = os.environ.get(\"OPENROUTER_API_KEY\", \"\")\n",
                "\n",
                "detector = TaskAwareDetector(\n",
                "    knowledge_bank_dir=\"knowledge_bank\",\n",
                "    yolo_model_path=\"yolov8n.pt\",\n",
                "    use_int8=True,       # 35.5 KB INT8 compact knowledge bank\n",
                "    openrouter_api_key=OPENROUTER_KEY,\n",
                "    openrouter_model=\"openrouter/auto\"\n",
                ")\n",
                "print(\"TaskAwareDetector loaded successfully on CPU!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. COCO-Tasks (Standard 14 Benchmark Tasks)\n",
                "Tests a standard COCO task query (e.g. *\"sit comfortable\"*) on a raw living room image.\n",
                "In this mode, detected objects are looked up via the precomputed **35.5 KB INT8 knowledge bank** in **sub-1 ms O(1)** time."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "image_path = \"sample_living_room_raw.jpg\"\n",
                "task_query = \"sit comfortable\"\n",
                "\n",
                "res = detector.select_best_object(image_path, task_query, decompose_task=False)\n",
                "\n",
                "print(f\"--- Task: '{task_query}' ---\")\n",
                "print(f\"Selected Object: {res['selected_object']}\")\n",
                "print(f\"Fused Score:     {res['score']:.4f}\")\n",
                "print(f\"Semantic Sim:    {res['semantic_similarity']:.4f}\")\n",
                "print(f\"Visual Conf:     {res['detection_confidence']:.4f}\")\n",
                "print(f\"Total CPU Time:  {res['timings']['total_ms']:.1f} ms\")\n",
                "\n",
                "vis_img = detector.visualize(image_path, res, output_path=\"output_sit_comfortable.jpg\")\n",
                "plt.figure(figsize=(10, 6))\n",
                "plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))\n",
                "plt.axis(\"off\")\n",
                "plt.title(f\"COCO-Tasks Mode: '{task_query}' -> Selected: {res['selected_object']}\")\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. OpenRouter Integration for Novel Untrained Tasks\n",
                "Standard COCO-Tasks is restricted to 14 predefined tasks. If you ask for *\"bake\"*, standard models have no entry.\n",
                "With **OpenRouter Dynamic Reasoning**, the LLM decomposes *\"bake\"* into thermal appliance affordances, matching *microwave* in a kitchen scene without retraining!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "kitchen_img = \"sample_kitchen_raw.jpg\"\n",
                "novel_task = \"bake\"\n",
                "\n",
                "res_bake = detector.select_best_object(kitchen_img, novel_task, decompose_task=True)\n",
                "\n",
                "print(f\"--- Novel Task: '{novel_task}' (OpenRouter Decomposed) ---\")\n",
                "print(f\"Selected Object: {res_bake['selected_object']}\")\n",
                "print(f\"Fused Score:     {res_bake['score']:.4f}\")\n",
                "print(f\"Semantic Sim:    {res_bake['semantic_similarity']:.4f}\")\n",
                "print(f\"Visual Conf:     {res_bake['detection_confidence']:.4f}\")\n",
                "\n",
                "vis_bake = detector.visualize(kitchen_img, res_bake, output_path=\"output_bake.jpg\")\n",
                "plt.figure(figsize=(10, 6))\n",
                "plt.imshow(cv2.cvtColor(vis_bake, cv2.COLOR_BGR2RGB))\n",
                "plt.axis(\"off\")\n",
                "plt.title(f\"OpenRouter Mode: '{novel_task}' -> Selected: {res_bake['selected_object']}\")\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Negative Scene Rejection Test\n",
                "When asked to *\"cut something\"* in a living room with no blades or scissors, the system must recognize that no suitable object exists and safely abort."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "neg_res = detector.select_best_object(\"sample_living_room_raw.jpg\", \"cut something\")\n",
                "print(f\"Status: {neg_res['status']}\")\n",
                "\n",
                "vis_neg = detector.visualize(\"sample_living_room_raw.jpg\", neg_res, output_path=\"output_negative.jpg\")\n",
                "plt.figure(figsize=(10, 6))\n",
                "plt.imshow(cv2.cvtColor(vis_neg, cv2.COLOR_BGR2RGB))\n",
                "plt.axis(\"off\")\n",
                "plt.title(f\"Negative Scene: 'cut something' -> {neg_res['status']}\")\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Open-Vocabulary Novel Objects Playground (Beyond COCO-80)\n",
                "Tests objects that **do not exist in the 80 COCO categories** (*ladder*, *cushion*, *frying pan*, *hammer*, *sponge*).\n",
                "OpenRouter distills physical affordance vectors for each novel term dynamically."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "novel_objects = [\"ladder\", \"cushion\", \"frying pan\", \"hammer\", \"sponge\"]\n",
                "novel_tasks = [\n",
                "    \"reach top shelf to grab books\",\n",
                "    \"make a hard wooden chair softer\",\n",
                "    \"fry an egg for breakfast\",\n",
                "    \"drive a steel nail into wall\",\n",
                "    \"soak up water spill on the counter\"\n",
                "]\n",
                "\n",
                "print(f\"Candidate Pool: {novel_objects}\\n\")\n",
                "for t in novel_tasks:\n",
                "    t_vec = detector.get_task_embedding(t, decompose=True)\n",
                "    scores = []\n",
                "    for obj in novel_objects:\n",
                "        o_vec = detector.get_object_embedding(obj)\n",
                "        sim = float(np.dot(o_vec, t_vec))\n",
                "        scores.append((obj, sim))\n",
                "    scores.sort(key=lambda x: x[1], reverse=True)\n",
                "    best_obj, best_sim = scores[0]\n",
                "    print(f\"Task: '{t}' -> Best Object: '{best_obj}' (Similarity: {best_sim:.4f})\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. How to Run the Interactive Streamlit Web Dashboard in Google Colab\n",
                "You can run the full interactive dashboard inside Google Colab using `localtunnel`.\n",
                "Run the cell below, click the generated public URL, and the complete Streamlit UI will open in your browser!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run this cell to launch the Web Dashboard inside Google Colab!\n",
                "!npm install -g localtunnel\n",
                "!curl ipv4.icanhazip.com\n",
                "!streamlit run app.py & npx localtunnel --port 8501"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("Task_Aware_Object_Selection_Pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Successfully generated updated Task_Aware_Object_Selection_Pipeline.ipynb!")
