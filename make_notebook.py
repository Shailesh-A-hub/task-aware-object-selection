"""
Generates the updated, comprehensive Jupyter Notebook:
Task_Aware_Object_Selection_Pipeline.ipynb
"""

import json

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# LLM-Distilled Semantic Knowledge for Task-Aware Object Selection\n",
                "### DVCon India 2026 / Academic Project - Complete Pipeline\n",
                "**Authors:** Shailesh A., Pradeep P., Sanjay K. (Team RTL Rebels 616, VIT Vellore)\n",
                "\n",
                "---\n",
                "### Architecture Highlights:\n",
                "1. **Detection Engine**: YOLOv8-nano (CPU lightweight).\n",
                "2. **Offline Knowledge Bank**: 80 COCO classes + 14 standard tasks encoded as 384-d vectors, quantized to **~35 KB (INT8) / ~70 KB (FP16)**.\n",
                "3. **Zero-Latency Direct Array Lookup**: O(1) embedding lookup from precomputed arrays (eliminating runtime text encoding for detected objects).\n",
                "4. **Exact Weighted Fusion**: `Score = 0.7 * Sim(oi, t) + 0.3 * Conf(oi)`.\n",
                "5. **Semantic Relevance Gating & Thresholding**: Discards objects below 0.25 threshold or with low semantic relevance.\n",
                "6. **OpenRouter Integration**: Dynamic LLM affordance distillation for novel objects and open-domain task queries."
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
                "# Step 2: Import modules\n",
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
                "print(\"All libraries successfully imported!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Load Precomputed Offline Knowledge Bank (~47 KB Footprint)\n",
                "Loads the precomputed 80-class COCO semantic knowledge bank. No text encoding is needed for detected objects at runtime."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load Knowledge Bank metadata and embeddings\n",
                "from task_aware_detector import TaskAwareDetector\n",
                "\n",
                "# Optional: Set your OpenRouter API Key for novel terms / tasks\n",
                "# os.environ[\"OPENROUTER_API_KEY\"] = \"your_openrouter_api_key_here\"\n",
                "\n",
                "detector = TaskAwareDetector(\n",
                "    yolo_model_path=\"yolov8n.pt\",\n",
                "    use_quantized_int8=False, # Set True for 35.5 KB INT8 footprint!\n",
                "    alpha=0.7,\n",
                "    threshold=0.25\n",
                ")\n",
                "\n",
                "print(f\"Knowledge bank loaded successfully!\")\n",
                "print(f\"Total precomputed classes: {len(detector.class_names)}\")\n",
                "print(f\"Embedding shape: {detector.obj_embeddings.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Test on an Image\n",
                "Upload or specify an image, then run task selection queries."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "image_path = \"decorate.jpg\" # Or specify your uploaded image path\n",
                "\n",
                "query = \"sit comfortable\"\n",
                "res = detector.select_best_object(image_path, query)\n",
                "\n",
                "print(f\"Task Query: '{res['task']}'\")\n",
                "print(f\"Status: {res['status']}\")\n",
                "print(f\"Latency: Detection = {res['timing_ms']['detection']} ms | Semantic = {res['timing_ms']['semantic_matching']} ms | Total = {res['timing_ms']['total']} ms\")\n",
                "\n",
                "print(\"\\n--- Ranked Candidates (Score = 0.7 * Sim + 0.3 * Conf) ---\")\n",
                "for c in res[\"all_candidates\"]:\n",
                "    print(f\"{c['name']:<14} | Fused: {c['fused_score']:.4f} | Sim: {c['semantic_similarity']:.4f} | Conf: {c['confidence']:.4f} | Valid: {c['is_relevant']}\")\n",
                "\n",
                "if res[\"selected_object\"]:\n",
                "    sel = res[\"selected_object\"]\n",
                "    print(f\"\\n>>> SELECTED: {sel['name']} with Fused Score: {sel['fused_score']}\")\n",
                "else:\n",
                "    print(\"\\n>>> NO SUITABLE OBJECT FOUND. Scene correctly rejected!\")\n",
                "\n",
                "# Display Visualization\n",
                "vis_img = detector.visualize(image_path, res, output_path=\"output_sit_comfortable.jpg\")\n",
                "plt.figure(figsize=(12, 8))\n",
                "plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))\n",
                "plt.axis(\"off\")\n",
                "plt.title(f\"Query: {query} -> Status: {res['status']}\")\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Threshold Rejection Verification (Negative Query Case)\n",
                "Testing query `'cut something'` on a living room scene with no cutting tools."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "negative_query = \"cut something\"\n",
                "neg_res = detector.select_best_object(image_path, negative_query)\n",
                "\n",
                "print(f\"Negative Query: '{neg_res['task']}'\")\n",
                "print(f\"Status: {neg_res['status']}\")\n",
                "\n",
                "vis_neg = detector.visualize(image_path, neg_res, output_path=\"output_cut_something.jpg\")\n",
                "plt.figure(figsize=(12, 8))\n",
                "plt.imshow(cv2.cvtColor(vis_neg, cv2.COLOR_BGR2RGB))\n",
                "plt.axis(\"off\")\n",
                "plt.title(f\"Query: {negative_query} -> Status: {neg_res['status']}\")\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. OpenRouter Integration for Novel Terms and Tasks\n",
                "Demonstrates how the pipeline handles novel objects outside COCO-80 and complex tasks."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Testing novel open-vocabulary objects outside COCO-80\n",
                "novel_term = \"ladder\"\n",
                "emb = detector.get_object_embedding(novel_term)\n",
                "print(f\"Novel Object: '{novel_term}' successfully embedded: shape {emb.shape}\")\n",
                "\n",
                "# Testing novel open-domain task\n",
                "novel_task = \"reach high ceiling fan\"\n",
                "t_emb = detector.get_task_embedding(novel_task, decompose=True)\n",
                "print(f\"Novel Task: '{novel_task}' processed: shape {t_emb.shape}\")"
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
print("Saved Task_Aware_Object_Selection_Pipeline.ipynb successfully!")
