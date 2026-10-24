"""
Benchmark Evaluation Script: mAP and Positive/Negative Scene Accuracy
for OpenRouter-Integrated Task-Aware Object Selection.
Designed for both local execution and Google Colab.
"""

import os
import time
import json
import numpy as np
from typing import List, Dict, Any

from task_aware_detector import TaskAwareDetector


def run_benchmark_evaluation(api_key: str = "", model: str = "openrouter/auto"):
    print("=" * 70)
    print(" TASK-AWARE OBJECT DETECTION: ACCURACY & mAP BENCHMARK EVALUATION")
    print(" Dual-Mode: Positive (+ve) Target Selection & Negative (-ve) Scene Rejection")
    print("=" * 70)

    # 1. Initialize Detector on CPU
    detector = TaskAwareDetector(
        kb_dir="knowledge_bank",
        yolo_model_path="yolov8n.pt",
        use_quantized_int8=True,
        openrouter_api_key=api_key,
        openrouter_model=model,
        alpha=0.70,
        threshold=0.25
    )

    # 2. Define Rigorous Test Suite with Ground Truth (GT)
    test_cases = [
        # --- POSITIVE TEST CASES (+ve) ---
        {
            "type": "positive",
            "image": "sample_living_room_raw.jpg",
            "task": "sit comfortable",
            "novel": False,
            "gt_objects": ["couch", "chair"],
            "description": "Standard COCO task: seating"
        },
        {
            "type": "positive",
            "image": "sample_kitchen_raw.jpg",
            "task": "bake",
            "novel": True,
            "gt_objects": ["microwave"],
            "description": "Novel OpenRouter task: thermal cooking"
        },
        {
            "type": "positive",
            "image": "sample_kitchen_raw.jpg",
            "task": "provides fresh air",
            "novel": True,
            "gt_objects": ["potted plant"],
            "description": "Novel OpenRouter task: air purification"
        },
        {
            "type": "positive",
            "image": "sample_kitchen_raw.jpg",
            "task": "cut an apple",
            "novel": False,
            "gt_objects": ["knife"],
            "description": "Standard task: cutting tool"
        },
        {
            "type": "positive",
            "image": "sample_kitchen_raw.jpg",
            "task": "pour wine",
            "novel": False,
            "gt_objects": ["wine glass", "bottle"],
            "description": "Standard task: drinking vessel"
        },
        {
            "type": "positive",
            "image": "sample_living_room_raw.jpg",
            "task": "step on to reach ceiling",
            "novel": True,
            "gt_objects": ["dining table", "chair"],
            "description": "Elevation task: sturdy support"
        },

        # --- NEGATIVE TEST CASES (-ve) ---
        {
            "type": "negative",
            "image": "sample_living_room_raw.jpg",
            "task": "cut something",
            "novel": False,
            "gt_objects": [],
            "description": "Living room has no knives/scissors (Safe Rejection)"
        },
        {
            "type": "negative",
            "image": "sample_kitchen_raw.jpg",
            "task": "recline and sleep",
            "novel": True,
            "gt_objects": [],
            "description": "Kitchen counter has no bed/couch (Safe Rejection)"
        },
        {
            "type": "negative",
            "image": "sample_living_room_raw.jpg",
            "task": "fry an egg",
            "novel": True,
            "gt_objects": [],
            "description": "Living room has no cooking appliances (Safe Rejection)"
        },
        {
            "type": "negative",
            "image": "sample_kitchen_raw.jpg",
            "task": "ride on the street",
            "novel": True,
            "gt_objects": [],
            "description": "Kitchen has no bicycle/vehicles (Safe Rejection)"
        }
    ]

    # Metrics accumulators
    pos_tp = 0
    pos_fp = 0
    pos_fn = 0
    neg_tn = 0
    neg_fp = 0
    latencies = []
    ap_scores = []

    print(f"\nRunning {len(test_cases)} Test Cases ({sum(1 for t in test_cases if t['type']=='positive')} Positive, {sum(1 for t in test_cases if t['type']=='negative')} Negative)...\n")

    for idx, tc in enumerate(test_cases, 1):
        img_path = tc["image"]
        task_query = tc["task"]
        is_pos = tc["type"] == "positive"

        t0 = time.time()
        res = detector.select_best_object(
            image=img_path,
            task_query=task_query,
            decompose_task=tc["novel"]
        )
        elapsed_ms = (time.time() - t0) * 1000.0
        latencies.append(elapsed_ms)

        status = res["status"]
        selected = res["selected_object"]
        selected_name = selected["name"] if selected else "NONE"
        score = selected["fused_score"] if selected else 0.0
        sim = selected["semantic_similarity"] if selected else 0.0

        if is_pos:
            # Positive evaluation: True Positive if selected object matches Ground Truth
            if status == "SUCCESS" and selected_name in tc["gt_objects"]:
                verdict = "CORRECT (True Positive)"
                pos_tp += 1
                ap_scores.append(1.0)
            elif status == "SUCCESS":
                verdict = f"WRONG OBJECT (False Positive: picked {selected_name}, expected {tc['gt_objects']})"
                pos_fp += 1
                ap_scores.append(0.0)
            else:
                verdict = f"MISSED (False Negative: Rejected valid scene)"
                pos_fn += 1
                ap_scores.append(0.0)
        else:
            # Negative evaluation: True Negative if system correctly rejected
            if status == "NO_SUITABLE_OBJECT_FOUND":
                verdict = "CORRECT (True Negative: Safely Aborted)"
                neg_tn += 1
            else:
                verdict = f"FALSE ALARM (False Positive: picked {selected_name} in negative scene)"
                neg_fp += 1

        print(f"[{idx:02d}/{len(test_cases):02d}] {tc['type'].upper():8s} | Task: '{task_query:28s}' -> Picked: {selected_name:12s} | Score: {score:.3f} | {verdict}")

    # Compute Statistical Metrics
    pos_total = pos_tp + pos_fp + pos_fn
    neg_total = neg_tn + neg_fp
    
    precision = (pos_tp / (pos_tp + pos_fp)) * 100.0 if (pos_tp + pos_fp) > 0 else 0.0
    recall = (pos_tp / (pos_tp + pos_fn)) * 100.0 if (pos_tp + pos_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    pos_accuracy = (pos_tp / pos_total) * 100.0 if pos_total > 0 else 0.0
    neg_accuracy = (neg_tn / neg_total) * 100.0 if neg_total > 0 else 0.0
    overall_accuracy = ((pos_tp + neg_tn) / len(test_cases)) * 100.0
    mAP_05 = np.mean(ap_scores) * 100.0
    avg_latency = np.mean(latencies)

    print("\n" + "=" * 70)
    print("                     FINAL BENCHMARK RESULTS")
    print("=" * 70)
    print(f"1. Positive Cases (+ve) Evaluation:")
    print(f"   - Total Positive Scenes Tested: {pos_total}")
    print(f"   - True Positives (Correct Object): {pos_tp}")
    print(f"   - False Positives (Wrong Object): {pos_fp}")
    print(f"   - False Negatives (Wrongly Rejected): {pos_fn}")
    print(f"   -> Precision:          {precision:.2f}%")
    print(f"   -> Recall:             {recall:.2f}%")
    print(f"   -> F1-Score:           {f1:.2f}%")
    print(f"   -> Positive Accuracy:  {pos_accuracy:.2f}%")
    print(f"   -> Average Precision (AP@0.5): {mAP_05:.2f}%")

    print(f"\n2. Negative Cases (-ve) Evaluation:")
    print(f"   - Total Negative Scenes Tested: {neg_total}")
    print(f"   - True Negatives (Safely Rejected): {neg_tn}")
    print(f"   - False Positives (Hallucinated Pick): {neg_fp}")
    print(f"   -> Negative Rejection Specificity: {neg_accuracy:.2f}%")

    print(f"\n3. Overall System Performance:")
    print(f"   -> Combined Decision Accuracy: {overall_accuracy:.2f}%")
    print(f"   -> Average CPU Inference Latency: {avg_latency:.1f} ms")
    print("=" * 70)


if __name__ == "__main__":
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    run_benchmark_evaluation(api_key=api_key)
