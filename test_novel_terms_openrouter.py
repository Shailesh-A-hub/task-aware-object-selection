"""
Test script for OpenRouter integration:
1. Demonstrates dynamic affordance distillation for novel objects outside the 80 COCO classes.
2. Demonstrates task decomposition for novel user task queries beyond standard COCO-Tasks.
"""

import os
from task_aware_detector import TaskAwareDetector

def test_novel_capabilities():
    print("="*70)
    print("TESTING NOVEL TERMS & OPENROUTER AFFORDANCE EXPANSION")
    print("="*70)

    # You can supply an OpenRouter API key here or leave None for smart fallback
    api_key = os.environ.get("OPENROUTER_API_KEY", None)
    detector = TaskAwareDetector(openrouter_api_key=api_key, alpha=0.7, threshold=0.25)

    print("\n--- 1. Testing Novel Object Terms (Outside COCO-80) ---")
    novel_objects = ["cushion", "ladder", "pliers", "flashlight", "frying pan"]
    for obj in novel_objects:
        # Check lookup vs dynamic distillation
        emb = detector.get_object_embedding(obj)
        cached_desc = detector.openrouter.cache.get(f"obj:{obj}", "N/A")
        print(f"Object: {obj:<12} -> Embedding shape: {emb.shape} | Description: {cached_desc}")

    print("\n--- 2. Testing Novel Task Decomposition ---")
    novel_tasks = [
        "reach high ceiling fan",
        "prop open heavy door",
        "protect head from heavy rain"
    ]
    for task in novel_tasks:
        decomposed = detector.openrouter.decompose_novel_task(task)
        t_emb = detector.get_task_embedding(task, decompose=True)
        print(f"User Task: '{task}'")
        print(f"  Decomposed Affordance: '{decomposed}'")
        print(f"  Task vector shape: {t_emb.shape}\n")

if __name__ == "__main__":
    test_novel_capabilities()
