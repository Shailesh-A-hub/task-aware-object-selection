"""
Demo test script demonstrating the complete Task-Aware Object Selection pipeline.
Tests multiple queries on the sample image:
1. 'sit comfortable' -> should select couch/chair
2. 'place flowers' -> should select vase
3. 'cut something' -> should trigger threshold rejection (safe negative handling)
4. Novel task 'keep cold beverages' or 'relax with a cup of coffee'
"""

import os
from task_aware_detector import TaskAwareDetector

def main():
    image_path = "decorate.jpg"
    if not os.path.exists(image_path):
        image_path = "decorate1.jpg"

    print(f"Testing on image: {image_path}")
    detector = TaskAwareDetector(use_quantized_int8=False, alpha=0.7, threshold=0.25)

    test_queries = [
        "sit comfortable",
        "place flowers",
        "water plant",
        "cut something"
    ]

    for q in test_queries:
        print("\n" + "="*60)
        print(f"QUERY: '{q}'")
        res = detector.select_best_object(image_path, q)
        print(f"Status: {res['status']}")
        print(f"Latencies (ms): Detection={res['timing_ms']['detection']}ms, Semantic Matching={res['timing_ms']['semantic_matching']}ms, Total={res['timing_ms']['total']}ms")
        
        print("\nCandidate Rankings (Fused Score = 0.7 * Sim + 0.3 * Conf):")
        for c in res["all_candidates"]:
            print(f" - {c['name']:<14} | Fused: {c['fused_score']:.4f} | Sim(70%): {c['semantic_similarity']:.4f} | Conf(30%): {c['confidence']:.4f}")
            
        if res["selected_object"]:
            sel = res["selected_object"]
            print(f"\n>>> FINAL SELECTED OBJECT: {sel['name']} with Score: {sel['fused_score']:.4f}")
        else:
            print(f"\n>>> NO OBJECT EXCEEDED THRESHOLD {res['threshold']}. Correctly rejected!")

        out_img = f"output_demo_{q.replace(' ', '_')}.jpg"
        detector.visualize(image_path, res, output_path=out_img)

if __name__ == "__main__":
    main()
