from task_aware_detector import TaskAwareDetector
import numpy as np

detector = TaskAwareDetector()

# A list of objects NEVER seen in COCO training:
novel_objects = ["ladder", "cushion", "frying pan", "wooden stool", "hammer", "flashlight", "sponge"]

test_cases = [
    ("reach top shelf", ["ladder", "wooden stool"]),
    ("fry an egg", ["frying pan"]),
    ("wipe spilled milk", ["sponge"]),
    ("hang picture frame on wall", ["hammer"]),
    ("make hard chair softer to sit on", ["cushion"])
]

print("="*70)
print("ZERO-SHOT EVALUATION OF NOVEL (UNTRAINED) OBJECTS")
print("="*70)

for task, expected in test_cases:
    task_vec = detector.get_task_embedding(task, decompose=True)
    scores = []
    for obj in novel_objects:
        obj_vec = detector.get_object_embedding(obj)
        sim = float(np.dot(task_vec, obj_vec))
        scores.append((obj, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    
    best_obj, best_sim = scores[0]
    print(f"\nTask: '{task}'")
    print(f" -> Top Selected Novel Object: '{best_obj}' (Similarity: {best_sim:.4f})")
    print(" -> All Novel Candidate Rankings:")
    for obj, sim in scores[:4]:
        print(f"    * {obj:<15}: {sim:.4f}")
