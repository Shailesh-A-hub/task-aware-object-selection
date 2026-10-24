"""
Demonstration Script for OpenRouter Backend Mechanics.
Shows the complete end-to-end execution flow of how OpenRouter dynamic reasoning works:
1. Cache Lookup -> 2. Prompting -> 3. API Dispatch -> 4. Affordance Distillation ->
5. Sentence-BERT Vectorization -> 6. Cosine Sim -> 7. Score Fusion & Decision.
"""

import os
import json
import time
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Setup API key and Models
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
CACHE_FILE = "knowledge_bank/openrouter_cache.json"

print("=" * 80)
print("     OPENROUTER BACKEND REASONING ENGINE: STEP-BY-STEP WORKFLOW")
print("=" * 80)

def step_1_check_cache(term: str, term_type: str = "task"):
    cache_key = f"{term_type}:{term.lower().strip()}"
    print(f"\n[STEP 1: LOCAL DISK CACHE LOOKUP]")
    print(f"Checking '{CACHE_FILE}' for key: '{cache_key}'...")
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if cache_key in cache:
                print(f"-> CACHE HIT! Found cached distillation: \"{cache[cache_key]}\"")
                print(f"-> Latency: 0.0 ms (Zero API calls needed!)")
                return cache[cache_key], True
        except Exception as e:
            pass
    print("-> CACHE MISS! Term requires live OpenRouter LLM reasoning.")
    return None, False

def step_2_call_openrouter(query: str, term_type: str = "task"):
    print(f"\n[STEP 2: CONSTRUCTING LLM PROMPT & API PAYLOAD]")
    if term_type == "task":
        prompt = (
            f"You are an expert AI physical affordance reasoning system.\n"
            f"User task: '{query}'.\n"
            f"Identify 2 to 3 common everyday physical objects, tools, or household appliances that humans use to solve this task, "
            f"and describe their key physical properties (e.g. shape, thermal resistance, sharp edge, hollow container, sturdy elevated support).\n"
            f"Answer in 1 concise sentence in this format: 'Objects: [suitable objects]; Affordances: [key physical characteristics]'."
        )
    else:
        prompt = (
            f"Provide a concise 1-sentence physical affordance and functional description for the object '{query}'. "
            f"Focus strictly on its physical shape, material, stability, portability, and primary affordances. "
            f"Do not include conversational filler."
        )
    
    print(f"Endpoint: https://openrouter.ai/api/v1/chat/completions")
    print(f"System Role: 'You are an expert AI physical affordance and robotics object reasoning assistant.'")
    print(f"User Prompt:\n\"\"\"\n{prompt}\n\"\"\"")
    
    headers = {
        "Authorization": f"Bearer {API_KEY.strip()}",
        "HTTP-Referer": "https://github.com/shailesh-vit/task-aware-object-selection",
        "X-Title": "Task-Aware Object Selection",
        "Content-Type": "application/json"
    }
    
    candidate_models = [
        "nvidia/nemotron-3.5-lightning:free",
        "openrouter/auto",
        "google/gemma-4-26b-a4b-it:free"
    ]
    
    print(f"\n[STEP 3: MULTI-MODEL FAILOVER DISPATCH]")
    print(f"Fallback Chain: {' -> '.join(candidate_models)}")
    
    t0 = time.time()
    for model in candidate_models:
        print(f"-> Attempting model: {model}...")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert AI physical affordance and robotics object reasoning assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 150
        }
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=12)
            elapsed_ms = (time.time() - t0) * 1000.0
            print(f"   Response Status: HTTP {r.status_code} ({elapsed_ms:.1f} ms)")
            if r.status_code == 200:
                data = r.json()
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    raw_content = msg.get("content") or msg.get("reasoning") or ""
                    if isinstance(raw_content, str) and raw_content.strip():
                        distilled_text = raw_content.strip()
                        print(f"\n[STEP 4: RAW LLM DISTILLATION OUTPUT]")
                        print(f"Active Model Used: {model}")
                        print(f"Distilled Output:\n\"{distilled_text}\"")
                        return distilled_text, elapsed_ms
            else:
                print(f"   Model {model} busy or rate-limited ({r.status_code}). Trying next fallback...")
        except Exception as e:
            print(f"   Connection failed: {e}. Trying next...")
            
    return f"Objects suitable for {query}", 0.0

def step_5_vectorize_sentence_bert(distilled_text: str):
    print(f"\n[STEP 5: SENTENCE-BERT DENSE VECTOR ENCODING (CPU)]")
    print(f"Loading 'all-MiniLM-L6-v2' (384-dimensional dense vector space)...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    t0 = time.time()
    vector = sbert.encode([distilled_text], convert_to_numpy=True, normalize_embeddings=True)[0]
    elapsed = (time.time() - t0) * 1000.0
    print(f"Encoded Vector Dimensions: {vector.shape} (384 floats)")
    print(f"First 5 Vector Components: {np.round(vector[:5], 4)}")
    print(f"L2 Unit Norm Check: ||v|| = {np.linalg.norm(vector):.4f} (Ready for Cosine Dot-Product)")
    print(f"CPU Encoding Time: {elapsed:.2f} ms")
    return vector

def step_6_score_candidates(task_vec: np.ndarray, detected_objects: list):
    print(f"\n[STEP 6: COSINE SIMILARITY & WEIGHTED SCORE FUSION]")
    print(f"Formula: Score = 0.70 * Sim(object, task) + 0.30 * Conf(object)")
    print(f"Dual Gates: Sim >= 0.12 (Semantic Gate) AND Score >= 0.35 (Relevance Gate)")
    print("-" * 75)
    
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    scored = []
    for obj in detected_objects:
        name = obj["name"]
        conf = obj["confidence"]
        desc = obj["description"]
        
        # Object vector
        o_vec = sbert.encode([desc], convert_to_numpy=True, normalize_embeddings=True)[0]
        sim = float(np.dot(task_vec, o_vec))
        score = (0.70 * sim) + (0.30 * conf)
        
        is_semantic = sim >= 0.12
        is_relevant = is_semantic and (score >= 0.35)
        
        scored.append({
            "name": name,
            "conf": conf,
            "sim": sim,
            "score": score,
            "valid": is_relevant
        })
        
        gate_status = "PASS (Eligible)" if is_relevant else "FAIL (Rejected)"
        print(f"Candidate: {name:15s} | Conf: {conf:.2f} | Sim: {sim:.4f} | Fused Score: {score:.4f} -> {gate_status}")
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    valid_candidates = [c for c in scored if c["valid"]]
    
    print("-" * 75)
    if valid_candidates:
        top1 = valid_candidates[0]
        print(f"[FINAL DECISION: SUCCESS]")
        print(f"Selected Object: {top1['name'].upper()} with Top Score = {top1['score']:.4f}")
        print(f"Assigned Visual Action: Green Bounding Box Overlay")
    else:
        print(f"[FINAL DECISION: SAFE NEGATIVE ABORT]")
        print(f"Output: NO_SUITABLE_OBJECT_FOUND")
        print(f"Assigned Visual Action: Red Alert Banner Overlay")

if __name__ == "__main__":
    test_query = "bake"
    cached_text, was_cached = step_1_check_cache(test_query, "task")
    if not was_cached:
        distilled_text, lat = step_2_call_openrouter(test_query, "task")
    else:
        distilled_text = cached_text
        
    task_vector = step_5_vectorize_sentence_bert(distilled_text)
    
    # Mock detected objects in kitchen scene
    mock_scene = [
        {"name": "microwave", "confidence": 0.82, "description": "countertop kitchen electrical box appliance with door; used for rapid dielectric heating and reheating of food items"},
        {"name": "knife", "confidence": 0.78, "description": "sharp handheld steel blade utensil used for slicing cutting and preparing food"},
        {"name": "apple", "confidence": 0.65, "description": "round edible sweet fruit with firm flesh eaten raw as snack"},
        {"name": "potted plant", "confidence": 0.91, "description": "decorative live flora in container with soil; living botanical plant that photosynthesizes"}
    ]
    step_6_score_candidates(task_vector, mock_scene)
