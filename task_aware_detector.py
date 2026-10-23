"""
Task-Aware Object Selection Pipeline.

Features:
- YOLOv8-nano lightweight object detector.
- Direct array lookup from precomputed 80-class offline knowledge bank (zero runtime text encoding for known classes).
- Exact weighted fusion: score = alpha * semantic_sim + (1 - alpha) * detection_conf (alpha=0.7).
- Relevance thresholding (0.25) to identify when no suitable object exists in the scene.
- OpenRouter integration for open-vocabulary novel objects and novel task decomposition.
- Profiling / latency breakdown (CPU timing).
- Comprehensive OpenCV visualizer with bounding boxes and score breakdowns.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from openrouter_engine import OpenRouterEngine


class TaskAwareDetector:
    def __init__(
        self,
        yolo_model_path: str = "yolov8n.pt",
        sentence_model_name: str = "all-MiniLM-L6-v2",
        kb_dir: str = "knowledge_bank",
        knowledge_bank_dir: Optional[str] = None,
        use_quantized_int8: bool = False,
        use_int8: Optional[bool] = None,
        openrouter_api_key: Optional[str] = None,
        openrouter_model: str = "openrouter/auto",
        alpha: float = 0.7,
        threshold: float = 0.25,
        **kwargs
    ):
        self.alpha = alpha
        self.threshold = threshold
        self.kb_dir = knowledge_bank_dir if knowledge_bank_dir is not None else kb_dir
        self.use_quantized_int8 = use_int8 if use_int8 is not None else use_quantized_int8

        print("[TaskAwareDetector] Initializing models and offline knowledge bank...")
        # 1. Load YOLOv8-nano
        self.yolo = YOLO(yolo_model_path)

        # 2. Load SentenceTransformer (for task queries and novel terms)
        self.text_model = SentenceTransformer(sentence_model_name)

        # 3. Load Offline Knowledge Bank
        self._load_knowledge_bank()

        # 4. OpenRouter Engine for novel terms and task decomposition
        self.openrouter = OpenRouterEngine(
            api_key=openrouter_api_key,
            model=openrouter_model,
            cache_file=os.path.join(kb_dir, "openrouter_cache.json")
        )

        # Dynamic cache for novel objects embedded at runtime
        self.dynamic_embeddings: Dict[str, np.ndarray] = {}

    def _load_knowledge_bank(self):
        """Loads precomputed 80-class knowledge bank and 14 COCO-Tasks metadata."""
        classes_meta_path = os.path.join(self.kb_dir, "coco80_classes.json")
        tasks_meta_path = os.path.join(self.kb_dir, "coco14_tasks.json")

        if not os.path.exists(classes_meta_path):
            raise FileNotFoundError(
                f"Knowledge bank metadata not found at {classes_meta_path}. "
                f"Please run generate_knowledge_bank.py first."
            )

        with open(classes_meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.class_names = data["classes"]
            self.class_to_idx = data["mapping"]
            self.class_descriptions = data["descriptions"]

        if os.path.exists(tasks_meta_path):
            with open(tasks_meta_path, "r", encoding="utf-8") as f:
                self.task_meta = json.load(f)
        else:
            self.task_meta = {"tasks": [], "descriptions": {}}

        # Load embedding matrices
        if self.use_quantized_int8:
            emb_file = os.path.join(self.kb_dir, "coco80_knowledge_bank_int8.npy")
            print(f"[TaskAwareDetector] Using INT8 quantized knowledge bank ({os.path.getsize(emb_file) / 1024:.2f} KB)")
            raw_int8 = np.load(emb_file)
            # Dequantize to float32 for cosine dot-product
            self.obj_embeddings = raw_int8.astype(np.float32) / 127.0
        else:
            emb_file = os.path.join(self.kb_dir, "coco80_knowledge_bank_fp16.npy")
            print(f"[TaskAwareDetector] Using Float16 knowledge bank ({os.path.getsize(emb_file) / 1024:.2f} KB)")
            raw_fp16 = np.load(emb_file)
            self.obj_embeddings = raw_fp16.astype(np.float32)

        # Normalize rows to unit vectors for pure dot-product cosine similarity
        norms = np.linalg.norm(self.obj_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.obj_embeddings = self.obj_embeddings / norms

    def get_object_embedding(self, class_name: str) -> np.ndarray:
        """
        Retrieves embedding vector for an object class:
        1. Fast direct array lookup from 80-class offline bank (0 ms).
        2. In-memory dynamic cache for previously queried novel classes.
        3. OpenRouter affordance distillation fallback for novel / open-vocabulary terms.
        """
        name_clean = class_name.lower().strip()

        # Tier 1: Direct 80-class precomputed lookup
        if name_clean in self.class_to_idx:
            idx = self.class_to_idx[name_clean]
            return self.obj_embeddings[idx]

        # Tier 2: Check dynamic cache
        if name_clean in self.dynamic_embeddings:
            return self.dynamic_embeddings[name_clean]

        # Tier 3: Novel term -> OpenRouter distillation
        print(f"[TaskAwareDetector] Novel term '{name_clean}' not in COCO-80 bank. Distilling via OpenRouter...")
        desc = self.openrouter.distill_novel_object(name_clean)
        vec = self.text_model.encode([desc], convert_to_numpy=True, normalize_embeddings=True)[0]
        self.dynamic_embeddings[name_clean] = vec
        return vec

    def get_task_embedding(self, task_query: str, decompose: bool = False) -> np.ndarray:
        """
        Encodes the task query string:
        - If decompose is True: uses OpenRouter (or smart offline affordance engine) to expand the task into physical affordance requirements.
        - Otherwise encodes task query directly with Sentence-BERT.
        """
        query_text = task_query.strip()
        if decompose:
            decomposed = self.openrouter.decompose_novel_task(query_text)
            if decomposed and decomposed != query_text:
                print(f"[TaskAwareDetector] Task affordance decomposition:\n  '{query_text}' -> '{decomposed}'")
                query_text = decomposed

        vec = self.text_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return vec

    def detect_objects(self, image: Union[str, np.ndarray], conf_threshold: float = 0.25) -> Tuple[List[Dict[str, Any]], float]:
        """Runs YOLOv8-nano and returns detected candidate bounding boxes."""
        t0 = time.time()
        results = self.yolo(image, conf=conf_threshold, verbose=False)
        det_time_ms = (time.time() - t0) * 1000.0

        detected_objects = []
        for result in results:
            names = result.names
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                object_name = names.get(class_id, f"obj_{class_id}")
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                detected_objects.append({
                    "name": object_name,
                    "class_id": class_id,
                    "confidence": confidence,
                    "box": xyxy
                })
        return detected_objects, det_time_ms

    def select_best_object(
        self,
        image: Union[str, np.ndarray],
        task_query: str,
        alpha: Optional[float] = None,
        threshold: Optional[float] = None,
        decompose_task: bool = False
    ) -> Dict[str, Any]:
        """
        Executes complete 4-stage pipeline:
        Stage 1: YOLO Object Detection
        Stage 2: Direct Array Lookup & Semantic Matching
        Stage 3: Weighted Fusion: Score = alpha * Sim + (1 - alpha) * Conf
        Stage 4: Threshold Filtering and Selection
        """
        if alpha is None:
            alpha = self.alpha
        if threshold is None:
            threshold = self.threshold

        t_total_start = time.time()

        # Stage 1: Detect objects
        detected, t_det_ms = self.detect_objects(image, conf_threshold=threshold)

        if not detected:
            return {
                "task": task_query,
                "status": "NO_OBJECTS_DETECTED",
                "message": "YOLO detector did not find any objects above detection threshold.",
                "selected_object": None,
                "all_candidates": [],
                "timing_ms": {"detection": t_det_ms, "semantic_matching": 0.0, "total": (time.time() - t_total_start) * 1000.0}
            }

        # Stage 2: Encode query and retrieve object embeddings
        t_sem_start = time.time()
        q_vec = self.get_task_embedding(task_query, decompose=decompose_task)

        scored_candidates = []
        for obj in detected:
            obj_name = obj["name"]
            conf = obj["confidence"]
            obj_vec = self.get_object_embedding(obj_name)

            # Cosine similarity (vectors are unit normalized)
            semantic_sim = float(np.dot(q_vec, obj_vec))

            # Stage 3: Exact Weighted Fusion
            # score(oi) = alpha * sim(oi, t) + (1 - alpha) * conf(oi)
            fused_score = (alpha * semantic_sim) + ((1.0 - alpha) * conf)

            # Determine relevance: must satisfy minimum semantic similarity (0.12)
            # and final fused score threshold (0.25) to avoid false selections of irrelevant objects with high YOLO confidence
            is_semantically_relevant = semantic_sim >= 0.12
            is_valid = is_semantically_relevant and (fused_score >= threshold)

            scored_candidates.append({
                "name": obj_name,
                "class_id": obj.get("class_id", -1),
                "box": obj["box"],
                "confidence": round(conf, 4),
                "semantic_similarity": round(semantic_sim, 4),
                "fused_score": round(fused_score, 4),
                "is_relevant": is_valid
            })

        # Sort descending by fused score
        scored_candidates.sort(key=lambda x: x["fused_score"], reverse=True)
        t_sem_ms = (time.time() - t_sem_start) * 1000.0
        t_total_ms = (time.time() - t_total_start) * 1000.0

        # Stage 4: Threshold filtering and top-1 selection
        valid_candidates = [c for c in scored_candidates if c["is_relevant"]]

        selected = valid_candidates[0] if valid_candidates else None
        status = "SUCCESS" if selected else "NO_SUITABLE_OBJECT_FOUND"

        return {
            "task": task_query,
            "status": status,
            "selected_object": selected,
            "all_candidates": scored_candidates,
            "threshold": threshold,
            "alpha": alpha,
            "timing_ms": {
                "detection": round(t_det_ms, 2),
                "semantic_matching": round(t_sem_ms, 2),
                "total": round(t_total_ms, 2)
            }
        }

    def visualize(
        self,
        image_path_or_array: Union[str, np.ndarray],
        result: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """Visualizes selection results, bounding boxes, scores, and status banner."""
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise ValueError(f"Could not read image from {image_path_or_array}")
        else:
            image = image_path_or_array.copy()

        h, w = image.shape[:2]
        task_text = result["task"]
        status = result["status"]
        selected = result["selected_object"]

        # Draw all candidates in subtle light gray / orange boxes
        for cand in result["all_candidates"]:
            box = cand["box"].astype(int)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (60, 60, 60), 1)

        # Draw selected object highlighted in vibrant green
        if selected is not None:
            box = selected["box"].astype(int)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)

            label = f"SELECTED: {selected['name']} | Score: {selected['fused_score']:.2f}"
            sub_label = f"(Sim: {selected['semantic_similarity']:.2f}, Conf: {selected['confidence']:.2f})"

            # Label banner
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(image, (box[0], max(0, box[1] - 45)), (box[0] + lw + 10, box[1]), (0, 200, 0), -1)
            cv2.putText(image, label, (box[0] + 5, max(18, box[1] - 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            cv2.putText(image, sub_label, (box[0] + 5, max(38, box[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)

        # Top status banner
        banner_color = (0, 140, 0) if status == "SUCCESS" else (0, 0, 180)
        cv2.rectangle(image, (0, 0), (w, 42), banner_color, -1)
        status_msg = f"Task: '{task_text}' | Status: {status} | Latency: {result['timing_ms']['total']:.1f}ms"
        cv2.putText(image, status_msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if output_path:
            cv2.imwrite(output_path, image)
            print(f"[TaskAwareDetector] Visualization saved to: {output_path}")

        return image
