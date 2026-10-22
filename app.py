"""
Streamlit Web Application: Task-Aware Object Selection System
Supports:
- Tab 1: Image Scene Selection (YOLOv8n + Offline Knowledge Bank + Weighted Fusion)
- Tab 2: Novel (Untrained) Objects Playground (Tests objects outside COCO-80 via OpenRouter LLM distillation)
- Live OpenRouter connection for novel terms and open-domain task queries
- Interactive Alpha (0.7) and Threshold (0.25) controls
- Latency and hardware footprint metrics (for DVCon / College Review)
- Candidate ranking table and OpenCV annotated visualizer
"""

import os
import time
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from task_aware_detector import TaskAwareDetector
from openrouter_engine import OpenRouterEngine

# Page configuration
st.set_page_config(
    page_title="Task-Aware Object Selector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Sleek Dark / Tech Theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2638 0%, #151b27 100%);
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #38bdf8;
        margin-top: 5px;
    }
    .success-banner {
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 12px 18px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 15px;
        border-left: 5px solid #10b981;
    }
    .reject-banner {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 12px 18px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 15px;
        border-left: 5px solid #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.title("🎯 Control Panel")
st.sidebar.markdown("---")

# OpenRouter Settings
st.sidebar.subheader("🌐 OpenRouter Integration")
openrouter_key = st.sidebar.text_input(
    "OpenRouter API Key",
    value=os.environ.get("OPENROUTER_API_KEY", ""),
    type="password",
    help="Enter your OpenRouter key to distill novel objects and decompose open-domain tasks on the fly."
)

model_choice = st.sidebar.selectbox(
    "LLM Model (OpenRouter)",
    [
        "openrouter/auto",
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3.5-lightning:free",
        "minimax/minimax-m3.1-flash:free",
        "liquid/lfm-2.5-2.6b:free",
        "openai/gpt-4o-mini"
    ],
    index=0
)

if st.sidebar.button("⚡ Test OpenRouter API Key"):
    if not openrouter_key.strip():
        st.sidebar.warning("Please enter your OpenRouter API Key first!")
    else:
        with st.sidebar.status("Connecting to OpenRouter..."):
            test_engine = OpenRouterEngine(api_key=openrouter_key.strip(), model=model_choice)
            test_res = test_engine.test_connection()
            if test_res["success"]:
                st.sidebar.success(f"Connected! Model: {test_res['model']} ({test_res['latency_ms']} ms)")
                st.sidebar.caption(f"LLM Response: \"{test_res['message']}\"")
            else:
                st.sidebar.error(f"Connection Failed: {test_res.get('error')}")

enable_decomposition = st.sidebar.checkbox(
    "Enable Task Affordance Decomposition",
    value=True,
    help="Uses OpenRouter to translate abstract user goals into explicit physical object affordances."
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Scoring Parameters")
alpha = st.sidebar.slider("Alpha (Semantic Weight)", min_value=0.0, max_value=1.0, value=0.70, step=0.05,
                          help="Score = Alpha * SemanticSim + (1 - Alpha) * DetectionConf")
threshold = st.sidebar.slider("Relevance Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05,
                              help="Minimum score below which objects are rejected as unsuitable.")

use_int8 = st.sidebar.checkbox("Use INT8 Knowledge Bank (35.5 KB)", value=False,
                               help="Enable ultra-compact quantized INT8 knowledge bank for edge CPU deployment.")

st.sidebar.markdown("---")
st.sidebar.info("""
**System Architecture Specs:**
- Base: Sawatzky et al. (CVPR 2019)
- Detector: YOLOv8-nano (CPU)
- Embeddings: 384-d Sentence-BERT
- Offline Knowledge Bank: 35.5 KB (INT8)
- End-to-End Latency: ~140-150 ms (Standard Edge CPU)
""")

# ----------------- INITIALIZE DETECTOR (CACHED) -----------------
@st.cache_resource(show_spinner="Loading models & 80-Class Knowledge Bank...")
def get_detector(int8_mode: bool, api_key: str, model: str):
    return TaskAwareDetector(
        yolo_model_path="yolov8n.pt",
        use_quantized_int8=int8_mode,
        openrouter_api_key=api_key if api_key else None,
        openrouter_model=model,
        alpha=0.7,
        threshold=0.25
    )

detector = get_detector(use_int8, openrouter_key, model_choice)
# Update dynamic parameters
detector.alpha = alpha
detector.threshold = threshold
detector.openrouter.api_key = openrouter_key
detector.openrouter.model = model_choice

# ----------------- MAIN UI -----------------
st.title("🤖 Task-Aware Object Selection System")
st.markdown("#### LLM-Distilled Semantic Knowledge on Edge Platforms | Zero-Shot & Training-Free")

tab_scene, tab_novel = st.tabs(["🖼️ Image Scene Selection", "🧪 Novel (Untrained) Objects Playground"])

# ================= TAB 1: IMAGE SCENE SELECTION =================
with tab_scene:
    col_img_input, col_query_input = st.columns([1, 1])

    with col_img_input:
        st.subheader("1. Input Scene")
        image_source = st.radio("Image Source", ["Use Sample Image", "Upload New Image"], horizontal=True)

        if image_source == "Use Sample Image":
            sample_options = {
                "🛋️ Clean Living Room (Raw Photo)": "sample_living_room_raw.jpg",
                "🍳 Clean Kitchen & Dining (Raw Photo)": "sample_kitchen_raw.jpg",
                "🏠 Decorate Sample 1": "decorate.jpg",
                "🏠 Decorate Sample 2": "decorate1.jpg"
            }
            sample_choice = st.selectbox("Select Sample:", list(sample_options.keys()))
            input_image_path = sample_options[sample_choice]
            display_pil = Image.open(input_image_path)
        else:
            uploaded_file = st.file_uploader(
                "Upload any normal, raw photo (JPG, PNG) with NO bounding boxes:",
                type=["jpg", "jpeg", "png"],
                help="Upload an ordinary camera or phone picture. The system automatically detects objects and draws boxes for you!"
            )
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                tfile.write(uploaded_file.read())
                input_image_path = tfile.name
                display_pil = Image.open(input_image_path)
            else:
                input_image_path = "sample_living_room_raw.jpg"
                display_pil = Image.open(input_image_path)

    with col_query_input:
        st.subheader("2. Task Query")
        
        query_mode = st.radio(
            "Query Mode:",
            ["✍️ Type Manual / Novel Query", "📋 Choose Official COCO-Tasks (14 Benchmarks)"],
            horizontal=True
        )

        official_14_tasks = [
            "sit comfortably",
            "step on something",
            "place flowers",
            "water plant",
            "open parcel",
            "open bottle of beer",
            "serve wine",
            "pour sugar",
            "smear butter",
            "extinguish fire",
            "pound carpet",
            "dig hole",
            "get lemon out of tea",
            "get potatoes out of fire"
        ]

        if query_mode == "✍️ Type Manual / Novel Query":
            task_query = st.text_input(
                "Enter Natural Language Task Query (Manual):",
                value="sit comfortable",
                help="Type ANY task, e.g. 'relax with a book', 'reheat leftover food', 'slice fruit', 'reach high ceiling fan'."
            )
            st.caption("💡 *Quick try:* `sit comfortable` | `place flowers` | `cut something` (negative test) | `reach high ceiling fan` (novel task)")
        else:
            chosen_coco_task = st.selectbox("Select Benchmark Task:", official_14_tasks)
            task_query = st.text_input("Confirm / Edit Task Query:", value=chosen_coco_task)

        with st.expander("ℹ️ Knowledge Bank Explainer: 80 Objects vs 14 Tasks"):
            st.markdown("""
            **What does the Offline Knowledge Bank contain?**
            - **80 Object Classes (Precomputed)**: All 80 COCO categories (couch, chair, table, bottle, knife, vase, etc.) have rich LLM affordance descriptions precomputed into 384-d vectors, stored offline in **`coco80_knowledge_bank_int8.npy` (35.5 KB)**.
            - **14 Official Tasks**: The 14 benchmark tasks defined in Sawatzky et al. (CVPR 2019).
            - **Manual / Novel Queries**: You can enter **any task query** above. Sentence-BERT encodes your query in 15 ms and compares it against all detected objects! If connected to OpenRouter, novel objects outside COCO and complex tasks are dynamically reasoned.
            """)

        run_btn = st.button("🚀 Run Task-Aware Selection", type="primary", use_container_width=True)

    st.markdown("---")

    # ----------------- EXECUTION & RESULTS -----------------
    if run_btn:
        with st.spinner("Processing scene and calculating semantic relevance..."):
            # Execute selection pipeline
            result = detector.select_best_object(
                input_image_path,
                task_query,
                alpha=alpha,
                threshold=threshold,
                decompose_task=enable_decomposition
            )

            annotated_img = detector.visualize(input_image_path, result)
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # Status Banner
        if result["status"] == "SUCCESS":
            sel = result["selected_object"]
            st.markdown(
                f"""<div class='success-banner'>
                ✅ <b>SUCCESS</b>: Selected <b>{sel['name'].upper()}</b> for task <i>'{task_query}'</i> | 
                Fused Score: <b>{sel['fused_score']:.4f}</b> (Semantic Sim: {sel['semantic_similarity']:.4f}, Confidence: {sel['confidence']:.4f})
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div class='reject-banner'>
                🚫 <b>REJECTED</b>: No suitable object found for task <i>'{task_query}'</i>. 
                All detected objects failed the minimum relevance threshold ({threshold:.2f}).
                </div>""",
                unsafe_allow_html=True
            )

        # Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">YOLO Detection</div>
                <div class="metric-value">{result['timing_ms']['detection']:.1f} ms</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Array Lookup & Matching</div>
                <div class="metric-value">{result['timing_ms']['semantic_matching']:.1f} ms</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Latency (CPU)</div>
                <div class="metric-value">{result['timing_ms']['total']:.1f} ms</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            kb_size = 35.5 if use_int8 else 60.1
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Knowledge Bank Size</div>
                <div class="metric-value">{kb_size:.1f} KB</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Layout for Image and Table
        res_col_left, res_col_right = st.columns([1.2, 1])

        with res_col_left:
            st.subheader("Annotated Scene Detection")
            st.image(annotated_rgb, use_container_width=True)

        with res_col_right:
            st.subheader("Candidate Objects Evaluation")
            if result["all_candidates"]:
                df = pd.DataFrame([
                    {
                        "Object": c["name"],
                        "Fused Score": f"{c['fused_score']:.4f}",
                        "Semantic Sim (70%)": f"{c['semantic_similarity']:.4f}",
                        "YOLO Conf (30%)": f"{c['confidence']:.4f}",
                        "Status": "✅ Selected" if result["selected_object"] and c["name"] == result["selected_object"]["name"] and c["fused_score"] == result["selected_object"]["fused_score"] else ("⚠️ Valid" if c.get("is_relevant") else "❌ Rejected")
                    }
                    for c in result["all_candidates"]
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No candidates detected by YOLO.")

            # OpenRouter Live Explanation Card
            st.subheader("🧠 OpenRouter Reasoning Info")
            cache_key = f"task:{task_query.lower().strip()}"
            cache_hit = detector.openrouter.cache.get(cache_key, None)

            if detector.openrouter.last_call_success and detector.openrouter.last_response:
                st.success(f"**🟢 Live OpenRouter LLM ({detector.openrouter.last_model_used}):**\n\n{detector.openrouter.last_response}")
            elif cache_hit and "affordance" in cache_hit.lower():
                st.info(f"**Cached LLM Affordances:**\n\n{cache_hit}")
            elif openrouter_key:
                if detector.openrouter.last_error:
                    st.error(f"**⚠️ OpenRouter Notice:** {detector.openrouter.last_error}\n\nFalling back to direct Sentence-BERT semantic matching.")
                else:
                    st.caption(f"Ready with model: `{detector.openrouter.last_model_used}`")
            else:
                st.caption("💡 *Note: OpenRouter API key not set. Using local offline knowledge bank & direct Sentence-BERT.*")

    else:
        # Initial preview before click
        p1, p2 = st.columns([1.2, 1])
        with p1:
            st.image(display_pil, caption="Selected Input Image", use_container_width=True)
        with p2:
            st.info("👈 Select or type a task query and click **'Run Task-Aware Selection'** to test the pipeline.")
            st.markdown("""
            **Try these benchmark queries:**
            - `sit comfortable` -> Evaluates couches, chairs, beds.
            - `place flowers` -> Evaluates vases, cups, vessels.
            - `water plant` -> Evaluates watering cans, bottles, cups.
            - `cut something` -> Negative test: should cleanly reject with red banner!
            - `reach high ceiling fan` -> Novel task: tests OpenRouter affordance expansion!
            """)

# ================= TAB 2: NOVEL (UNTRAINED) OBJECTS PLAYGROUND =================
with tab_novel:
    st.subheader("🧪 Zero-Shot Reasoning on Novel (Untrained) Objects")
    st.markdown("""
    This playground demonstrates **how the system handles objects that are completely absent from COCO training**.
    While conventional models can only recognize their fixed 80 categories, our **OpenRouter affordance distillation**
    dynamically reasons about the physical properties of **any arbitrary object** on the fly!
    """)

    col_nov1, col_nov2 = st.columns([1.2, 1])

    with col_nov1:
        st.markdown("**1. Enter Untrained Objects (None of these exist in COCO-80):**")
        default_novel_objs = "ladder, cushion, frying pan, wooden stool, hammer, flashlight, sponge, teapot, screwdriver"
        novel_input_str = st.text_area(
            "Comma-separated list of novel objects:",
            value=default_novel_objs,
            height=90,
            help="Type any novel object names separated by commas."
        )

    with col_nov2:
        st.markdown("**2. Target Task Query:**")
        preset_novel_tasks = [
            "reach top shelf",
            "fry an egg",
            "make hard chair softer to sit on",
            "wipe spilled milk",
            "tighten loose screw",
            "boil water for tea",
            "illuminate dark basement"
        ]
        picked_task = st.selectbox("Quick task presets:", preset_novel_tasks)
        novel_task = st.text_input("Or enter any custom task query:", value=picked_task)

        eval_novel_btn = st.button("🚀 Evaluate Novel Objects via OpenRouter", type="primary", use_container_width=True)

    st.markdown("---")

    if eval_novel_btn:
        with st.spinner("Distilling physical affordances for novel objects..."):
            novel_obj_list = [o.strip() for o in novel_input_str.split(",") if o.strip()]
            task_vec = detector.get_task_embedding(novel_task, decompose=enable_decomposition)

            novel_results = []
            for nobj in novel_obj_list:
                # Get or distill embedding
                n_vec = detector.get_object_embedding(nobj)
                sim = float(np.dot(task_vec, n_vec))
                affordance_desc = detector.openrouter.cache.get(f"obj:{nobj.lower()}", "Generic physical object")

                novel_results.append({
                    "Novel Object": nobj,
                    "Semantic Similarity": round(sim, 4),
                    "Affordance Description": affordance_desc
                })

            novel_results.sort(key=lambda x: x["Semantic Similarity"], reverse=True)

        top_choice = novel_results[0]
        st.markdown(
            f"""<div class='success-banner'>
            🏆 <b>TOP SELECTED NOVEL OBJECT</b>: <b>{top_choice['Novel Object'].upper()}</b> for task <i>'{novel_task}'</i> | 
            Semantic Match: <b>{top_choice['Semantic Similarity']:.4f}</b>
            </div>""",
            unsafe_allow_html=True
        )

        # Display Comparison Table and Chart
        col_tbl, col_chart = st.columns([1.3, 1])

        with col_tbl:
            st.subheader("Leaderboard of Novel Candidates")
            df_novel = pd.DataFrame([
                {
                    "Rank": i + 1,
                    "Object": r["Novel Object"],
                    "Similarity": f"{r['Semantic Similarity']:.4f}",
                    "Status": "🏆 Best Choice" if i == 0 else ("🥈 Viable Alternative" if r["Semantic Similarity"] >= 0.35 else "❌ Unsuitable"),
                    "Affordances": r["Affordance Description"][:75] + "..." if len(r["Affordance Description"]) > 75 else r["Affordance Description"]
                }
                for i, r in enumerate(novel_results)
            ])
            st.dataframe(df_novel, use_container_width=True, hide_index=True)

        with col_chart:
            st.subheader("Affordance Similarity Match")
            chart_data = pd.DataFrame({
                "Object": [r["Novel Object"] for r in novel_results],
                "Similarity": [r["Semantic Similarity"] for r in novel_results]
            }).set_index("Object")
            st.bar_chart(chart_data)

        # Show full affordance explanation
        with st.expander("📖 View Full OpenRouter Distilled Descriptions for All Novel Objects"):
            for r in novel_results:
                st.markdown(f"- **{r['Novel Object']}**: {r['Affordance Description']}")
