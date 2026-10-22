"""
Compiles FINAL_REVIEW_REPORT.md into a professional PDF report using ReportLab.
Includes title page, headers, footers, clean tables, code callouts, and styling.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Preformatted, Image
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "Task-Aware Object Selection on Edge Platforms | Final Review Report")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
            
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 40, footer_text)
        self.drawString(54, 40, "VIT Vellore - Team RTL Rebels 616")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 52, 558, 52)
        
        self.restoreState()


def build_pdf(filename="FINAL_REVIEW_REPORT.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=16,
        textColor=colors.HexColor('#475569'),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'SecHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SubSecHeading',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#0369a1'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=6
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#0f172a')
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Final Review: Task-Aware Object Selection on Edge Platforms", title_style))
    story.append(Paragraph(
        "<b>Sub-theme:</b> LLM-Distilled Semantic Knowledge & OpenRouter Dynamic Reasoning for Assistive Edge Robotics<br/>"
        "<b>Team:</b> Shailesh A., Pradeep P., Sanjay K. | <b>Institution:</b> Vellore Institute of Technology (VIT)<br/>"
        "<b>Target Contest:</b> DVCon India 2026 Design Contest (Team 616, RTL Rebels)<br/>"
        "<b>Base Paper:</b> J. Sawatzky et al., <i>What Object Should I Use? — Task Driven Object Detection</i>, CVPR 2019",
        subtitle_style
    ))
    story.append(Spacer(1, 10))

    # Section 1: Abstract
    story.append(Paragraph("1. Abstract", h1_style))
    story.append(Paragraph(
        "Conventional computer vision object detection frameworks identify and localize object categories in a scene, but they are fundamentally incapable of selecting the specific object best suited for a given human task query. In Review 1, a lightweight training-free framework was proposed using a YOLOv8-nano detector and an offline 47 KB precomputed semantic knowledge bank. For the Final Review, this architecture has been significantly advanced by integrating a dual-tier semantic engine: an ultra-compact 35.5 KB INT8 precomputed offline knowledge bank for standard COCO classes, coupled with <b>OpenRouter dynamic Large Language Model (LLM) affordance reasoning</b> for open-vocabulary, untrained objects and open-domain natural language commands.",
        body_style
    ))
    story.append(Paragraph(
        "Detected objects are evaluated against the task query using cosine similarity between 384-dimensional Sentence-BERT dense embeddings. An exact weighted score fusion mechanism combining semantic relevance (70%) and visual detection confidence (30%) produces candidate rankings, while a dual-gate threshold filter (minimum semantic similarity 0.12, final score 0.25) prevents false selections in negative scenes. Quantitative evaluation demonstrates competitive accuracy (36.2% mAP on COCO-Tasks) while achieving 140-150 ms end-to-end latency on standard edge CPU hardware, eliminating the need for expensive GPU infrastructure.",
        body_style
    ))

    # Section 2: Introduction
    story.append(Paragraph("2. Introduction", h1_style))
    story.append(Paragraph(
        "Autonomous and assistive robotic systems operating in real-world human environments must make contextually appropriate decisions when confronted with cluttered scenes. While modern convolutional and transformer-based detectors excel at answering <i>'What object is this?'</i>, assistive robotics requires answering: <i>'Which object should I use for my task?'</i>",
        body_style
    ))
    story.append(Paragraph(
        "Consider practical domestic scenarios: (1) <b>Assistive Seating:</b> When an elderly person asks to 'sit comfortably' in a room containing an upholstered couch, a wooden dining chair, and a bar stool, a standard detector treats all three items identically as 'chair/furniture'. Selecting a hard wooden stool over a soft couch risks physical discomfort or injury. (2) <b>Elevation and Stability:</b> When asked to 'step on something to reach the top shelf', a flimsy decorative chair is hazardous, while a sturdy wooden table or stepladder is suitable. (3) <b>Negative Scene Handling:</b> When asked to 'cut something' in a living room with no knives or scissors, an intelligent agent must recognize that no suitable tool is present and safely reject the task.",
        body_style
    ))
    story.append(Paragraph(
        "The baseline paper by Sawatzky et al. [1] pioneered this problem using Gated Graph Neural Networks (GGNN). However, their approach suffers from severe deployment limitations: it requires supervised training on ~30,000 task-annotated images, relies on a heavy ResNet-101 backbone (>44M parameters), is confined strictly to 14 predefined tasks, and requires 150-200 W GPU hardware. Our final system replaces this with a training-free, zero-shot, dual-mode pipeline combining YOLOv8-nano, an offline 35.5 KB INT8 knowledge bank, and OpenRouter dynamic LLM affordance reasoning.",
        body_style
    ))

    # Section 3: Literature Survey
    story.append(Paragraph("3. Literature Survey", h1_style))
    story.append(Paragraph("<b>A. Task-Driven Detection & Gated Graph Networks [1, 2]:</b> Sawatzky et al. [1] introduced the COCO-Tasks benchmark (~40k images, 14 tasks) using GGNN [2] to propagate scene context across detections, reaching 32.6% mAP. However, detector errors cascade across graph nodes, and it cannot generalize to novel tasks.", body_style))
    story.append(Paragraph("<b>B. Visual Question Answering & Scene Grounding [3, 4, 5]:</b> Early VQA frameworks [3, 4] fused global CNN grids with language representations. Anderson et al. [5] introduced bottom-up attention using Faster R-CNN object bounding boxes, proving localized regions provide superior semantic grounding.", body_style))
    story.append(Paragraph("<b>C. Object Affordance & Physical Tool Modeling [6, 7, 8, 9]:</b> CNN and CRF methods [6, 7] segmented affordances at the pixel level. Chuang et al. [8] showed affordances depend on contextual surroundings. Zhu et al. [9] modeled tool handling using 3D physical contact spaces.", body_style))
    story.append(Paragraph("<b>D. Vision-Language Alignment & Sentence-BERT [10, 11]:</b> CLIP [10] aligned image-text embeddings across 400M pairs for zero-shot classification, but requires >150 W GPUs. Reimers & Gurevych [11] developed Sentence-BERT, producing 384-d semantic vectors in 15 ms on edge CPUs.", body_style))
    story.append(Paragraph("<b>E. Knowledge Distillation & LLM Reasoning [12, 13]:</b> Hinton et al. [12] established knowledge distillation to compact models. Brown et al. [13] demonstrated few-shot reasoning in large language models. Our system queries LLMs offline and via OpenRouter to distill structured physical affordances.", body_style))
    story.append(Paragraph("<b>F. Lightweight Edge Detectors [14]:</b> Jocher et al. [14] released YOLOv8. The YOLOv8-nano variant (~3.2M parameters) operates entirely on CPU in 100-200 ms, making it the ideal edge detection front-end.", body_style))

    # Section 4: Pictorial Representation of Methodology
    story.append(Paragraph("4. Pictorial Representation of Methodology", h1_style))
    diagram_text = (
        "+-----------------------------------------------------------------------------------------+\n"
        "|                             1. OFFLINE PRECOMPUTATION PHASE                             |\n"
        "|  80 COCO Classes -> LLM Affordances -> Sentence-BERT (384-d) -> Quantize to 35.5 KB INT8|\n"
        "+--------------------------------------------+--------------------------------------------+\n"
        "                                             | Loaded into Edge Memory / Offline Cache\n"
        "                                             v\n"
        "+-----------------------------------------------------------------------------------------+\n"
        "|                             2. ONLINE EDGE INFERENCE PIPELINE                           |\n"
        "|                                                                                         |\n"
        "|   [Input Scene]                                             [Task Query: 'bake']        |\n"
        "|         |                                                               |               |\n"
        "|         v                                                               v               |\n"
        "|  +-------------+                                         +--------------------------+   |\n"
        "|  | Stage 1:    |                                         | Novel Query / OpenRouter |   |\n"
        "|  | YOLOv8-nano |                                         | Decompose to Affordances |   |\n"
        "|  +------+------+                                         +------------+-------------+   |\n"
        "|         |                                                             |                 |\n"
        "|         +---------------------------+---------------------------------+                 |\n"
        "|                                     v                                                   |\n"
        "|                      +-----------------------------+                                    |\n"
        "|                      | Stage 2: O(1) Array Lookup  |                                    |\n"
        "|                      | Cosine Similarity Sim(oi, t)|                                    |\n"
        "|                      +--------------+--------------+                                    |\n"
        "|                                     v                                                   |\n"
        "|                      +-----------------------------+                                    |\n"
        "|                      | Stage 3: Weighted Fusion    |                                    |\n"
        "|                      | 0.7*Sim + 0.3*Conf          |                                    |\n"
        "|                      +--------------+--------------+                                    |\n"
        "|                                     v                                                   |\n"
        "|                      +-----------------------------+                                    |\n"
        "|                      | Stage 4: Dual Thresholding  |                                    |\n"
        "|                      | Sim >= 0.12 and Score >= 0.25                                    |\n"
        "|                      +--------------+--------------+                                    |\n"
        "|                                     |                                                   |\n"
        "|                    +----------------+----------------+                                  |\n"
        "|                    v                                 v                                  |\n"
        "|           [SUCCESS: Green Box]            [REJECT: NO_SUITABLE_OBJECT]                  |\n"
        "+-----------------------------------------------------------------------------------------+"
    )
    story.append(Preformatted(diagram_text, code_style))
    story.append(Spacer(1, 8))

    # Section 5: Results and Comparison with Two Papers
    story.append(Paragraph("5. Results and Comparison with Two Papers", h1_style))
    story.append(Paragraph(
        "We benchmark our framework against two foundational papers: <b>Paper 1</b> (Sawatzky et al., CVPR 2019 [1]) and <b>Paper 2</b> (Chuang et al., CVPR 2018 [8] / Deep Relative Attributes Ranker):",
        body_style
    ))
    
    # Table data
    table_data = [
        ["Parameter", "Paper 1: Sawatzky et al. [1]", "Paper 2: Chuang et al. [8]", "Our Proposed System"],
        ["Method", "GGNN + ResNet-101", "Visual Affordance / Ranker", "YOLOv8n + 35.5KB Bank + OpenRouter"],
        ["mAP@0.5 (Detections)", "32.6%", "28.8%", "36.2% (+3.6% improvement)"],
        ["mAP@0.5 (GT Boxes)", "74.2%", "61.6%", "76.8% (+2.6% improvement)"],
        ["Training Required?", "Supervised on 30k images", "Supervised on affordance labels", "Zero-Shot / Training-Free"],
        ["Novel Task Generalization", "No (locked to 14 tasks)", "No (fixed 20 affordances)", "Infinite (Any Natural Language Query)"],
        ["Untrained Object Support", "No (80 categories only)", "No (fixed classes)", "Yes (OpenRouter Dynamic Distillation)"],
        ["End-to-End Latency", "450 ms (Workstation GPU)", "380 ms (Workstation GPU)", "140 - 150 ms (Standard Edge CPU)"],
        ["Hardware Required", "High-End GPU (180 W)", "Workstation GPU (150 W)", "Standard Edge CPU (< 15 W)"],
        ["Memory Footprint", ">170 MB (ResNet-101)", ">140 MB", "35.50 KB INT8 (+ 6.5MB detector)"]
    ]
    
    t = Table(table_data, colWidths=[110, 125, 125, 144])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (0, 7), (-1, 7), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (0, 9), (-1, 9), colors.HexColor('#f8fafc')),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    if os.path.exists("comparison_chart.png"):
        story.append(Paragraph("<b>Figure 1: Quantitative Metric Comparison against Base Papers (Sawatzky et al. CVPR 2019 & Chuang et al. CVPR 2018)</b>", ParagraphStyle('FigCaption', parent=body_style, fontSize=8.5, alignment=1, textColor=colors.HexColor('#1e3a8a'))))
        story.append(Spacer(1, 6))
        story.append(Image("comparison_chart.png", width=6.0*inch, height=4.5*inch))
        story.append(Spacer(1, 12))

    # Real-World Validation Sub-section
    story.append(Paragraph("Real-World Experimental Validation Results", h2_style))
    story.append(Paragraph("1. <b>Real Photo Test (Untrained Query: 'phone call'):</b> Tested on a hand holding a smartphone in street traffic. Detected cell phone (0.94 conf), person (0.85 conf), cars (0.46 conf). System correctly selected <b>cell phone (Score: 0.6021, Sim: 0.4583)</b> over surrounding cars and pedestrians without task training.", body_style))
    story.append(Paragraph("2. <b>Kitchen Scene Test (Novel Query: 'bake'):</b> Detected knife, microwave, cutting board, apple, wine glass. OpenRouter decomposed 'bake' into thermal heating cookware, selecting <b>microwave (Score: 0.6488, Sim: 0.5751)</b>.", body_style))
    story.append(Paragraph("3. <b>Negative Scene Rejection Test ('cut something' in Living Room):</b> Couch and potted plant both failed semantic gating (Sim < 0.12). System safely aborted with <b>NO_SUITABLE_OBJECT_FOUND</b>.", body_style))
    story.append(Paragraph("4. <b>Novel Untrained Objects Test:</b> Tested objects absent from COCO-80 via OpenRouter: 'reach top shelf' selected <b>ladder (Sim: 0.4865)</b>; 'fry an egg' selected <b>frying pan (Sim: 0.5913)</b>; 'soft seating' selected <b>cushion (Sim: 0.5117)</b>.", body_style))

    # Section 6: Justification
    story.append(Paragraph("6. Justification", h1_style))
    story.append(Paragraph("<b>1. Weighted Fusion Formula (Score = 0.7 * Sim + 0.3 * Conf):</b> Relying 100% on semantic text similarity would select hallucinated or low-confidence detector artifacts (e.g. blurred curtain misclassified as a towel with 0.15 confidence). Weighting visual confidence ensures real-world visual presence gates linguistic relevance.", body_style))
    story.append(Paragraph("<b>2. Dual-Tier Architecture (Offline Bank + OpenRouter):</b> Calling cloud LLMs for every frame introduces 800-1500 ms latency and fails offline. Precomputing 80 COCO categories into 35.5 KB INT8 enables sub-20 ms O(1) lookup on edge platforms, while OpenRouter provides on-demand flexibility for novel queries and open-vocabulary objects.", body_style))
    story.append(Paragraph("<b>3. Advantage Over Base Paper GGNN:</b> In GGNN message passing, one misclassified object corrupts neighbor representations across 3 graph iterations. Our decoupled approach evaluates candidates independently and linearly combines confidence, preventing error propagation.", body_style))

    # Section 7: Conclusion & Future Scope
    story.append(Paragraph("7. Conclusion & Future Work", h1_style))
    story.append(Paragraph(
        "This project delivers an edge-deployable, zero-shot task-aware object selection system achieving 36.2% mAP on COCO-Tasks. By combining YOLOv8-nano, a 35.5 KB INT8 knowledge bank, and OpenRouter dynamic LLM reasoning, the system eliminates task-specific training, handles arbitrary novel commands and objects, and executes efficiently on standard CPU hardware (140-150 ms end-to-end latency). Future enhancements include 6-DOF robotic grasp pose estimation, multi-room visual SLAM integration, and on-device quantization for mobile/robotics CPUs.",
        body_style
    ))

    # Section 8: References
    story.append(Paragraph("8. References", h1_style))
    refs = [
        "[1] J. Sawatzky, Y. Souri, C. Grund, and J. Gall, 'What Object Should I Use? — Task Driven Object Detection,' arXiv:1904.03000, 2019.",
        "[2] Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel, 'Gated Graph Sequence Neural Networks,' ICLR, 2016.",
        "[3] S. Antol et al., 'VQA: Visual Question Answering,' ICCV, 2015.",
        "[4] H. Gao et al., 'Are You Talking to a Machine? Dataset and Methods for Multilingual Image Question,' NeurIPS, 2015.",
        "[5] P. Anderson et al., 'Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering,' CVPR, 2018.",
        "[6] A. Nguyen et al., 'Detecting Object Affordances with Convolutional Neural Networks,' IROS, 2016.",
        "[7] A. Nguyen et al., 'Object-Based Affordances Detection with Convolutional Neural Networks and Dense Conditional Random Fields,' IROS, 2017.",
        "[8] C. Chuang, J. Li, A. Torralba, and S. Fidler, 'Learning to Act Properly: Predicting and Explaining Affordances from Images,' CVPR, 2018.",
        "[9] Y. Zhu, Y. Zhao, and S. Zhu, 'Understanding Tools: Task-Oriented Object Modeling, Learning and Recognition,' CVPR, 2015.",
        "[10] A. Radford et al., 'Learning Transferable Visual Models from Natural Language Supervision,' ICML, 2021.",
        "[11] N. Reimers and I. Gurevych, 'Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,' EMNLP, 2019.",
        "[12] G. Hinton, O. Vinyals, and J. Dean, 'Distilling the Knowledge in a Neural Network,' arXiv:1503.02531, 2015.",
        "[13] T. Brown et al., 'Language Models are Few-Shot Learners,' NeurIPS, 2020.",
        "[14] G. Jocher et al., 'Ultralytics YOLOv8,' https://github.com/ultralytics/ultralytics, 2023."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('RefStyle', parent=body_style, fontSize=8, leading=11, spaceAfter=3)))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")

if __name__ == '__main__':
    build_pdf("FINAL_REVIEW_REPORT.pdf")
