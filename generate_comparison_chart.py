"""
Generates publication-quality comparison charts focusing on ARCHITECTURAL METHODS:
- Method 1: Deep Affordance Ranker (VGG-16 Backbone) [Chuang et al., CVPR 2018]
- Method 2: Graph Message-Passing (GGNN + ResNet-101) [Sawatzky et al., CVPR 2019]
- Method 3: Decoupled Dual-Tier (YOLOv8n + INT8 S-BERT) [Proposed System]
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
fig.suptitle("Architectural & Methodological Benchmark Comparison", fontsize=16, fontweight='bold', y=0.98)

methods = [
    "Deep Affordance Ranker\n(VGG-16 Backbone)",
    "Graph Message-Passing\n(GGNN + ResNet-101)",
    "Decoupled Dual-Tier\n(YOLOv8n + INT8 Bank)"
]
colors = ['#94a3b8', '#64748b', '#2563eb']

# ----------------- SUBPLOT 1: Accuracy (mAP@0.5 %) -----------------
ax1 = axs[0, 0]
x = np.arange(len(methods))
width = 0.35

det_map = [28.8, 32.6, 36.2]
gt_map = [61.6, 74.2, 76.8]

rects1 = ax1.bar(x - width/2, det_map, width, label='Detections mAP@0.5', color='#3b82f6', edgecolor='black', linewidth=0.5)
rects2 = ax1.bar(x + width/2, gt_map, width, label='GT Boxes mAP@0.5', color='#10b981', edgecolor='black', linewidth=0.5)

ax1.set_ylabel('mAP@0.5 (%)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Task-Driven Selection Accuracy (Higher is Better)', fontsize=12, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=9.5)
ax1.set_ylim(0, 92)
ax1.legend(loc='upper left', frameon=True)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# ----------------- SUBPLOT 2: End-to-End Latency (ms) -----------------
ax2 = axs[0, 1]
latencies = [380, 450, 145]
bars2 = ax2.bar(methods, latencies, color=['#cbd5e1', '#cbd5e1', '#0284c7'], width=0.5, edgecolor='black', linewidth=0.5)
ax2.set_ylabel('Latency (ms)', fontsize=11, fontweight='bold')
ax2.set_title('(b) End-to-End Inference Latency (Lower is Better)', fontsize=12, fontweight='bold', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=9.5)
ax2.set_ylim(0, 520)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars2:
    h = bar.get_height()
    device_label = " (GPU)" if h > 200 else " (CPU)"
    ax2.annotate(f'{h} ms{device_label}', xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

# ----------------- SUBPLOT 3: Power Consumption (Watts) -----------------
ax3 = axs[1, 0]
power = [150, 180, 14]
bars3 = ax3.bar(methods, power, color=['#fda4af', '#fda4af', '#059669'], width=0.5, edgecolor='black', linewidth=0.5)
ax3.set_ylabel('Operational Power Draw (Watts)', fontsize=11, fontweight='bold')
ax3.set_title('(c) Hardware Power Consumption (Lower is Better)', fontsize=12, fontweight='bold', pad=10)
ax3.set_xticks(x)
ax3.set_xticklabels(methods, fontsize=9.5)
ax3.set_ylim(0, 220)
ax3.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars3:
    h = bar.get_height()
    ax3.annotate(f'{h} W', xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax3.text(0.68, 0.78, "92.2% Power Savings\nvia CPU Decoupling", transform=ax3.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", fc="#dcfce7", ec="#16a34a", lw=1.2), fontsize=9, fontweight='bold', color="#166534")

# ----------------- SUBPLOT 4: Training Overhead vs Generalization -----------------
ax4 = axs[1, 1]
training_images = [15000, 30000, 0]
bars4 = ax4.bar(methods, training_images, color=['#fde047', '#fde047', '#8b5cf6'], width=0.5, edgecolor='black', linewidth=0.5)
ax4.set_ylabel('Task-Annotated Training Images', fontsize=11, fontweight='bold')
ax4.set_title('(d) Task Supervision Requirement (0 = Training-Free)', fontsize=12, fontweight='bold', pad=10)
ax4.set_xticks(x)
ax4.set_xticklabels(methods, fontsize=9.5)
ax4.set_ylim(0, 35000)
ax4.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars4:
    h = bar.get_height()
    txt = f"{h:,} imgs" if h > 0 else "0 (Zero-Shot / No Retraining)"
    ax4.annotate(txt, xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig("comparison_chart.png", dpi=300, bbox_inches='tight')
plt.savefig("report/graph.png", dpi=300, bbox_inches='tight')
print("Successfully generated comparison_chart.png and report/graph.png with method-based comparisons!")

