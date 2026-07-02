"""Generate the Language-3D system architecture figure (Fig 1).

Produces a publication-quality PDF/PNG showing the multi-agent pipeline,
dual-channel verification loop, and output package.  No external deps
beyond matplotlib (already in the project's virtual env for VTK).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Colors (clean academic palette)
C_INPUT = "#E8F0FE"
C_AGENT = "#4285F4"
C_VERIFY = "#34A853"
C_FIX = "#EA4335"
C_OUTPUT = "#FBBC04"
C_ARROW = "#5F6368"
C_BG = "#F8F9FA"

fig, ax = plt.subplots(1, 1, figsize=(12, 5.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.5)
ax.axis("off")
ax.set_facecolor("white")

def box(x, y, w, h, text, color, fontsize=8, textcolor="white", style="round"):
    """Draw a rounded box with centered text."""
    fancy = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"{style},pad=0.15",
        facecolor=color, edgecolor="#333", linewidth=0.8, zorder=2,
    )
    ax.add_patch(fancy)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, fontweight="bold", zorder=3)

def arrow(x1, y1, x2, y2, color=C_ARROW, style="-|>", lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def darrow(x1, y1, x2, y2, color="#EA4335", style="<|-|>", lw=1.0):
    """Double-headed arrow (feedback loop)."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0.3"))

# Title
ax.text(6, 5.2, "Language-3D: System Architecture",
        ha="center", fontsize=14, fontweight="bold")

# Input
box(0.2, 3.5, 1.6, 0.8, "Natural Language\n\"4-DOF arm\nwith gripper\"",
    C_INPUT, fontsize=7, textcolor="#333")

# Agent pipeline (horizontal chain)
agents = [
    (2.2, "Architect\n(GLM-5.2)", C_AGENT),
    (3.8, "Solver\n(FCL+anchor)", C_AGENT),
    (5.4, "CAD Engineer\n(FreeCAD)", C_AGENT),
]
for x, label, color in agents:
    box(x, 3.5, 1.4, 0.8, label, color, fontsize=7)

# Arrows between agents
for i in range(len(agents) - 1):
    x1 = agents[i][0] + 1.4
    x2 = agents[i+1][0]
    arrow(x1, 3.9, x2, 3.9)
arrow(1.8, 3.9, 2.2, 3.9)  # input → architect

# Assembly JSON intermediate
box(3.6, 2.5, 1.0, 0.5, "assembly\nJSON", "#E0E0E0", fontsize=6, textcolor="#333")

# Verification (dual channel)
box(7.0, 4.2, 1.6, 0.8, "VLM Verify\n(GLM-4.6V)", C_VERIFY, fontsize=7)
box(7.0, 2.8, 1.6, 0.8, "Geometric\nArbitration\n(FCL+seq)", C_VERIFY, fontsize=6.5)

arrow(6.8, 4.0, 7.0, 4.5)   # cad → vlm
arrow(6.8, 3.8, 7.0, 3.2)   # cad → geo
darrow(8.6, 4.4, 8.6, 3.4, color="#333", style="<|-|>")  # vlm ↔ geo arbitration

# Fixer
box(7.0, 1.3, 1.6, 0.8, "Fixer\n(severity\nrouting)", C_FIX, fontsize=7)
arrow(7.8, 2.8, 7.8, 2.1, color=C_FIX, style="-|>")  # verify fail → fixer

# Feedback loop (fixer → back to agent stages)
ax.annotate("", xy=(4.5, 1.7), xytext=(7.0, 1.7),
            arrowprops=dict(arrowstyle="-|>", color=C_FIX, lw=1.2,
                            connectionstyle="arc3,rad=-0.4"))
ax.text(5.5, 1.0, "feedback\n(targeted fix)", ha="center", fontsize=6,
        color=C_FIX, style="italic")

# Export (output package)
box(9.5, 3.5, 2.3, 0.8, "Engineering\nPackage Export", C_OUTPUT, fontsize=7, textcolor="#333")
arrow(8.6, 3.9, 9.5, 3.9, color="#333")  # verify pass → export

# Output artifacts (vertical list)
outputs = [
    "STL + STEP meshes",
    "URDF (MuJoCo-ready)",
    "BOM (COTS parts)",
    "Firmware (C++/Arduino)",
    "ROS2 package",
    "Assembly guide",
]
for i, item in enumerate(outputs):
    y = 2.8 - i * 0.35
    ax.text(9.6, y, f"• {item}", fontsize=6.5, color="#333", family="monospace")

# Physics validation (bottom)
box(2.2, 0.2, 4.8, 0.7, "MuJoCo Physics Validation: <actuator> model + ground contact + grasp test + collision sweep",
    "#E8F5E9", fontsize=6.5, textcolor="#333")
arrow(6.0, 0.9, 6.0, 1.3, color="#34A853", style="-|>")  # physics → fixer/verify

# Legend
ax.text(0.2, 4.8, "Pipeline stages:", fontsize=7, fontweight="bold", color="#333")
for i, (label, color) in enumerate([
    ("LLM Agent", C_AGENT), ("Verification", C_VERIFY),
    ("Fixer", C_FIX), ("Output", C_OUTPUT)
]):
    ax.add_patch(mpatches.Rectangle((0.2 + i*1.3, 4.5), 0.25, 0.2,
                                     facecolor=color, edgecolor="#333", lw=0.5))
    ax.text(0.5 + i*1.3, 4.6, label, fontsize=6, color="#333", va="center")

plt.tight_layout()
out_dir = Path("docs/paper")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "fig1_architecture.pdf", dpi=300, bbox_inches="tight")
fig.savefig(out_dir / "fig1_architecture.png", dpi=200, bbox_inches="tight")
print(f"Saved: {out_dir / 'fig1_architecture.pdf'}")
print(f"Saved: {out_dir / 'fig1_architecture.png'}")
