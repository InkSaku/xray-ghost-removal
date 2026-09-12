"""Generate the six figures referenced in the Hangzhou NSF feasibility report.

Figures 1-2 are produced from the real preliminary DICOM data and the
existing ghost-removal code. Figures 3-6 are schematic flowcharts drawn
programmatically (labelled 拟开展/示意 in the proposal).

Outputs: figures/final/fig1_ghost_visibility.png ... fig6_cascade.png
Run: python scripts/generate_proposal_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.dicom_utils import load_dicom
from src.models.physics_model import detect_air_mask
from src.models.linear_ghost import estimate_alphas, remove_ghost_gated

# --- Chinese-capable font -------------------------------------------------
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 300

OUT = Path("figures/final")
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("data/raw/残影图像")

# Palette (colorblind-friendly)
C_PHYS = "#4C72B0"
C_CAL = "#55A868"
C_PRE = "#C44E52"
C_FT = "#8172B3"
C_VAL = "#CCB974"
C_GATE = "#64B5CD"
C_BOX = "#E8EEF4"
C_EDGE = "#2E4057"


# ============================ shared drawing helpers =====================
def box(ax, x, y, w, h, title, body="", fc=C_BOX, ec=C_EDGE, tsize=11, bsize=9,
        tcolor="#12263A"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    cx, cy = x + w / 2, y + h / 2
    if body:
        ax.text(cx, y + h - 0.02 * h + h * 0.30, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", color=tcolor, zorder=3)
        ax.text(cx, cy - h * 0.16, body, ha="center", va="center",
                fontsize=bsize, color="#33475B", zorder=3, wrap=True)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=tsize,
                fontweight="bold", color=tcolor, zorder=3)
    return (cx, cy)


def arrow(ax, p1, p2, color=C_EDGE, style="-|>", lw=1.8, rad=0.0):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                        linewidth=lw, color=color,
                        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)


def diamond(ax, cx, cy, w, h, text, fc=C_GATE, ec=C_EDGE, size=10):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec,
                         linewidth=1.6, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size,
            fontweight="bold", color="#12263A", zorder=3)


def clean_axes(ax, xlim=(0, 10), ylim=(0, 10)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


# ============================ FIG 1 (real data) ==========================
def fig1():
    cur, _ = load_dicom(DATA / "4.dcm")
    prev, _ = load_dicom(DATA / "3.dcm")
    prev2, _ = load_dicom(DATA / "2.dcm")

    air = detect_air_mask(cur)
    amed = float(np.median(cur[air]))
    astd = float(np.std(cur[air]))

    # ghost estimate to localise the artifact for the annotation arrow
    a, bgs, r2, _ = estimate_alphas(cur, [prev, prev2])
    bg = float(np.percentile(prev, 75))
    gmap = a[0] * (prev - bg)
    gmap_air = np.where(air, np.abs(gmap), 0.0)
    gmap_s = ndimage.gaussian_filter(gmap_air, 25)
    peak = np.unravel_index(np.argmax(gmap_s), gmap_s.shape)  # (row, col)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2))
    # left: normal display window
    v1, v99 = np.percentile(cur, [1, 99])
    axes[0].imshow(cur, cmap="gray", vmin=v1, vmax=v99)
    axes[0].set_title("常规窗宽窗位\n残影几乎不可见", fontsize=12)
    # right: compressed window around the air level -> reveals the ghost
    axes[1].imshow(cur, cmap="gray", vmin=amed - 1.8 * astd, vmax=amed + 1.8 * astd)
    axes[1].set_title(f"压缩窗宽（背景区±1.8σ）\n前序工件残影显现（α≈{a[0]:.4f}）", fontsize=12)
    # annotate ghost
    axes[1].annotate("残影轮廓\n（形似未熔合/裂纹）",
                     xy=(peak[1], peak[0]), xytext=(peak[1] + 700, peak[0] + 650),
                     color="#FFD400", fontsize=11, fontweight="bold", ha="center",
                     arrowprops=dict(arrowstyle="->", color="#FFD400", lw=2.2))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("图1  余辉残影在高对比评片下被误判为缺陷（由前期真实数据生成）",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "fig1_ghost_visibility.png", bbox_inches="tight")
    plt.close(fig)
    print(f"fig1 done  (alpha={a[0]:.5f}, r2={r2:.3f}, peak={peak})")


# ============================ FIG 2 (real data) ==========================
def fig2():
    cur, _ = load_dicom(DATA / "4.dcm")
    prev, _ = load_dicom(DATA / "3.dcm")
    prev2, _ = load_dicom(DATA / "2.dcm")
    res = remove_ghost_gated(cur, [prev, prev2], r2_threshold=0.0)  # force apply for the true thumb
    cleaned = res.cleaned

    a, bgs, r2, _ = estimate_alphas(cur, [prev, prev2])
    alpha = a[0]

    # background-region samples for the scatter
    air = detect_air_mask(cur)[::4, ::4]
    cds = cur[::4, ::4]
    pds = prev[::4, ::4]
    air_level = float(np.median(cds[air]))
    bg = float(np.percentile(prev, 75))
    x = (pds[air] - bg)
    y = (cds[air] - air_level)
    if x.size > 4000:
        sel = np.random.default_rng(0).choice(x.size, 4000, replace=False)
        x, y = x[sel], y[sel]

    fig = plt.figure(figsize=(10, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.5], hspace=0.32, wspace=0.15)

    # --- top: additive-model thumbnails ---
    v1, v99 = np.percentile(cur, [2, 98])
    thumbs = [(cur, "I_obs  观测(含残影)"), (cleaned, "I_true  真实(去残影)"),
              (prev, "I_prev  前序帧")]
    for j, (img, lab) in enumerate(thumbs):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img[::6, ::6], cmap="gray", vmin=v1, vmax=v99)
        ax.set_title(lab, fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.5, 0.60,
             r"$I_{obs}(n)=I_{true}(n)+\alpha\,(I_{prev}-bg)+$噪声",
             ha="center", fontsize=13)

    # --- bottom: real least-squares regression ---
    axr = fig.add_subplot(gs[1, :])
    axr.scatter(x, y, s=4, alpha=0.25, color=C_PHYS, edgecolors="none",
                label="背景区像素样本")
    xs = np.linspace(x.min(), x.max(), 100)
    axr.plot(xs, alpha * xs, color=C_PRE, lw=2.4,
             label=f"最小二乘拟合  斜率 α ≈ {alpha:.4f}")
    axr.axhline(0, color="#999", lw=0.8); axr.axvline(0, color="#999", lw=0.8)
    axr.set_xlabel("前序帧背景像素  (I_prev − bg)", fontsize=11)
    axr.set_ylabel("当前帧背景像素  (I_obs − 背景电平)", fontsize=11)
    axr.set_title(f"背景区最小二乘标定混叠系数 α（真实前期数据，R²={r2:.3f}）",
                  fontsize=11.5)
    axr.legend(loc="upper left", fontsize=10, framealpha=0.9)
    axr.grid(alpha=0.25)

    fig.suptitle("图2  CsI余辉残影叠加模型与背景区 α 标定（由前期代码生成）",
                 fontsize=13, fontweight="bold", y=0.97)
    fig.savefig(OUT / "fig2_alpha_regression.png", bbox_inches="tight")
    plt.close(fig)
    print(f"fig2 done  (alpha={alpha:.5f}, r2={r2:.3f})")


# ============================ FIG 3 route ================================
def fig3():
    fig, ax = plt.subplots(figsize=(13, 4.6)); clean_axes(ax, (0, 13), (0, 4.6))
    stages = [
        ("物理基线", "DICOM读取\n背景区检测\n可分离性门控减除", C_PHYS),
        ("α 标定", "CsI真实配对采集\n余辉衰减/饱和\n时空标定", C_CAL),
        ("合成预训练", "物理一致合成样本\nU-Net预训练", C_PRE),
        ("真实微调", "配对数据\n监督微调", C_FT),
        ("门控与级联验证", "全参考/无参考评价\n下游缺陷检测精度", C_VAL),
    ]
    w, h, gap = 2.15, 2.2, 0.30
    x = 0.25; centers = []
    for title, body, col in stages:
        c = box(ax, x, 1.3, w, h, title, body, fc=col + "22", ec=col, tsize=12,
                bsize=9)
        centers.append((x, c)); x += w + gap
    for i in range(len(stages) - 1):
        x0 = centers[i][0] + w
        arrow(ax, (x0, 2.4), (x0 + gap, 2.4), color=C_EDGE, lw=2.0)
    ax.text(6.5, 0.6, "输入 → 输出贯穿各研究内容；门控保证“宁可不改、不可改错”",
            ha="center", fontsize=10, color="#33475B")
    ax.set_title("图3  项目总体技术路线（拟开展，示意）", fontsize=13,
                 fontweight="bold")
    fig.savefig(OUT / "fig3_route.png", bbox_inches="tight"); plt.close(fig)
    print("fig3 done")


# ============================ FIG 4 gating ===============================
def fig4():
    fig, ax = plt.subplots(figsize=(7.2, 8.6)); clean_axes(ax, (0, 7.2), (0, 8.6))
    c1 = box(ax, 1.6, 7.4, 4.0, 0.9, "输入：当前帧 + 前序帧", fc=C_BOX)
    c2 = box(ax, 1.6, 6.0, 4.0, 0.9, "背景区最小二乘估计 α 与噪声 σ", fc=C_BOX)
    diamond(ax, 3.6, 4.4, 4.6, 1.9, "混叠信噪比\n α·√N·s_x / σ  >  z ?", fc=C_GATE + "cc",
            size=10.5)
    cyes = box(ax, 0.4, 1.9, 3.0, 1.0, "执行去除", "输出洁净图像", fc=C_CAL + "22",
               ec=C_CAL)
    cno = box(ax, 3.9, 1.9, 3.0, 1.0, "保持原图不变", "记录“安全跳过”", fc=C_PRE + "22",
              ec=C_PRE)
    arrow(ax, (3.6, 7.4), (3.6, 6.9)); arrow(ax, (3.6, 6.0), (3.6, 5.35))
    arrow(ax, (1.3, 4.4), (1.9, 2.9)); ax.text(1.0, 3.7, "是", fontsize=11,
                                               fontweight="bold", color=C_CAL)
    arrow(ax, (5.9, 4.4), (5.4, 2.9)); ax.text(6.0, 3.7, "否", fontsize=11,
                                               fontweight="bold", color=C_PRE)
    ax.text(3.6, 0.9, "阈值 z 由可分离性理论 Var(α̂)≈σ²/(N·s_x²) 确定，替代 R² 启发式",
            ha="center", fontsize=9.5, color="#33475B")
    ax.set_title("图4  可分离性门控的决策流程（拟开展，示意）", fontsize=13,
                 fontweight="bold")
    fig.savefig(OUT / "fig4_gating.png", bbox_inches="tight"); plt.close(fig)
    print("fig4 done")


# ============================ FIG 5 FPD protocol =========================
def fig5():
    fig, ax = plt.subplots(figsize=(12.5, 6.2)); clean_axes(ax, (0, 12.5), (0, 6.2))
    steps = [
        ("① 长暗场间隔", "余辉充分衰减\n（不作擦除）", C_PHYS),
        ("② 成像工件 A", "暗场校正\n→ clean（洁净真值）", C_CAL),
        ("③ 高曝光成像 B", "→ previous\n（残影源）", C_FT),
        ("④ 短延时 Δt 后\n再成像 A", "→ ghosted\n（含真实余辉残影）", C_PRE),
    ]
    w, h, gap = 2.55, 2.1, 0.35; x = 0.25; centers = []
    for title, body, col in steps:
        c = box(ax, x, 3.4, w, h, title, body, fc=col + "22", ec=col, tsize=11,
                bsize=9)
        centers.append((x, c)); x += w + gap
    for i in range(len(steps) - 1):
        x0 = centers[i][0] + w
        arrow(ax, (x0, 4.45), (x0 + gap, 4.45), lw=2.0)
    box(ax, 1.2, 1.1, 5.3, 1.6,
        "α_true = mean(ghosted − clean) / mean(previous − bg)",
        "= f ( 剂量,  Δt,  位置 )",
        fc="#FFF7E6", ec=C_VAL, tsize=11, bsize=11)
    box(ax, 6.9, 1.1, 5.3, 1.6, "跨样本变化的变量",
        "B曝光剂量 · 帧间时间 Δt（余辉衰减轴）\nB形态 · 板上位置 · A曝光量\n先导：最小可行 10 对",
        fc=C_BOX, ec=C_EDGE, tsize=11, bsize=9)
    arrow(ax, (3.6, 3.4), (3.6, 2.7), rad=0.0)
    ax.set_title("图5  面向平板探测器(FPD)的真实配对数据采集协议（拟开展，示意）",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT / "fig5_fpd_protocol.png", bbox_inches="tight"); plt.close(fig)
    print("fig5 done")


# ============================ FIG 6 cascade ==============================
def fig6():
    fig, ax = plt.subplots(figsize=(12.0, 6.6)); clean_axes(ax, (0, 12.0), (0, 6.6))
    src = box(ax, 0.3, 3.0, 3.0, 1.5, "独立真实\n含残影焊缝数据",
              "（合作企业提供，\n带缺陷标注）", fc=C_BOX, tsize=11, bsize=8.5)
    # control path (top)
    det1 = box(ax, 5.2, 4.5, 3.0, 1.3, "AI 缺陷识别", "对照：含残影输入", fc=C_PRE + "22",
               ec=C_PRE)
    # experiment path (bottom)
    deg = box(ax, 4.3, 1.1, 2.7, 1.3, "去残影模型", "本项目", fc=C_CAL + "22", ec=C_CAL)
    det2 = box(ax, 7.6, 1.1, 3.0, 1.3, "AI 缺陷识别", "实验：去残影输入", fc=C_PRE + "22",
               ec=C_PRE)
    metr = box(ax, 9.0, 4.2, 2.7, 1.9, "指标对比",
               "Precision / Recall\nmAP\n残影虚警率", fc=C_VAL + "33", ec=C_VAL,
               tsize=11, bsize=9)
    arrow(ax, (3.3, 4.0), (5.2, 5.15), rad=-0.15)
    arrow(ax, (3.3, 3.5), (4.3, 1.75), rad=0.15)
    arrow(ax, (7.0, 1.75), (7.6, 1.75))
    arrow(ax, (8.2, 5.15), (9.0, 5.15))
    arrow(ax, (10.6, 2.4), (10.4, 4.2), rad=0.2)
    ax.text(6.0, 0.35,
            "先导阶段：向 GDXray 公开焊缝图像注入合成余辉，度量 YOLO 虚警率上升→去残影后回落",
            ha="center", fontsize=9.5, color="#33475B")
    ax.set_title("图6  去伪影与下游缺陷识别级联及其独立评估（拟开展，示意）",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT / "fig6_cascade.png", bbox_inches="tight"); plt.close(fig)
    print("fig6 done")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("\nAll figures written to", OUT.resolve())
