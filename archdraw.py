import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams['svg.fonttype'] = 'none'  # 导出文字为矢量，不转曲
plt.rcParams['font.size'] = 9

# -------------------------- 画布（矢量专用） --------------------------
fig, ax = plt.subplots(figsize=(12, 16), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 17)
ax.axis('off')

# 配色（柔和论文风）
colors = {
    "vit": "#fef0d1",     # DINO
    "toolinit": "#e0edf8",# 原创核心
    "spatial": "#f8e2e5", # 关键点
    "kp": "#e3f5e3",      # 投影
    "diff": "#f3e6f5",    # 扩散头
    "text": "#222222"
}

# -------------------------- 绘图工具 --------------------------
def box(x, y, w, h, color, texts):
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="#666", linewidth=1.2
    )
    ax.add_patch(rect)
    ty = y + h/2
    for i, t in enumerate(texts):
        ax.text(x + w/2, ty + (len(texts)/2 - i)*0.25,
                t, ha="center", va="center", color=colors["text"])

def arr(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->,head_width=0.18,head_length=0.25",
                        color="#555", lw=1.1))

# -------------------------- 绘制模块 --------------------------
box(3.0, 15.0, 4.0, 0.8, "#ffffff", [r"Input: [B, 4, 3, 96, 96]"])
arr(5.0, 15.0, 5.0, 14.2)

box(2.5, 11.8, 5.0, 2.3, colors["vit"], [
    "ViTBackbone (DINO vits8 frozen)",
    "21.9M total | 197K trainable",
    "384 → 512 projection",
    "Output: [B*4, 144, 512]"
])
arr(5.0, 11.8, 5.0, 11.0)

box(2.5, 8.4, 5.0, 2.5, colors["toolinit"], [
    "ToolinitModel (原创核心)",
    "3×SerialHybridBlock",
    "MHA + 4×Mamba per block | d_model=512",
    "25.3M trainable",
    "Output: [B*4, 144, 512]"
])
arr(5.0, 8.4, 5.0, 7.6)

box(2.5, 4.7, 5.0, 2.8, colors["spatial"], [
    "SpatialSoftmax",
    "K=16 keypoints | grid=12×12",
    "Conv2d 512→16 | spatial softmax",
    "8.2K trainable",
    "Output: [B*4,16,2]  &  [B*4,16,512]"
])
arr(5.0, 4.7, 5.0, 3.9)

box(2.5, 1.8, 5.0, 2.0, colors["kp"], [
    "Keypoint Projection",
    "feat_compress: 512→32 | proj: 2176→512",
    "1.1M trainable | Output: [B, 512]"
])
arr(5.0, 1.8, 5.0, 1.0)

box(2.5, 0.2, 5.0, 0.9, colors["diff"], [
    "DiffusionActionHead | ConditionalUnet1d",
    "Output: [B, 4, 2]"
])

# 标题
ax.text(5.0, 16.2, "V4 Model Architecture Pipeline",
        ha="center", fontsize=14, weight="bold")

# 输出矢量 SVG
plt.tight_layout()
plt.savefig(
    "V4_Architecture_VECTOR.svg",
    dpi=300,
    bbox_inches="tight",
    format="svg"
)
plt.show()