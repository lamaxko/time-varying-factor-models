from matplotlib import colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# New base hex values from the image

BASE_COLORS = {
    "asparagus":      "#7E9971",
    "reseda_green_1": "#6B8D64",
    "reseda_green_2": "#62875E",
    "fern_green_1":   "#5D845B",
    "fern_green_2":   "#588157",
    "hunter_green_1": "#496E4C",
    "hunter_green_2": "#426446",
    "hunter_green_3": "#3A5A40",
    "brunswick_1":    "#375441",
    "brunswick_2":    "#344E41"
}

# Generate light-to-dark gradients for each base color
def generate_color_gradient(hex_color, steps=10):
    rgb = np.array(mcolors.to_rgb(hex_color))
    white = np.ones(3)
    return [mcolors.to_hex(white * (1 - i / (steps - 1)) + rgb * (i / (steps - 1))) for i in range(steps)]

# Build gradients dictionary
GRADIENTS = {name: generate_color_gradient(color, 10) for name, color in BASE_COLORS.items()}

# Palette base as a list
PALETTE_BASE = list(BASE_COLORS.values())

# Style application function
def apply_default_style():
    plt.style.use("default")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titleweight": "bold",
        "axes.titlecolor": BASE_COLORS["hunter_green_1"],
        "axes.labelsize": 12,
        "axes.labelcolor": "#333333",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "figure.dpi": 300
    })
