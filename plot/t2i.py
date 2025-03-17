import matplotlib.pyplot as plt
import numpy as np

# --- Define Regions and Data ---
regions = ["Harassment", "Hate", "Illegal-Activity", "Self-Harm", "Sexual", "Shocking", "Violence"]  # 7 categories
data = [20, 21.13, 61, 22, 74, 56, 52]  # Corresponding values

# --- Colors (Adjust as needed) ---
colors = [
    "#E8B096",
    "#766787",
    "#45a4b4",
    "#afcfa6",
    "#dbd186",
    "#87c9c3",
    "#D67785",
]
inner_ring_color = "#f0f0f0"

# --- Plot Setup ---
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# --- Angles and Widths ---
num_regions = len(regions)
angles = np.linspace(0, 2 * np.pi, num_regions, endpoint=False)
widths = np.diff(angles)[0]
angles = np.concatenate((angles, [angles[0]]))
data = np.concatenate((data, [data[0]]))
colors = np.concatenate((colors, [colors[0]]))

# --- Plotting the Bars ---
ax.bar(angles[:-1], data[:-1], width=widths, bottom=0.0, color=colors, alpha=0.8)

# --- Optional Inner Ring ---
# data3 = [20] * len(regions)
# data3 = np.concatenate((data3, [data3[0]]))
# ax.bar(angles[:-1], data3[:-1], width=widths, bottom=0.0, color=inner_ring_color, alpha=0.4)

# --- Radial Axis ---
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20", "40", "60", "80", "100"])
ax.tick_params(axis='y', colors='black')
ax.spines['polar'].set_visible(False)
ax.grid(True, color='lightgray', linestyle='--')

# --- Region Labels and Value Labels ---
label_radius = max(data) * 1.1

# --- Customizable Font Size for Region Labels ---
region_fontsize = 15  #  Set the desired font size here

for i, (angle, region, value) in enumerate(zip(angles[:-1], regions, data[:-1])):
    # Region Label
    text_angle = angle
    if 0.5 * np.pi < angle < 1.5 * np.pi:
        text_angle += np.pi

    ax.text(
        angle,
        label_radius,
        region,
        ha="center",
        va="center",
        rotation=np.degrees(text_angle),
        fontsize=region_fontsize,  # Use the custom font size
        fontweight='bold'         # Make the font bold
    )

    # Value Label (Inside the bar, HORIZONTAL)
    value_radius = value * 0.7
    ax.text(
        angle,
        value_radius,
        str(value),
        ha="center",
        va="center",
        rotation=0,  # Set rotation to 0 for horizontal text
        fontsize=15,
        color="black",
        fontweight='bold'
    )

    # --- Circles ---
    ax.plot(angle, data[i], marker="o", color="black", markersize=6)

ax.set_xticklabels([])
plt.tight_layout()
plt.show()
plt.savefig('./janus.pdf', format='pdf', bbox_inches='tight')