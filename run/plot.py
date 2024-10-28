import numpy as np
import matplotlib.pyplot as plt

# Data
levels = [1, 2, 3, 4]
psnr_values = [33.1625, 33.275, 33.37625, 33.47]

# Plotting the curve with labels and grid
plt.plot(levels, psnr_values, marker='o', linestyle='--', label='PSNR', color='blue', linewidth=2.5, alpha=0.5)

# Adding labels to each point
for i, txt in enumerate(psnr_values):
    plt.annotate(f'{txt:.2f}', (levels[i], psnr_values[i]), textcoords="offset points", xytext=(0, 8), ha='center', color='red', fontsize=12)

# Adding labels, title, and grid
plt.xlabel('Number of Levels', fontsize=12)
plt.ylabel('PSNR', fontsize=12)
# plt.title('PSNR vs. Number of Levels', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
# plt.axis('off')

# Customize the spines to create a lighter bounding box
for spine in plt.gca().spines.values():
    spine.set_edgecolor('gray')
    spine.set_linewidth(0.5)

plt.xticks(np.arange(min(levels), max(levels) + 1, 1))

# plt.box(False)

# Display the plot with legend
plt.legend()
plt.gcf().set_size_inches(8, 6)
plt.savefig('psnr.png', bbox_inches='tight', pad_inches=0.0, dpi=300)

