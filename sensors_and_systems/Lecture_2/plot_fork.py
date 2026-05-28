"""
Plot the fork function (piecewise PDF)
"""

import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    if isinstance(x, np.ndarray):
        return np.array([f(xi) for xi in x])
    
    if 0 < x <= 1:
        return (3 / 5) * x
    elif 1 < x <= 3:
        return ((math.sqrt(3) * x - math.sqrt(12)) ** 2) / 5
    elif 3 < x <= 4:
        return 12 / 5 - (3 / 5) * x
    return 0.0

# Create x values for plotting
x = np.linspace(-0.5, 4.5, 1000)
y = f(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='f(x)')
plt.fill_between(x, y, alpha=0.3)
plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('Piecewise Probability Density Function', fontsize=14)
plt.xlim(-0.5, 4.5)
plt.ylim(-0.1, 0.7)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.legend(fontsize=11)

# Annotate the regions
plt.text(0.5, 0.55, 'Region 1:\nf(x) = (3/5)x\n[0 < x ≤ 1]', fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.text(2, 0.6, 'Region 2:\nf(x) = ((√3·x - √12)²)/5\n[1 < x ≤ 3]', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
plt.text(3.5, 0.35, 'Region 3:\nf(x) = 12/5 - (3/5)x\n[3 < x ≤ 4]', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

plt.tight_layout()
plt.savefig('fork_function_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plot saved as fork_function_plot.png")
