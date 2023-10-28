import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

"""
  This is a simple but effective example program. You can check whether your data is well distributed.
"""

# Read from file
data = np.genfromtxt('seq_and_value.tsv', delimiter='\t', usecols=1)

# Draw hist
bin_edges = np.arange(0, 3, 0.02)  # The interval of the histogram is divided at 0.2 intervals
plt.hist(data, bins=bin_edges, density=True, alpha=0.6, color='b')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution')

# Save and show image
plt.savefig('Distribution.png')
plt.show()

# Check normality
k2, p = stats.normaltest(data)
alpha = 0.05

if p < alpha:
    print("The data does not follow a normal distribution,p=",p)
else:
    print("The data follow a normal distribution,p=",p)
    