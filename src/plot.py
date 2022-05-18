# =============================================================================
# Plot functions 
# =============================================================================
from src import utils
plt.rcParams['figure.figsize'] =(14,8)
params = {'legend.fontsize': 'x-large',
'figure.figsize': (18, 8),
'axes.labelsize': '16',
'axes.titlesize': '16',
'xtick.labelsize':'14',
'ytick.labelsize':'14',
'font.family': 'Times new roman'}
pylab.rcParams.update(params)
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
