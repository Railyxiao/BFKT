import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv('result.csv')

data.columns = [
    '59,1', '59,1', '59,1', '3,0', '3,1', '3,0', '3,0', '3,1', '3,1', '3,1', '3,0', '3,1', '3,1', '3,1', '108,1',
    '94,0', '94,1', '94,1', '94,1', '94,1', '94,1', '67,0', '67,0', '67,0', '67,0', '95,1', '95,0', '95,1', '95,1',
    '96,0', '96,1', '92,1', '92,1', '92,1', '92,1', '98,1'
]

data.index = [
    'c(67)',
    'c(3)',
    'c(59)',
    'c(92)',
    'c(108)',
    'c(94)',
    'c(95)',
    'c(96)',
    'c(98)',
]


sns.heatmap(data=data, cmap="YlGnBu", annot=True, linewidth=2, linecolor='white',
            cbar=True, vmax=1.0, vmin=0.0, center=0, square=True,
            robust=True,
            xticklabels=data.columns,
            annot_kws={'color': 'white', 'size': 1, 'family': None, 'style': None, 'weight': 15},
            cbar_kws={'orientation': 'vertical', 'shrink': 1, 'extend': 'max', 'location': 'right'})
plt.show()
