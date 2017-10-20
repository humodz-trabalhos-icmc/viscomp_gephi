# Hugo Moraes Dzin 8532186

import numpy as np
import matplotlib.pyplot as plt


header = [
    'id',
    'label',
    'value',
    'degree',
    'closeness',
    'betweeness',
    'clustering',
]

# READ DATA
data = np.genfromtxt(
    skip_header=1,
    fname='gephi/polbooks.csv',
    delimiter=';',
    names=header,
    dtype='i,a100,a1' + 4*',f',
)


# Take last 4 rows as a normal 2d ndarray
# Numpy may throw a FutureWarning but it's not an issue
stats = np.array(data[header[-4:]].view((np.float32, 4)))

# Eigenvector decomposition
M = np.copy(stats)

col_mean = M.mean(axis=0)
col_std = M.std(axis=0)

default_options = np.seterr(divide='ignore')

M = (M - col_mean) / col_std
M[np.isnan(M)] = 0  # col_std may have zeros

np.seterr(**default_options)

# Covariance and eigenvector decomposition
C = np.cov(M.T, bias=1)

evalues, evectors = np.linalg.eig(C)

# Sort by decreasing order of eigenvalues
new_order = np.argsort(evalues)[::-1]
evalues = evalues[new_order]
evectors = evectors[:, new_order]

final_data = np.dot(M, evectors)
pca = final_data.T

# Divide each column by sum(abs(column)), then multiply by 100
weight_evalues = 100 * evalues / np.abs(evalues).sum()
weight_evectors = 100 * evectors / np.abs(evectors).sum(axis=0)


# PLOTS

# Plot covariance Matrix
plt.figure()
plt.matshow(C)
plt.title('Covariance Matrix')
plt.colorbar()
plt.savefig('plots/cov_matrix.png', bbox_inches='tight')
plt.close()

# Eigenvalue % plots
plt.bar(range(1, len(weight_evalues) + 1), weight_evalues)
plt.title('Explanation of Variance')
plt.xlabel('PCA Component')
plt.ylabel('Eigenvalue % of Explanation')
plt.savefig('plots/eigenvalues.png', bbox_inches='tight')
plt.close()


# PCA component plots
def plot_pca(cmp1, cmp2):
    plt.scatter(pca[cmp1], pca[cmp2])
    plt.title('PCA {} vs {}'.format(cmp1+1, cmp2+1))
    plt.xlabel('PCA Component {}'.format(cmp1+1))
    plt.ylabel('PCA Component {}'.format(cmp2+1))
    plt.savefig(
        'plots/pca_{}vs{}.png'.format(cmp1+1, cmp2+1),
        bbox_inches='tight')
    plt.close()


plot_pca(0, 1)
plot_pca(0, 2)
plot_pca(0, 3)


ax0 = 2
ax1 = 3

plt.scatter(M[:, ax0], M[:, ax1])
plt.plot([0, evectors[ax0, 0]], [0, evectors[ax1, 0]], 'r',
         label='PCA 1')
plt.plot([0, evectors[ax0, 1]], [0, evectors[ax1, 1]], 'g',
         label='PCA 2')
plt.title('PCA Component Vectors')
plt.xlabel(header[3 + ax0])
plt.ylabel(header[3 + ax1])
plt.legend()
plt.savefig(
    'plots/pca_vectors.png',
    bbox_inches='tight')
plt.close()
