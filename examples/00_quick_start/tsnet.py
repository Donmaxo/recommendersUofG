import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


with open("nrms_rd_embds.json", 'r') as f:
    lst = f.read()

lst = lst.split('\n')
uh = np.array(json.loads(lst[0]))
u = np.array(json.loads(lst[1]))
cn = np.array(json.loads(lst[2]))
uh_all = np.array(json.loads(lst[3]))

tsne1 = TSNE(n_components=2, perplexity=30, n_iter=1000)
tsne2 = TSNE(n_components=2, perplexity=30, n_iter=1000)
tsne3 = TSNE(n_components=2, perplexity=30, n_iter=1000)
uh_tsne = tsne1.fit_transform(uh)
u_tsne = tsne2.fit_transform(u[0])
cn_tsne = tsne3.fit_transform(cn[0])

fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].scatter(u_tsne[:, 0], u_tsne[:, 1], color='g')
ax[0].set_title('t-SNE user emb')
ax[1].scatter(cn_tsne[:, 0], cn_tsne[:, 1], color='b')
ax[1].set_title('t-SNE news emb')
ax[2].scatter(uh_tsne[:, 0], uh_tsne[:, 1], color='r')
ax[2].set_title('t-SNE user_hist embs')

plt.savefig('scatter.png')

print(u_tsne)
