import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


with open("nrms_rd_embds.json", 'r') as f:
    lst = f.read()

lst = lst.split('\nvim')
uh = list(json.loads(lst[0]))
u = list(json.loads(lst[1]))
cn = list(json.loads(lst[2]))
#uh_all = list(json.loads(lst[3]))

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)

to_plot = tsne.fit_transform(uh.append(u[0]).append([cn[0]]))

# fig, ax = plt.subplots(nrows=1, ncols=3)
fig = plt.figure()
ax = fig.add_subplot()

ax.scatter(u_tsne[-2:-1, 0], u_tsne[-2:-1, 1], color='r')
ax.scatter(cn_tsne[-1:, 0], cn_tsne[-1:, 1], color='g')
ax.scatter(uh_tsne[:-2, 0], uh_tsne[:-2, 1], color='b')

plt.savefig('scatter.png')

print(u_tsne)
