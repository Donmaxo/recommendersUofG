import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys


def run(name, i):
    with open(name, 'r') as f:
        lst = f.read()

    lst = lst.split('\n')
    uh = list(json.loads(lst[0]))  # user history
    u = list(json.loads(lst[1]))   # user embedding
    ur = list(json.loads(lst[2]))  # user reldiff embeddings (length of candidate_news)
    cn = list(json.loads(lst[3]))  # candidate news embeddings
    ur_all = list(json.loads(lst[4])) # all user reldiff with candidate news with user history
    pred_rd = list(json.loads(lst[5]))  # all predictions ordered by reldiff
    pred_dot = list(json.loads(lst[6])) # all predictions ordered by dot product

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)

    all_vecs = uh
    all_vecs.extend([u, ur[i], cn[i]])

    to_plot = tsne.fit_transform(all_vecs)


    ur_plt = plt.scatter(to_plot[-2:-1, 0], to_plot[-2:-1, 1], color='r') # user reldiff embedding
    u_plt = plt.scatter(to_plot[-3:-2, 0], to_plot[-3:-2, 1], color='g')  # user dot product embedding
    cn_plt = plt.scatter(to_plot[-1:, 0], to_plot[-1:, 1], color='m')     # candidate news embedding
    uh_plt = plt.scatter(to_plot[:-3, 0], to_plot[:-3, 1], color='b')     # clicked news histories

    plt.legend((ur_plt, u_plt, cn_plt, uh_plt),
               ("User Reldiff Embedding", "User dot Embedding", "Embedding of candidate news", "Embeddings of user history"),
               bbox_to_anchor=(1.5, 1.05),
               loc='upper right',
               fancybox=True)

    plt.savefig(f'scatterplot_t-sne-u5-cn{i}.png')

    print(f"scatter.png")


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
