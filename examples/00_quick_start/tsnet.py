import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
from openTSNE import TSNE as OTSNE


def run(name, i):
    with open(name, 'r') as f:
        lst = f.read()
    u_index = name[name.index('-') + 1 : name.index('.')]
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


    uh_plt = plt.scatter(to_plot[:-3, 0], to_plot[:-3, 1], color='b')     # clicked news histories
    u_plt = plt.scatter(to_plot[-3:-2, 0], to_plot[-3:-2, 1], color='g')  # user dot product embedding
    ur_plt = plt.scatter(to_plot[-2:-1, 0], to_plot[-2:-1, 1], color='r') # user reldiff embedding
    cn_plt = plt.scatter(to_plot[-1:, 0], to_plot[-1:, 1], color='m')     # candidate news embedding

    plt.legend((ur_plt, u_plt, cn_plt, uh_plt),
               ("User Reldiff Embedding", "User Embedding", "Embedding of candidate news", "Embeddings of user history"),
               bbox_to_anchor=(1.5, 1.05),
               loc='upper right',
               fancybox=True)

    plt.savefig(f'scatterplot_t-sne-u{u_index}-cn{i}.png', bbox_inches='tight')
    plt.show()

    print("rd: ", pred_rd)
    print('dot:', pred_dot)

    print(f'scatterplot_t-sne-u{u_index}-cn{i}.png')

    ## Other graph approach
    otsne = OTSNE(
        perplexity=30,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=True
    )

    uh_embd = otsne.fit(np.array(uh))
    u_embd = uh_embd.transform(np.array(u))
    ur_embd = uh_embd.transform(np.array(ur[i]))
    cn_embd = uh_embd.transform(np.array(cn[i]))

    fig, ax = plt.sublots(figsize=(8,8))
    utils.plot(uh_embd, uh, colors=utils.MACOSKO_COLORS)
    plt.savefig('scatter.png')



if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
