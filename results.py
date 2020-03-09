from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

model = Word2Vec.load("word2vec.model")
# df = pd.read_csv("lyrics.csv")

# df["Title_Artist"] = df["Track Name"] + " by " + df["Artist Name"]

# titles = df["Title_Artist"].values.tolist()[:32]
with open("titles.pickle", "rb") as f:
    # pickle.dump(songs, f)
    titles = pickle.load(f)[:1]
    print(titles)


def cosine_distance(model, word, target_list, num):
    cosine_dict = {}
    word_list = []
    a = model[word]
    for item in target_list:
        if item != word:
            b = model[item]
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            cosine_dict[item] = cos_sim
    dist_sort = sorted(
        cosine_dict.items(), key=lambda dist: dist[1], reverse=True
    )  ## in Descedning order
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]


word_clusters = titles

embedding_clusters = [[model[word]] for word in titles]
# word_clusters = []
# embedding_clusters = []
# for word in titles:
#     embeddings = []
#     words = []
#     for similar_word, _ in cosine_distance(model, word, titles, num=1):
#         words.append(similar_word)
#         embeddings.append(model[similar_word])
#     embedding_clusters.append(embeddings)
#     word_clusters.append(words)


embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
tsne_model_en_2d = TSNE(
    perplexity=3, n_components=2, init="pca", n_iter=3500, random_state=32
)
embeddings_en_2d = np.array(
    tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))
).reshape(n, m, 2)


def tsne_plot_similar_words(
    title, labels, embedding_clusters, word_clusters, a, filename=None
):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    # print(len(labels[0]), len(embedding_clusters[0]), len(word_clusters[0]))

    for label, embeddings, words, color in zip(
        labels, embedding_clusters, word_clusters, colors
    ):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=1, label=label)
        plt.annotate(
            label,
            alpha=0.7,
            xy=(x, y),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            size=8,
        )
    #     for i, word in enumerate(words):
    #         plt.annotate(
    #             word,
    #             alpha=0.5,
    #             xy=(x[i], y[i]),
    #             xytext=(5, 2),
    #             textcoords="offset points",
    #             ha="right",
    #             va="bottom",
    #             size=8,
    #         )
    # # plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format="png", dpi=150, bbox_inches="tight")
    plt.show()


tsne_plot_similar_words(
    "Similar songs from RRG playlist",
    titles,
    embeddings_en_2d,
    word_clusters,
    0.7,
    "similar_words.png",
)


# def display_closestwords_tsnescatterplot(model, word, size):

#     arr = np.empty((0, size), dtype="f")
#     word_labels = titles

#     for word in titles:
#         close_words = model.similar_by_word(word)

#         arr = np.append(arr, np.array([model[word]]), axis=0)
#         for wrd_score in close_words:
#             wrd_vector = model[wrd_score[0]]
#             word_labels.append(wrd_score[0])
#             arr = np.append(arr, np.array([wrd_vector]), axis=0)

#     tsne = TSNE(perplexity=1, n_components=2, init="pca", n_iter=250, random_state=32)
#     # np.set_printoptions(suppress=True)
#     Y = tsne.fit_transform(arr)
#     x_coords = Y[:, 0]
#     y_coords = Y[:, 1]
#     plt.scatter(x_coords, y_coords)
#     for label, x, y in zip(word_labels, x_coords, y_coords):
#         plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
#         plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
#         plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)

#         plt.title("Similar songs from RRG playlist")
#         plt.grid(True)
#         plt.show()


# display_closestwords_tsnescatterplot(model, titles, 300)
