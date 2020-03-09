from gensim.models import Word2Vec
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk.corpus
import multiprocessing as mp
import tqdm
import numpy as np

lmtzr = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))


def clean(song):
    lyric, title = song
    try:
        cleaned_lyric = [
            lmtzr.lemmatize(word)
            for word in lyric
            if ((word != "") and word not in stop_words)
        ]
        cleaned_lyric.append(title)
        return cleaned_lyric, title
    except:
        return None, None


if __name__ == "__main__":

    df = pd.read_csv("lyrics.csv")
    df.dropna(subset=["lyrics"], inplace=True)
    df = df[df["lyrics"] != "[Instrumental]"]
    # df.sort_values("year", inplace=True, ascending=True)

    df["Title_Artist"] = df["song"].str.lower() + " by " + df["artist"].str.lower()

    df["Lyrics_List"] = (
        df["lyrics"]
        .str.replace("\n", " ")
        .str.replace("'", "_")
        .str.replace(r"[\W]", " ")
        .str.lower()
        .str.split(" ")
    )

    lyrics = df["Lyrics_List"].values.tolist()
    titles = df["Title_Artist"].values.tolist()

    del df

    pool = mp.Pool()
    songs = []
    valid_titles = []
    for result in tqdm.tqdm(
        pool.imap_unordered(clean, zip(lyrics, titles)), total=len(titles)
    ):
        if not result[0]:
            continue
        else:
            valid_titles.append(result[1])
            songs.append(result[0])
    pool.close()
    pool.join()

    pd.DataFrame(valid_titles).to_csv("titles.csv")

    model = Word2Vec(songs, min_count=1, size=300, workers=8, window=3)
    model.save("word2vec.model")

