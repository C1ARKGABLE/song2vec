# $start
import pandas as pd
import lyricsgenius
import os
from pandarallel import pandarallel


def query_songs(stuff):
    try:
        return genius.search_song(
            stuff["Track Name"].split("-")[0], stuff["Artist Name"]
        ).lyrics
    except:
        return None


if __name__ == "__main__":
    pandarallel.initialize()

    df = pd.read_csv("red_river_gorge.csv")

    df = pd.concat([df["Track Name"], df["Artist Name"]], axis=1)

    genius = lyricsgenius.Genius("KEY")

    genius.verbose = True
    genius.remove_section_headers = True
    genius.skip_non_songs = False

    df["Lyrics"] = df.parallel_apply(query_songs, axis=1)
    df.dropna(subset=["Lyrics"])

    df.to_csv("lyrics_RRG.csv")

