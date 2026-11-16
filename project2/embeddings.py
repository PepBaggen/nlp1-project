"""Download and load pretrained word embeddings, keeping only the words we need."""
import shutil
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from part2 import clslib

ROOT = Path(__file__).resolve().parent.parent   # nlp1-project/, works from notebooks and from the terminal
EMB_DIR = ROOT / "project2"
PAD, UNK = "<pad>", "<unk>"

OPTIONS = {
    "glove50": dict(url="https://surfdrive.surf.nl/s/qYfZ6PiNrMLCwpS/download",
                    file="glove.2024.wikigiga.50d.zip", dims=50, skip_first=False),
    "glove300": dict(url="https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.300d.zip",
                     file="glove.2024.wikigiga.300d.zip", dims=300, skip_first=False),
    "fasttext300": dict(url="https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip",
                        file="wiki-news-300d-1M.vec.zip", dims=300, skip_first=True),
}


def download(name):
    """Download the zip once; an interrupted download leaves only a .part file."""
    opt = OPTIONS[name]
    path = EMB_DIR / opt["file"]
    if not path.is_file():
        print(f"Downloading {opt['file']} ...")
        tmp = path.with_name(path.name + ".part")
        with urlopen(opt["url"]) as response, open(tmp, "wb") as f:
            shutil.copyfileobj(response, f, length=1 << 20)
        tmp.rename(path)
    return path


def _wrap(words, vectors):
    """Put a word list + matrix in the course's PretrainedEmbeddings class (row 0 = <pad>, row 1 = <unk>)."""
    emb = clslib.PretrainedEmbeddings.__new__(clslib.PretrainedEmbeddings)
    emb.vocab = clslib.Vocabulary(init=list(words), default=UNK, frozen=True)
    emb.vectors = np.asarray(vectors, dtype=np.float32)
    emb.num_dims = emb.vectors.shape[1]
    emb.unk_token, emb.pad_token = UNK, PAD
    return emb


def load(name, keep_words, force=False):
    """
    Load embedding model `name`, keeping only the rows for `keep_words`.

    Same rules as clslib.PretrainedEmbeddings: words are lowercased, duplicates keep their first
    vector, <pad> is all zeros and <unk> is the mean of ALL vectors in the file. A model only ever
    looks up the rows of words in our texts, so the classifiers are exactly the same as with the
    full table. The small table is cached as .npz, so the next load takes seconds.
    """
    opt = OPTIONS[name]
    cache = EMB_DIR / f"{name}.isear.npz"
    if cache.is_file() and not force:
        data = np.load(cache)
        return _wrap(data["words"].tolist(), data["vectors"])

    path = download(name)
    keep = {w.lower() for w in keep_words}
    seen = {PAD, UNK}
    total, count = np.zeros(opt["dims"]), 0
    words, kept = [], []
    print(f"Reading {path.name} (a minute or two the first time) ...")
    for line_num, line in enumerate(clslib.read_lines_from_file(str(path))):
        if opt["skip_first"] and line_num == 0:
            continue
        parts = line.split()
        if len(parts) - 1 != opt["dims"]:
            continue
        word = parts[0].lower()
        if word in seen:
            continue
        seen.add(word)
        vector = np.asarray(parts[1:], dtype=np.float32)
        total += vector
        count += 1
        if word in keep:
            words.append(word)
            kept.append(vector)

    vectors = np.vstack([np.zeros(opt["dims"], dtype=np.float32), (total / count).astype(np.float32)] + kept)
    words = [PAD, UNK] + words
    np.savez(cache, words=np.array(words), vectors=vectors)
    print(f"{name}: kept {len(words) - 2} of {count} words")
    return _wrap(words, vectors)