"""Where the real tokenizers and datasets live, and how a release recipe finds them.

A ``debugmodel`` uses the toy tokenizer and the few hundred C4 samples that ship
inside the torchtitan checkout. A release recipe (docs/model-release-criteria.md,
R1) needs the real HF tokenizer and real data, which are far too big to vendor,
so they live outside the repo and are located through one environment variable::

    ASCEND_TITAN_ASSETS=/opt/assets     # default; local disk, never NFS
      hf/Qwen3-0.6B/{tokenizer.json,...}
      c4/en/c4-train.00000-of-01024.json.gz

Populate it with ``scripts/fetch_assets.sh``. Nothing here downloads anything:
a recipe that silently pulled 300 MB the first time it ran would be a bad
surprise on a shared box, so a missing asset raises with the command to fix it.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ROOT = "/opt/assets"


def assets_root() -> Path:
    return Path(os.environ.get("ASCEND_TITAN_ASSETS", DEFAULT_ROOT))


def hf_assets_path(repo_name: str) -> str:
    """The local directory holding one model's HF tokenizer files.

    ``repo_name`` is the HF repo's last component, e.g. ``Qwen3-0.6B``.
    """
    path = assets_root() / "hf" / repo_name
    if not (path / "tokenizer.json").is_file():
        raise FileNotFoundError(
            f"no HF tokenizer at {path}. Fetch it with:\n"
            f"  ./scripts/fetch_assets.sh tokenizer Qwen/{repo_name}\n"
            f"(or point ASCEND_TITAN_ASSETS at a tree that already has it)"
        )
    return str(path)


def c4_shards(count: int = 2) -> list[str]:
    """Real C4 training shards, as ``load_dataset('json', data_files=...)`` inputs."""
    shards = sorted((assets_root() / "c4" / "en").glob("c4-train.*.json.gz"))
    if len(shards) < count:
        raise FileNotFoundError(
            f"need {count} C4 shards under {assets_root() / 'c4' / 'en'}, found {len(shards)}. "
            f"Fetch them with:\n  ./scripts/fetch_assets.sh c4 {count}"
        )
    return [str(p) for p in shards[:count]]


DEFAULT_SUBSET_DOCS = 50_000


def c4_subset(docs: int = DEFAULT_SUBSET_DOCS) -> str:
    """Path to a plain-json subset of real C4, building it once if needed.

    Why a subset instead of the raw shards: a shard is ~360k documents, and
    ``load_dataset("json", ...)`` is a random-access source, so **every rank**
    materialises the whole thing into its own arrow cache before step 1. On 8
    ranks that is minutes of CPU and gigabytes of cache for a run that will read
    a few tens of thousands of documents. 50k real C4 documents at 4096 context
    is ~10^8 tokens -- far more than any of our runs consume -- and loads in
    seconds.

    This is still *real* C4 text, which is what R1 asks for; only the quantity
    is cut. A genuine pretraining run would stream the full dataset.
    """
    out = assets_root() / "c4" / f"c4-subset-{docs}.json"
    if out.is_file():
        return str(out)
    import gzip
    import itertools

    shard = c4_shards(1)[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.partial")
    with gzip.open(shard, "rt", encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
        for line in itertools.islice(src, docs):
            dst.write(line)
    tmp.rename(out)
    return str(out)


def local_c4_dataset(docs: int = DEFAULT_SUBSET_DOCS):
    """A ``SingleDatasetConfig`` reading real C4 text from local disk.

    Upstream's ``DATASETS['c4']`` streams from huggingface.co, which this network
    cannot reach; the shards are fetched once through a mirror and read locally,
    which is also the more reproducible input for a benchmark.
    """
    from torchtitan.components.data import SingleDatasetConfig
    from torchtitan.components.data.sources import HuggingFaceRandomAccessSource
    from torchtitan.hf_datasets.text_datasets import TextProcessor

    return SingleDatasetConfig(
        source=HuggingFaceRandomAccessSource.Config(
            path="json",
            split="train",
            load_dataset_kwargs={"data_files": c4_subset(docs)},
        ),
        processor=TextProcessor.Config(),
        post_filters=(lambda sample: sample is not None,),
    )
