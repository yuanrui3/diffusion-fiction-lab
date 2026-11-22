# Third-party material

The MIT licence in `LICENSE` covers the code, the notes and the data created for this project. Some files are derived from other sources, which keep their own terms:

- **Wikisource.** `data/recall/` (回目 of 红楼梦, 水浒传, 三国演义, 西游记, 儒林外史, 海上花列传), `data/dev/` (回目 of 镜花缘) and `data/lexicon/pingshui.json` (the 平水韵 character lists) are taken from Chinese Wikisource, https://zh.wikisource.org. The underlying texts are in the public domain. Wikisource's own contributions are available under CC BY-SA 4.0. Page titles and revision numbers are recorded in `data/recall/sources.json`, `data/recall/summary_sample.json` and in the metadata of `pingshui.json`.
- **Models.** No model weights are included. The code downloads them from Hugging Face, each under its own licence: LLaDA-8B-Instruct (MIT), Dream-v0-Instruct-7B (Apache-2.0), Qwen2.5-7B-Instruct (Apache-2.0), bge-m3 (MIT). The LTP tagger used for the parallelism score is free for research and personal use; commercial use needs a licence from its authors.
- **Python packages** are listed in `pyproject.toml` and `uv.lock` and are not redistributed here.
