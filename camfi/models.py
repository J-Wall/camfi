"""Provides ``model_urls``, a ``dict`` mapping model name strings to URLs which point to
pre-trained Camfi autoannotation models.

Currently defined models:

v1
    Trained on the `2019 Cabramurra dataset <https://doi.org/10.5281/zenodo.4950570>`_
    using Camfi v1.0. Should still be compatible with Camfi v2, but this is untested.

v2
    Trained on the `2019 Cabramurra dataset <https://doi.org/10.5281/zenodo.4950570>`_
    using Camfi v2.1.3. Tested with Camfi v2, and trained for longer than v1.

release
    Can change with each release of Camfi.
    Currently set to **v1**.
"""
_release_model = "v2"
model_urls = {
    "v1": "https://github.com/J-Wall/camfi/releases/download/1.0/20210519_4_model.pth",
    "v2": "https://github.com/J-Wall/camfi/releases/download/v2.1.3/20210809_14_model.pth",
}

model_urls["release"] = model_urls[_release_model]
