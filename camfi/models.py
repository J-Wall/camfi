"""Provides ``model_urls``, a ``dict`` mapping model name strings to URLs which point to
pre-trained Camfi autoannotation models.

Currently defined models:

v1
    Trained on the `2019 Cabramurra dataset <https://doi.org/10.5281/zenodo.4950570>`_
    using Camfi v1.0. Should still be compatible with Camfi v2, but this is untested.

v2
    Trained on the `2019 Cabramurra dataset <https://doi.org/10.5281/zenodo.4950570>`_
    using Camfi v2.1.3. Tested with Camfi v2, and trained for longer than v1.

v2.1.4
    Trained on the following datasets using Camfi v2.1.4 on the following datasets:
    `2019 Cabramurra dataset <https://doi.org/10.5281/zenodo.4950570>`_,
    `2019-2020 Mt Kosciuszko dataset <https://doi.org/10.5281/zenodo.5039891>`_,
    `2020-2021 Mt Kosciuszko dataset <https://doi.org/10.5281/zenodo.5040011>`_,
    `2019-2020 Ken Green Bogong dataset <https://doi.org/10.5281/zenodo.4971714>`_,
    `2020-2021 Ken Green Bogong dataset <https://doi.org/10.5281/zenodo.4972022>`_,
    and
    `2020-2021 Mt Gingera dataset <https://doi.org/10.5281/zenodo.5040018>`_.

release
    Can change with each release of Camfi.
    Currently set to **v2.1.4**.
"""
_release_model = "v2.1.4"
model_urls = {
    "v1": "https://github.com/J-Wall/camfi/releases/download/1.0/20210519_4_model.pth",
    "v2": "https://github.com/J-Wall/camfi/releases/download/v2.1.3/20210809_14_model.pth",
    "v2.1.4": "https://github.com/J-Wall/camfi/releases/download/v2.1.4/20210823_14_model.pth",
}

model_urls["release"] = model_urls[_release_model]
