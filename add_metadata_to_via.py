#!/usr/bin/env python

from datetime import datetime as dt
import json
from multiprocessing import Pool
import sys

import fire
import imageio


def get_metadata(image_file):
    """
    Parameters
    ----------
    image_file : str
        Path to image file

    Returns
    -------
    out : dict
        Contains the metadata of the image, with all keys and values coerced to str.
    """
    with imageio.get_reader(image_file) as img_reader:
        img = img_reader.get_data(0)

    out = {}
    for exif_key in img.meta["EXIF_MAIN"].keys():
        out[str(exif_key)] = str(img.meta["EXIF_MAIN"][exif_key])

    return out


def _get_metadata_with_key(img_tup):
    return img_tup[0], get_metadata(img_tup[1])


def add_metadata(processes=1):
    """
    Parameters
    ----------
    processes : int
        number of child processes to spawn
    """
    pool = Pool(processes)

    annotation_project = json.load(sys.stdin)
    file_attributes = set()

    img_paths = [
        (img_key, annotation_project["_via_img_metadata"][img_key]["filename"])
        for img_key in annotation_project["_via_img_metadata"].keys()
    ]

    for img_key, metadata_dict in pool.imap_unordered(
        _get_metadata_with_key, img_paths, 100
    ):
        annotation_project["_via_img_metadata"][img_key]["file_attributes"].update(
            metadata_dict
        )
        file_attributes.update(metadata_dict)

    pool.close()
    pool.join()

    for file_attribute in file_attributes:
        annotation_project["_via_attributes"]["file"][str(file_attribute)] = dict(
            type="text", description="", default_value=""
        )

    json.dump(annotation_project, sys.stdout, separators=(",", ":"), sort_keys=True)


def main():
    fire.Fire(add_metadata)


if __name__ == "__main__":
    main()
