#!/usr/bin/env python

from datetime import datetime as dt
import json
from math import inf
from multiprocessing import Pool
import sys
from tqdm import tqdm
from zipfile import ZipFile

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


class AnnotationUtils(object):
    def __init__(self, processes=1, i=None, o=None):
        """
        Parameters
        ----------
        processes : int
            number of child processes to spawn
        o: str
            output file. Defaults to sys.stdout
        """
        self.processes = processes
        self.i = i
        self.o = o

    def _output(self, out_str):
        if self.o is None:
            print(out_str)
        else:
            with open(self.o, "w") as f:
                print(out_str, file=f)

    def _load(self):
        if self.i is None:
            annotations = json.load(sys.stdin)
        else:
            with open(self.i, "r") as jf:
                annotations = json.load(jf)
        return annotations

    def add_metadata(self):
        """
        Adds image (EXIF) metadata to VIA project.

        Usage: `cat <via_project_file.json> | annotation_utils.py add_metadata > <out>`
        """
        pool = Pool(self.processes)

        annotation_project = self._load()
        file_attributes = set()

        img_paths = [
            (img_key, annotation_project["_via_img_metadata"][img_key]["filename"])
            for img_key in annotation_project["_via_img_metadata"].keys()
        ]

        for img_key, metadata_dict in tqdm(
            pool.imap_unordered(_get_metadata_with_key, img_paths, 10),
            total=len(img_paths),
            desc="Adding metadata",
            unit="img",
            smoothing=0.0,
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

        self._output(
            json.dumps(annotation_project, separators=(",", ":"), sort_keys=True)
        )

    def remove_unannotated(self):
        """
        Removes image metadata from VIA project file for images which have no
        annotations.
        """
        annotation_project = self._load()
        new_metadata = {}

        for key, img_data in annotation_project["_via_img_metadata"].items():
            if len(img_data["regions"]) > 0:
                new_metadata[key] = img_data

        annotation_project["_via_img_metadata"] = new_metadata

        self._output(
            json.dumps(annotation_project, separators=(",", ":"), sort_keys=True)
        )

    def merge_annotations(self, *annotation_files):
        """
        Takes a list of VIA project files and merges them into one.
        """
        with open(annotation_files[0], "r") as jf:
            annotations = json.load(jf)

        for f in annotation_files[1:]:
            with open(f, "r") as jf:
                next_anns = json.load(jf)
            annotations["_via_img_metadata"].update(next_anns["_via_img_metadata"])

        self._output(json.dumps(annotations, separators=(",", ":"), sort_keys=True))

    def zip_images(self, **kwargs):
        """
        Makes a zip archive of all the images in the provided VIA project file.
        If --i is set, then the annotation file itself will be included in the zip file.
        kwargs are passed to zipfile.ZipFile
        """
        assert self.o is not None, "Must specify output zipfile using --o=[file]"

        annotations = self._load()

        with ZipFile(self.o, mode="w", **kwargs) as outzip:
            if self.i is not None:
                outzip.write(self.i)
            for img_data in tqdm(
                annotations["_via_img_metadata"].values(),
                desc="Zipping images",
                unit="img",
                smoothing=0.1,
            ):
                outzip.write(img_data["filename"])

    def filter(self, by, minimum=-inf, maximum=inf, mode="warn"):
        """
        Filters VIA annotations by enforcing a minimum and/or maximum value for a
        numerical region attribute (eg. "score" which is defined during automatic
        automatic annotation)

        Parameters
        ----------

        by : str
            The region_attributes key to filter annotations by.

        minimum : float
            The minimum value of the region attribute to pass the filter

        maximum : float
            The maximum value of the region attribute to pass the filter

        mode : str
            One of {"pass", "fail", "raise", "warn"}. Defines how annotations missing
            the `by` region attribute are handled.
                "pass": These annotations pass the filter
                "fail": These annotations are removed
                "raise": A KeyError is raised if an annotation is missing the attribute
                "warn": Like "pass" but a warning is printed to sys.stderr
        """

        def _raise(ex):
            raise ex

        def _warn(ex):
            print(
                f"Warning: Missing region_attribute '{by}' for {img_data['filename']}",
                file=sys.stderr,
            )
            return True

        mode_fn = {
            "pass": lambda x: True,
            "fail": lambda x: False,
            "raise": _raise,
            "warn": _warn,
        }[mode]

        def _annotation_passes(region):
            try:
                return minimum < float(region["region_attributes"][by]) < maximum
            except KeyError as ex:
                return mode_fn(ex)

        annotation_project = self._load()

        for img_data in annotation_project["_via_img_metadata"].values():
            img_data["regions"] = list(filter(_annotation_passes, img_data["regions"]))

        self._output(
            json.dumps(annotation_project, separators=(",", ":"), sort_keys=True)
        )


def main():
    fire.Fire(AnnotationUtils)


if __name__ == "__main__":
    main()
