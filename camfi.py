#!/usr/bin/env python

from datetime import datetime as dt
from functools import wraps
import itertools
import json
from math import inf, sqrt
from multiprocessing import Pool
import os
import random
import sys
from urllib.parse import urlparse
from urllib.request import urlopen, urlretrieve
from zipfile import ZipFile

import exif
import fire
import imageio
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import scipy as sp
from shapely.geometry import LineString
from skimage import draw, transform
import torch
from tqdm import tqdm, trange

import camfiutils as utils

RLS_MODEL = "https://github.com/J-Wall/camfi/releases/download/1.0/20210519_4_model.pth


class DefaultDict(dict):
    def __init__(self, default, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default = default

    def __missing__(self, key):
        self[key] = self.default
        return self.default

    def __repr__(self):
        return f"{str(type(self)).split('.')[-1][:-2]}({self.default}, {super().__repr__()})"


class UniqueDict(DefaultDict):
    def __missing__(self, key):
        while self.default in self.values():
            self.default += 1
        self[key] = self.default
        self.default += 1
        return self[key]


class CamfiDataset:
    def __init__(
        self,
        root,
        transforms,
        *via_project_files,
        crop=None,
        point_r=10,
        min_annotations=0,
        max_annotations=np.inf,
        inference_mode=False,
        labels=DefaultDict(1),
        mask_dilate=5,
        exclude=None,
    ):
        self.root = root
        self.transforms = transforms
        self.crop = crop
        self.point_r = point_r
        self.inference_mode = inference_mode
        self.labels = labels
        self.mask_dilate = mask_dilate

        self.imgs = []
        self.annotation_data = []
        self.img_keys = []

        if exclude is None:
            exclude = set()

        for f in via_project_files:
            with open(f, "r") as jf:
                via_annotations = json.load(jf)["_via_img_metadata"]

            for img_key, img_data in via_annotations.items():
                # Only load images with annotations (but not too many)
                if (min_annotations < len(img_data["regions"]) < max_annotations) and (
                    img_data["filename"] not in exclude
                ):
                    self.imgs.append(img_data["filename"])
                    self.annotation_data.append(img_data["regions"])
                    self.img_keys.append(img_key)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        if self.crop is not None:
            img = img.crop(box=self.crop)

        n_objects = len(self.annotation_data[idx])

        target = {}
        if not self.inference_mode:
            boxes = self.get_boxes(idx, img.size[::-1])
            labels = self.get_labels(idx)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((n_objects,), dtype=torch.int64)
            masks = self.get_masks(idx, img.size[::-1])
            # keypoints = self.get_keypoints(idx)

            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["masks"] = masks
            # target["keypoints"] = keypoints

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_keypoints(self, idx):
        raise NotImplementedError

    def get_masks(self, idx, img_shape):
        n_objects = len(self.annotation_data[idx])
        masks = np.zeros((n_objects, img_shape[0], img_shape[1]), dtype="u1")

        for region_i, region in enumerate(self.annotation_data[idx]):
            if region["shape_attributes"]["name"] == "polyline":
                x = region["shape_attributes"]["all_points_x"]
                y = region["shape_attributes"]["all_points_y"]
                for point_i in range(len(x) - 1):
                    rr, cc = draw.line(
                        y[point_i], x[point_i], y[point_i + 1], x[point_i + 1]
                    )
                    rr, cc = dilate_idx(rr, cc, self.mask_dilate, img_shape=img_shape)
                    masks[region_i, rr, cc] = 1

            elif region["shape_attributes"]["name"] in {"circle", "point"}:
                cx = region["shape_attributes"]["cx"]
                cy = region["shape_attributes"]["cy"]
                rr, cc = dilate_idx(cy, cx, self.mask_dilate, img_shape=img_shape)
                masks[region_i, rr, cc] = 1

            else:
                raise NotImplementedError

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        return masks

    def get_boxes(self, idx, img_shape):
        boxes = []
        for region in self.annotation_data[idx]:
            if region["shape_attributes"]["name"] == "polyline":
                x = region["shape_attributes"]["all_points_x"]
                y = region["shape_attributes"]["all_points_y"]
                r = self.point_r
                x0 = max(min(x) - r, 0)
                y0 = max(min(y) - r, 0)
                x1 = min(max(x) + r, img_shape[1])
                y1 = min(max(y) + r, img_shape[0])
                boxes.append([x0, y0, x1, y1])

            elif region["shape_attributes"]["name"] == "circle":
                cx = region["shape_attributes"]["cx"]
                cy = region["shape_attributes"]["cy"]
                r = region["shape_attributes"]["r"]
                x0 = max(cx - r, 0)
                x1 = min(cx + r, img_shape[1])
                y0 = max(cy - r, 0)
                y1 = min(cy + r, img_shape[0])
                boxes.append([x0, y0, x1, y1])

            elif region["shape_attributes"]["name"] == "point":
                cx = region["shape_attributes"]["cx"]
                cy = region["shape_attributes"]["cy"]
                r = self.point_r
                x0 = max(cx - r, 0)
                x1 = min(cx + r, img_shape[1])
                y0 = max(cy - r, 0)
                y1 = min(cy + r, img_shape[0])
                boxes.append([x0, y0, x1, y1])

            else:
                raise NotImplementedError

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        return boxes

    def get_labels(self, idx):
        labels = []
        for region in self.annotation_data[idx]:
            labels.append(self.labels[region["shape_attributes"]["name"]])

        labels = torch.as_tensor(labels, dtype=torch.int64)
        return labels


def dilate_idx(rr, cc, d, img_shape=None):
    """
    Parameters
    ----------

    rr : array or int
        row indices
    cc : array or int
        column indices (must have same shape as rr)
    d : int
        dilation factor
    img_shape : (rows, cols)
        shape of image (indices which lie outside this will be ommitted)

    Returns
    -------

    rr_dilated : array
    cc_dilated : array
    """
    d2 = d * d
    offset_r, offset_c = zip(
        *itertools.filterfalse(
            lambda x: x[0] ** 2 + x[1] ** 2 > d2,
            itertools.product(range(-d, d + 1), repeat=2),
        )
    )
    rr_dilated = np.stack([rr + i for i in offset_r]).ravel()
    cc_dilated = np.stack([cc + i for i in offset_c]).ravel()
    mask = rr_dilated >= 0
    mask[rr_dilated >= img_shape[0]] = False
    mask[cc_dilated < 0] = False
    mask[cc_dilated >= img_shape[1]] = False

    return rr_dilated[mask], cc_dilated[mask]


def train_model(
    *via_projects,
    load_pretrained_model=None,
    img_dir=None,
    crop=None,
    point_r=10,
    mask_dilate=5,
    min_annotations=0,
    max_annotations=np.inf,
    exclude=None,
    device="cpu",
    num_classes=2,
    batch_size=5,
    num_workers=2,
    num_epochs=5,
    outdir=".",
    model_name=None,
    save_intermediate=False,
):
    """
    Parameters
    ----------

    via_projects: list of paths
        Path(s) to VIA project files with annotated motion blurs for training on

    load_pretrained_model: str
        Path to model parameters file. If set, will load the pretrained parameters

    img_dir
        Path to direcotry containing images. By default inferred from first element in
        via_projects

    crop: x0,y0,x1,y1
        Crop images before processing. By default no crop. Original camfi data uses
        --crop=0,0,4608,3312

    point_r: int
        Margin added to the coordinates of annotations to determine the bounding box of
        the annotation

    mask_dilate: int
        Radius of dilation to apply to training masks

    min_annotations: int
        Skip images which have fewer than min_annotations annotations. E.g. to only
        train on images with at least one annotation set `min_annotations=1`

    max_annotations: int
        Skip images which have more than max_annotations annotations. Set this if you
        are running into memory issues when training on a GPU.

    exclude: str
        Path to file containing list of images to exclude (one per line). E.g. to
        exclude a test set

    device: str
        E.g. "cpu" or "cuda"

    num_classes: int
        Number of target classes (including background)

    batch_size: int
        Number of images to load at once

    num_workers: int
        Number of worker processes for data loader to spawn

    num_epochs: int
        Number of epochs to train

    outdir: str
        Path to directory where to save model(s)

    model_name: str
        Identifier to include in model save file. By default the current date in
        YYYYmmdd format

    save_intermediate: bool
        If True, model is saved after each epoch
    """
    # Set params
    device = torch.device(device)
    if img_dir is None:
        img_dir = os.path.dirname(via_projects[0])
    if model_name is None:
        model_name = dt.now().strftime("%Y%m%d")
    if exclude is not None:
        with open(exclude, "r") as f:
            exclude_set = set(line.strip() for line in f)
    else:
        exclude_set = set()

    # Define dataset and data loader
    dataset = CamfiDataset(
        img_dir,
        utils.get_transform(train=True),
        *via_projects,
        crop=crop,
        point_r=point_r,
        mask_dilate=mask_dilate,
        min_annotations=min_annotations,
        max_annotations=max_annotations,
        exclude=exclude_set,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=utils.collate_fn,
    )

    # Initialise model
    model = utils.get_model_instance_segmentation(num_classes)
    if load_pretrained_model is not None:
        model.load_state_dict(torch.load(load_pretrained_model))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        utils.train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=10
        )
        # update the learning rate
        lr_scheduler.step()

        if save_intermediate or epoch == num_epochs - 1:
            save_path = os.path.join(outdir, f"{model_name}_{epoch}_model.pth")
            torch.save(model.state_dict(), save_path)

    print(f"Training complete. Model saved at {save_path}")


def _sec_trivial_0(points):
    return 0.0, 0.0, 0.0


def _sec_trivial_1(points):
    return points[0][0], points[0][1], 0.0


def _sec_trivial_2(points):
    return (
        0.5 * (points[0][0] + points[1][0]),
        0.5 * (points[0][1] + points[1][1]),
        0.5
        * sqrt((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2),
    )


def _sec_trivial_3(points):
    (x1, y1), (x2, y2), (x3, y3) = points
    A = np.array([[x3 - x1, y3 - y1], [x3 - x2, y3 - y2]])
    Y = np.array(
        [
            (x3 ** 2 + y3 ** 2 - x1 ** 2 - y1 ** 2),
            (x3 ** 2 + y3 ** 2 - x2 ** 2 - y2 ** 2),
        ]
    )
    if np.linalg.det(A) == 0:
        return _sec_trivial_2((min(points), max(points)))
    Ainv = np.linalg.inv(A)
    X = 0.5 * np.dot(Ainv, Y)
    return X[0], X[1], sqrt((X[0] - x1) ** 2 + (X[1] - y1) ** 2)


def _sec_trivial(points):
    return [_sec_trivial_0, _sec_trivial_1, _sec_trivial_2, _sec_trivial_3][
        len(points)
    ](points)


def smallest_enclosing_circle(points):
    """
    Performs Welzl's algorithm to find the smallest enclosing circle of a set of points
    in a cartesian plane.

    Parameters
    ----------

    points : sequence of 2-tuples or (N, 2)-array

    Returns
    -------

    x, y, r
    """

    def welzl(P, R):
        if len(P) == 0 or len(R) == 3:
            return _sec_trivial(R)
        p = P[0]
        D = welzl(P[1:], R)
        if sqrt((D[0] - p[0]) ** 2 + (D[1] - p[1]) ** 2) <= D[1]:
            return D

        return welzl(P[1:], R + (p,))

    P = [tuple(p) for p in points]
    random.shuffle(P)
    return welzl(tuple(P), ())


class Annotator:
    """
    Parameters
    ----------

    via_project
        Path to via project json file

    model
        Either a path to state dict file which defines the segmentation model, or a url
        pointing to a model to download from the internet, or "release" or "latest".
        See `camfi download-model --help` for more information.

    num_classes : int
        Number of classes in the model. Must correspond with how model was trained

    img_dir
        Path to direcotry containing images. By default inferred from via_project

    crop : x0,y0,x1,y1
        Crop images before processing. By default no crop. Original camfi data uses
        --crop=0,0,4608,3312

    device : str
        Specifies device to run inference on. Set to cuda to use gpu.

    backup_device : str
        Specifies device to run inference on when a runtime error occurs while using
        device. Probably only makes sense to set this to cpu if device=cuda

    split_angle : float
        Approximate maximum angle between polyline segments in degrees.

    poly_order : int
        Order of polynomial used for fitting motion blur paths.

    endpoint_method : str[,method_argument,...]
        Method to find endpoints of motion blurs. Currently implemented:
            --endpoint_method=truncate,n  (where n is a positive int)
            --endpoint_method=quantile,q  (where q is a float between 0. and 1.)

    score_thresh : float
        Score threshold between 0. and 1. for annotations

    overlap_thresh : float
        Minimum proportion of overlap between two instance segmentation masks to
        infer that one of the masks should be discarded

    edge_thresh : int
        Minimum distance an annotation has to be from the edge of the image before it is
        converted from polyline to circle

    o : str
        Path to output file. Default is to output to stdout
    """

    def __init__(
        self,
        via_project,
        model="release",
        num_classes=2,
        img_dir=None,
        crop=None,
        device="cpu",
        backup_device=None,
        split_angle=15.0,
        poly_order=2,
        endpoint_method=("truncate", 10),
        score_thresh=0.4,
        overlap_thresh=0.4,
        edge_thresh=10,
        o=None,
    ):
        print(f"Loading model: {model}", file=sys.stderr)
        self.model = utils.get_model_instance_segmentation(
            num_classes, pretrained=False
        )
        model_path = AnnotationUtils().download_model(model=model)
        self.model.load_state_dict(torch.load(model_path))

        self.via_project = via_project

        if img_dir is None:
            self.img_dir = os.path.dirname(self.via_project)
        else:
            self.img_dir = img_dir

        print(f"Loading dataset: {self.via_project}", file=sys.stderr)
        self.dataset = CamfiDataset(
            self.img_dir,
            utils.get_transform(train=False),
            self.via_project,
            crop=crop,
            min_annotations=-1,
            inference_mode=True,
        )

        print(f"Putting model on device: {device}", file=sys.stderr)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        if backup_device is not None:
            print(f"Loading backup model on device: {backup_device}", file=sys.stderr)
            self.backup = True
            self.backup_model = utils.get_model_instance_segmentation(
                num_classes, pretrained=False
            )
            self.backup_model.load_state_dict(torch.load(model_path))
            self.backup_device = torch.device(backup_device)
            self.backup_model.to(self.backup_device)
            self.backup_model.eval()
        else:
            self.backup = False

        self.split_angle = split_angle * np.pi / 180.0
        self.poly_order = poly_order

        self._set_endpoint_method(endpoint_method)

        self.score_thresh = score_thresh
        self.overlap_thresh = overlap_thresh
        self.edge_thresh = edge_thresh

        self.o = o
        print("Initialisation complete", file=sys.stderr)

    def _output(self, out_str):
        if self.o is None:
            print(out_str)
        else:
            with open(self.o, "w") as f:
                print(out_str, file=f)

    def _set_endpoint_method(self, endpoint_method):
        if isinstance(endpoint_method, tuple):
            self.getendpoints = lambda x: self.__getattribute__(
                f"_endpoint_{endpoint_method[0]}"
            )(x, *endpoint_method[1:])
        else:
            self.getendpoints = self.__getattribute__(f"_endpoint_{endpoint_method[0]}")

    def _endpoint_quantile(self, fit_mask_vals, q):
        return np.where(fit_mask_vals >= fit_mask_vals.quantile(q))[0][[0, -1]]

    def _endpoint_truncate(self, fit_mask_vals, n):
        return np.array([n, len(fit_mask_vals) - n])

    def get_prediction(self, img_idx):
        img, _ = self.dataset[img_idx]
        with torch.no_grad():
            try:
                prediction = self.model([img.to(self.device)])[0]
            except RuntimeError:
                if self.backup:
                    prediction = self.backup_model([img.to(self.backup_device)])[0]
                else:
                    raise

        return {key: val.cpu() for key, val in prediction.items()}

    def filter_annotations(self, prediction):
        """
        Applies self.score_thresh and self.overlap_thresh to filter out poor quality
        annotations.

        Parameters
        ----------

        prediction : dict
            Output of model inference. Must have keys {"boxes", "scores", "masks"}

        Returns
        -------

        dict
            Filtered prediction with same keys as prediction
        """
        # Remove predictions with below-threshold score
        keep = np.where(prediction["scores"] >= self.score_thresh)[0]
        prediction = {key: val[keep, ...] for key, val in prediction.items()}
        n_predictions = len(prediction["scores"])

        if n_predictions == 0:
            return {
                "boxes": torch.tensor([]),
                "labels": torch.tensor([]),
                "scores": torch.tensor([]),
                "masks": torch.tensor([]),
            }

        # Calculate mask overlaps for all pairs of predicted instances
        mask_overlaps = np.zeros((n_predictions, n_predictions), dtype="f4")

        for i, j in filter(
            lambda x: bb_overlap(x[0], x[1], prediction["boxes"]),
            itertools.combinations(range(n_predictions), 2),
        ):
            mask_overlaps[i, j] = torch.minimum(
                prediction["masks"][i, ...], prediction["masks"][j, ...]
            ).sum() / min(
                prediction["masks"][i, ...].sum(), prediction["masks"][j, ...].sum()
            )
            mask_overlaps[j, i] = mask_overlaps[i, j]

        # Remove worst overlapping instances until there are no above-threshold overlaps
        keep = set(range(n_predictions))

        overlap_mask = mask_overlaps.max(axis=1) >= self.overlap_thresh
        while np.any(overlap_mask):
            # Figure out which overlapping annotation has the worst score
            overlap_annotations = np.where(overlap_mask)[0]
            to_discard = overlap_annotations[
                np.argmin(prediction["scores"][overlap_annotations])
            ]
            # Remove the annotation
            keep.remove(to_discard)
            mask_overlaps[to_discard, :] = 0.0
            mask_overlaps[:, to_discard] = 0.0

            overlap_mask = mask_overlaps.max(axis=1) >= self.overlap_thresh

        keep = np.array(list(keep))
        return {key: val[keep] for key, val in prediction.items()}

    def fit_poly(self, box, mask):
        """
        Uses polynomial regression to fit a polyline annotation to the provided
        segmentation mask.

        Parameters
        ----------

        box : tensor or array [x0, y0, x1, y1]
            Coordinates of bounding box corners (x0, y0) and (x1, y1). Note that y
            refers to row and x refers to column.

        mask : tensor or array
            Segmentation mask of instance with shape (image_width, image_height)

        Returns
        -------

        all_points_x : list
            x-coordinates defining polyline

        all_points_y : list
            y-coordinates defining polyline. Will have same length as all_points_x
        """
        x0, y0, x1, y1 = [int(e) for e in box]
        portrait = y1 - y0 > x1 - x0
        crop_mask = mask[y0:y1, x0:x1]

        y, x = np.where(crop_mask > 0.0)
        weights = np.array(crop_mask[y, x]).flatten()

        # Set longest axis as independent variable and fit polynomial
        ind = (x, y)[portrait]
        dep = (y, x)[portrait]
        poly_fit = np.polynomial.Polynomial.fit(ind, dep, self.poly_order, w=weights)

        # Find endpoints
        ind_vals = np.arange(crop_mask.shape[not portrait])
        dep_vals = poly_fit(ind_vals)
        val_mask = np.logical_and(dep_vals < crop_mask.shape[portrait], dep_vals >= 0)
        y_vals = (dep_vals, ind_vals)[portrait][val_mask]
        x_vals = (ind_vals, dep_vals)[portrait][val_mask]
        fit_mask_vals = crop_mask[y_vals, x_vals]

        endpoints = ind_vals[self.getendpoints(fit_mask_vals)]

        # Approximate polynomial segment with polyline
        end_gradients = poly_fit.deriv()(endpoints)
        end_angles = np.arctan(end_gradients)
        angle_diff = abs(end_angles[1] - end_angles[0])
        all_points_ind, all_points_dep = poly_fit.linspace(
            n=int(np.ceil(angle_diff / self.split_angle) + 2), domain=endpoints
        )
        all_points_x = list((all_points_ind, all_points_dep)[portrait] + x0)
        all_points_y = list((all_points_dep, all_points_ind)[portrait] + y0)

        return all_points_x, all_points_y

    def convert_to_circle(self, all_points_x, all_points_y, img_shape):
        min_x = min(all_points_x)
        max_x = max(all_points_x)
        min_y = min(all_points_y)
        max_y = max(all_points_y)

        if (
            min_x < self.edge_thresh
            or min_y < self.edge_thresh
            or max_x > img_shape[1] - self.edge_thresh
            or max_y > img_shape[1] - self.edge_thresh
        ):
            cx, cy, r = smallest_enclosing_circle(zip(all_points_x, all_points_y))
            return cx, cy, r

        else:
            return False

    def annotate_img(self, img_idx):
        """
        Calls self.get_prediction, self.filter_annotations, and self.fit_poly to
        produce annotations for an image specified with img_idx.

        Parameters
        ----------

        img_idx : int
            Index of image in via project

        Returns
        -------

        regions : list
            List of annotations in via format
        """
        prediction = self.get_prediction(img_idx)
        prediction = self.filter_annotations(prediction)

        regions = []

        for i in range(len(prediction["scores"])):
            box = prediction["boxes"][i, :]
            mask = prediction["masks"][i, 0, ...]
            score = prediction["scores"][i]
            all_points_x, all_points_y = self.fit_poly(box, mask)
            cx_cy_r = self.convert_to_circle(all_points_x, all_points_y, mask.shape)
            if cx_cy_r:
                cx, cy, r = cx_cy_r
                regions.append(
                    {
                        "region_attributes": {"score": float(score)},
                        "shape_attributes": {
                            "cx": cx,
                            "cy": cy,
                            "r": r,
                            "name": "circle",
                        },
                    }
                )
            else:
                regions.append(
                    {
                        "region_attributes": {"score": float(score)},
                        "shape_attributes": {
                            "all_points_x": all_points_x,
                            "all_points_y": all_points_y,
                            "name": "polyline",
                        },
                    }
                )

        return regions

    def annotate(self):
        """
        Calls self.annotate on all images and outputs a VIA project json file, specified
        with the --output program parameter.
        Copies the contents of self.via_project, and just replaces region information.
        """
        with open(self.via_project, "r") as f:
            project = json.load(f)

        tot_annotations = 0

        pb = trange(len(self.dataset), desc="Annotating images", unit="img")
        for img_idx in pb:
            img_key = self.dataset.img_keys[img_idx]
            regions = self.annotate_img(img_idx)
            project["_via_img_metadata"][img_key]["regions"] = regions
            tot_annotations += len(regions)
            pb.set_postfix({"tot_annotations": tot_annotations}, refresh=False)

        print(f"Saving annotations to: {self.o}", file=sys.stderr)
        self._output(json.dumps(project, separators=(",", ":"), sort_keys=True))


def bb_overlap(i, j, boxes):
    return (
        boxes[i, 0] < boxes[j, 2]
        and boxes[i, 1] < boxes[j, 3]
        and boxes[i, 2] > boxes[j, 0]
        and boxes[i, 3] > boxes[j, 1]
    )


@wraps(Annotator, updated=())
def annotate(*args, **kwargs):
    annotator = Annotator(*args, **kwargs)
    annotator.annotate()


def extract_rois(regions, image_path, scan_distance):
    """
    Extracts regions of interest (ROIs) from an image using polyline annotations

    Parameters
    ----------

    regions : dict
        contains information fopr each annotation

    image_path : str
        path to annotated image

    scan_distance : int
        half-width of rois for motion blurs

    Returns
    -------

    capture_time : datetime
        image capture from image metadata

    exposure_time : float
       exposure time of image in seconds

    blurs : list of tuples [(roi, y_diff), ...]
        roi: rotated and cropped regions of interest
        y_diff: number of rows the blur spans
    """
    # Load image and get exposure time from metadata
    reader = imageio.get_reader(image_path)
    img = reader.get_data(0)
    reader.close()
    capture_time = dt.strptime(
        img.meta["EXIF_MAIN"]["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S"
    )
    exposure_meta = img.meta["EXIF_MAIN"]["ExposureTime"]
    if isinstance(exposure_meta, tuple):
        exposure_time = exposure_meta[0] / exposure_meta[1]
    else:
        exposure_time = float(img.meta["EXIF_MAIN"]["ExposureTime"])

    #
    xs, ys = [], []
    for a in range(len(regions)):
        if regions[a]["shape_attributes"]["name"] == "polyline":
            x = regions[a]["shape_attributes"]["all_points_x"]
            y = regions[a]["shape_attributes"]["all_points_y"]
            xs.append(x)
            ys.append(y)

    blurs = []
    for x, y in zip(xs, ys):
        sections = []
        tot_length = 0
        for i in range(len(x) - 1):
            # Calculate angle of section
            try:
                perp_grad = (x[i] - x[i + 1]) / (y[i + 1] - y[i])
            except ZeroDivisionError:
                perp_grad = np.inf
            rotation = np.arctan2(y[i + 1] - y[i], x[i + 1] - x[i])

            # Calculate upper corner of ROI
            if rotation > 0:
                trans_x = x[i] + scan_distance / np.sqrt(perp_grad ** 2 + 1)
                if perp_grad == np.inf:
                    trans_y = y[i] + scan_distance
                else:
                    trans_y = y[i] + scan_distance * perp_grad / np.sqrt(
                        perp_grad ** 2 + 1
                    )
            else:
                trans_x = x[i] - scan_distance / np.sqrt(perp_grad ** 2 + 1)
                if perp_grad == np.inf:
                    trans_y = y[i] - scan_distance
                else:
                    trans_y = y[i] - scan_distance * perp_grad / np.sqrt(
                        perp_grad ** 2 + 1
                    )

            # Rotate and translate image
            transform_matrix = transform.EuclideanTransform(
                rotation=rotation, translation=(trans_x, trans_y)
            )
            warped_img = transform.warp(img, transform_matrix)

            # Crop rotated image to ROI
            section_length = int(
                round(np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2))
            )
            cropped_img = warped_img[: 2 * scan_distance, :section_length, :]

            sections.append(cropped_img)
            tot_length += section_length

        # Join sections to form complete ROI
        joined_img = np.hstack(sections)
        blurs.append((joined_img, abs(y[-1] - y[0])))

    return capture_time, exposure_time, blurs


def process_blur(roi, exposure_time, y_diff, line_rate, max_dist=None):
    """
    Takes a straigtened region of interest image and a y_diff value (from extract_rois)
    to measure the wingbeat frequency of the moth in the ROI.

    Parameters
    ----------

    roi : array
        rotated and cropped regions of interest

    exposure_time : float
        exposure time of image in seconds

    y_diff : int
        number of rows the blur spans

    line_rate : float or int
        the line rate of the rolling shutter

    max_dist : int
        maximum number of columns to calculate autocorrelation over. Defaults to a
        half of the length of the image

    Returns
    -------

    spec_density : array
        y-values on spectrogram

    snr : float
        signal to noise ratio

    up_freq : float
        Wingbeat frequency estimate, assuming upward motion

    down_freq : float
        Wingbeat frequency estimate, assuming downward motion
    """
    if max_dist is None:
        max_dist = roi.shape[1] // 2

    spectral_density = np.zeros((max_dist, roi.shape[1] - max_dist), dtype=np.float64)
    gr_roi = roi.mean(axis=2)  # Operate on greyscale image
    # Calculate autocorrelation
    for step in range(max_dist):
        spectral_density[step, ...] = np.mean(
            (gr_roi[:, :-max_dist] - np.mean(gr_roi[:, :-max_dist], axis=0))
            * (
                gr_roi[:, step : step - max_dist]
                - np.mean(gr_roi[:, step : step - max_dist], axis=0)
            ),
            axis=0,
        ) / (
            np.std(gr_roi[:, :-max_dist], axis=0)
            * np.std(gr_roi[:, step : step - max_dist], axis=0)
        )

    # Find wingbeat peak
    total_spectral_density = spectral_density.mean(axis=1)[1:]
    peak_idxs = np.where(
        (total_spectral_density > np.roll(total_spectral_density, -1))
        & (total_spectral_density > np.roll(total_spectral_density, 1))
    )[0][1:]
    sorted_peak_idxs = peak_idxs[np.argsort(total_spectral_density[peak_idxs])][::-1]
    try:
        snrs = []
        best_peak = sorted_peak_idxs[0]  # Temporary value
        for peak_idx in sorted_peak_idxs:
            # Find snr
            trough_values = np.concatenate(
                [
                    total_spectral_density[peak_idx // 3 : (peak_idx * 3) // 4],
                    total_spectral_density[(peak_idx * 5) // 3 : (peak_idx * 7) // 4],
                ]
            )
            snr = (total_spectral_density[peak_idx] - np.mean(trough_values)) / np.std(
                trough_values
            )
            if len(snrs) > 0:
                if snrs[-1] > snr:
                    snr = snrs[-1]
                    break  # Previous peak was the best peak
            snrs.append(snr)
            best_peak = peak_idx

    except (ValueError, IndexError):
        best_peak = -1
        first_trough_idx = -1
        snr = np.nan

    # Calculate wingbeat frequency from peak.
    # Note that due to the ambiguity in the direction of the moth's flight,
    # as well as rolling shutter, there are two possible frequency estimates,
    # with the lower frequency corresponding to an upward direction and the
    # higher frequency corresponding to a downward direction of flight with
    # respect to the camera's orientation.
    corrected_exposure_time = np.sort(
        exposure_time + np.array([y_diff, -y_diff]) / line_rate
    )
    period = [
        np.arange(1, max_dist) * et / roi.shape[1] for et in corrected_exposure_time
    ]
    up_freq, down_freq = [1 / period[i][best_peak] for i in (0, 1)]

    return (
        corrected_exposure_time[0],
        corrected_exposure_time[1],
        period[0],
        period[1],
        total_spectral_density,
        snr,
        up_freq,
        down_freq,
        best_peak,
    )


def make_supplementary_figure(
    file_path, annotation_idx, roi, spectral_density, best_peak, snr
):
    """
    Saves supplementary figure for the wingbeat measurement for a particular annotation.

    Parameters
    ----------

    file_path : str
        Path supplementary figure file (will be overwritten if it already exists)

    annotation_idx : int
        Index of annotation (within the image). Used in plot title.

    roi : array
        rotated and cropped regions of interest

    spectral_density : array
        y-values on spectrogram

    best_peak : int
        period of wingbeat in pixels

    snr : float
        signal-to-noise ratio of autocorrelation at best_peak
    """
    fig = plt.figure()

    ax1 = fig.add_subplot(
        211, title=f"Linearised view of insect motion blur (moth {annotation_idx})"
    )
    ax1.imshow(roi)
    ax1.axvline(best_peak, c="r")

    ax2 = fig.add_subplot(
        212,
        title=f"Autocorrelation along motion blur (SNR: {snr:.2f})",
        ylabel="Correlation",
        xlabel="Distance (pixel columns)",
    )
    ax2.plot(spectral_density)
    ax2.axvline(best_peak, c="r")
    ax2.fill_between(
        [best_peak // 4, (best_peak * 3) // 4], 0, 1, color="k", alpha=0.25, zorder=0
    )
    ax2.fill_between(
        [(best_peak * 5) // 4, (best_peak * 7) // 4],
        0,
        1,
        color="k",
        alpha=0.25,
        zorder=0,
    )
    try:
        ax2.set_ylim(spectral_density.min() - 0.01, spectral_density[best_peak] + 0.01)
    except ValueError:
        pass

    try:
        fig.savefig(file_path)
    except FileNotFoundError:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fig.savefig(file_path)

    plt.close(fig)


def process_annotations(args):
    """
    Passed to worker processes by `AnnotationUtils.extract_wingbeats`. Calls
    `extract_rois`, then `process_blur` on each region. Optionally calls
    `make_supplementary_fig`.

    Parameters
    ----------

    args : tuple

    Returns
    -------

    results : list
    """
    image_metadata, scan_distance, line_rate, max_dist, supplementary_fig = args
    results = []
    regions = image_metadata["regions"]
    if len(regions) > 0:  # Check there are annotations on the image
        image_path = image_metadata["filename"]

        capture_time, exposure_time, blurs = extract_rois(
            regions, image_path, scan_distance
        )
        annotation_idx = 0
        for roi, y_diff in blurs:
            (
                et_up,
                et_dn,
                period_up,
                period_dn,
                spec_dens,
                snr,
                wb_freq_up,
                wb_freq_dn,
                best_peak,
            ) = process_blur(roi, exposure_time, y_diff, line_rate, max_dist)

            if snr is not np.nan:
                results.append(
                    (
                        image_path,
                        capture_time,
                        annotation_idx,
                        best_peak,
                        roi.shape[1],
                        snr,
                        wb_freq_up,
                        wb_freq_dn,
                        et_up,
                        et_dn,
                        ",".join([str(pu) for pu in period_up]),
                        ",".join([str(pd) for pd in period_dn]),
                        ",".join([str(sd) for sd in spec_dens]),
                    )
                )

                if supplementary_fig is not None:
                    supp_fig_path = os.path.join(
                        supplementary_fig,
                        os.path.splitext(image_metadata["filename"])[0]
                        + f"_{annotation_idx}_wingbeat.png",
                    )
                    make_supplementary_figure(
                        supp_fig_path, annotation_idx, roi, spec_dens, best_peak, snr
                    )

                annotation_idx += 1

    return results


def get_metadata(image_file, exif_tags):
    """
    Parameters
    ----------

    image_file : str
        Path to image file

    exif_tags : list
        Metadata tags to include

    Returns
    -------

    out : dict
        Contains the metadata of the image, with all keys and values coerced to str.
    """
    with open(image_file, "rb") as img_file:
        img = exif.Image(img_file)

    out = {}
    for exif_key in exif_tags:
        out[exif_key] = str(img[exif_key])

    return out


def _get_metadata_with_key(img_tup):
    return img_tup[0], get_metadata(img_tup[1], img_tup[2])


def bb_from_via_shape_attributes(shape):
    """
    Get bounding box from VIA shape attributes

    Parameters
    ----------

    shape: dict
        "shape_attributes" dict

    Returns
    -------

    minx, miny, maxx, maxy
    """
    if shape["name"] == "polyline":
        return LineString(zip(shape["all_points_x"], shape["all_points_y"])).bounds
    elif shape["name"] == "circle":
        return (
            shape["cx"] - shape["r"],
            shape["cy"] - shape["r"],
            shape["cx"] + shape["r"],
            shape["cy"] + shape["r"],
        )
    elif shape["name"] == "point":
        return shape["cx"] - 5, shape["cy"] - 5, shape["cx"] + 5, shape["cy"] + 5
    else:
        raise NotImplementedError(shape["name"])


def bb_intersection_over_union(box0, box1):
    """
    Get the intersection over union of two boxes

    Parameters
    ----------

    box0: floats minx, miny, maxx, maxy
    box1: floats minx, miny, maxx, maxy

    Returns
    -------

    iou: float
        Between 0. and 1.
    """
    minx0, miny0, maxx0, maxy0 = box0
    minx1, miny1, maxx1, maxy1 = box1
    minx_inter = max(minx0, minx1)
    miny_inter = max(miny0, miny1)
    maxx_inter = min(maxx0, maxx1)
    maxy_inter = min(maxy0, maxy1)

    if maxx_inter < minx_inter or maxy_inter < miny_inter:
        return 0.0

    area_inter = (maxx_inter - minx_inter) * (maxy_inter - miny_inter)
    area0 = (maxx0 - minx0) * (maxy0 - miny0)
    area1 = (maxx1 - minx1) * (maxy1 - miny1)
    area_union = area0 + area1 - area_inter

    return area_inter / area_union


class AnnotationUtils:
    """
    Provides utilities for working with camfi projects

    Parameters
    ----------

    processes : int
        number of child processes to spawn

    i: str
        path to input VIA project json file. Defaults to sys.stdin

    o: str
        path to output file. Defaults to sys.stdout
    """

    def __init__(self, processes=1, i=None, o=None):
        self.processes = processes
        self.i = i
        self.o = o

    def _output(self, out_str, mode="w"):
        """
        Used by methods on `AnnotationUtils` (rather than print) to output to file or
        sys.stdout.
        """
        if self.o is None:
            print(out_str)
        else:
            with open(self.o, mode) as f:
                print(out_str, file=f)

    def _load(self):
        """
        Used by methods on `AnnotationUtils` to load json file (or sys.stdin)

        Returns
        -------

        annotations : dict
        """
        if self.i is None:
            annotations = json.load(sys.stdin)
        else:
            with open(self.i, "r") as jf:
                annotations = json.load(jf)
        return annotations

    def download_model(self, model="release"):
        """
        Downloads a pretrained image annotation model, returning the path to the model

        Parameters
        ----------

        model: str
            Name of model. Can be one of {"release", "latest"} or a url pointing to the
            model file on the internet. Alternatively, a path to an existing local file
            can be given, in which case the path is returned and nothing else is done.

        Returns
        -------

        model_path: str
            Location that model is saved.
        """
        if model == "release":
            model_url = RLS_MODEL
        elif model == "latest":
            with urlopen(
                "https://raw.githubusercontent.com/J-Wall/camfi/main/models/latest"
            ) as f:
                model_url = f.readline().decode().strip()
        elif os.path.exists(model):
            return model
        else:
            model_url = model

        model_path = os.path.normpath(
            os.path.expanduser(
                os.path.join("~/.camfi", os.path.basename(urlparse(model_url).path))
            )
        )

        if os.path.exists(model_path):
            print(
                f"Model already downloaded.\nLocated at: {model_path}.", file=sys.stderr
            )

        else:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(
                f"Downloading model from {model_url}. Saving to {model_path}",
                file=sys.stderr,
            )
            urlretrieve(model_url, model_path)

        return model_path

    def add_metadata(self, *exif_tags):
        """
        Adds image (EXIF) metadata to VIA project by reading image files. Optionally
        spawns multiple processes (reading the images is usually I/O bound and can take
        some time).
        """
        pool = Pool(self.processes)

        annotation_project = self._load()
        file_attributes = set()

        img_paths = [
            (
                img_key,
                annotation_project["_via_img_metadata"][img_key]["filename"],
                exif_tags,
            )
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
        Takes a list of VIA project files and merges them into one. Ignores --i in
        favour of *annotation_files.

        Parameters
        ---------

        *annotation_files
            list of VIA project json files to merge. Project and VIA settings are taken
            from the first file.
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

        Parameters
        ----------

        **kwargs
            Passed to zipfile.ZipFile
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

    def validate_annotations(self, ground_truth, iou_thresh=0.5):
        """
        Compares annotation file against a ground-truth annotation file for automatic
        annotation validation puposes.

        Validation data is output to a json dict, which includes:

        all_ious: list of [iou, score] pairs
            iou is the Intersection over Union of the bounding boxes of true positives
            to their matched ground truth annotation. All matched annotations are
            included.
            score is the prediction score of the automatic annotation

        polyline_hausdorff_distances: list of [h_dist, score] pairs
            h_dist is the hausdorff distance of a true positive polyline annotation,
            where the annotation is matched to a polyline ground truth annotation. Only
            polyline annotations which matched to a polyline ground truth annotation are
            included.
            score is the prediction score of the automatic annotation

        length_differences: list of [l_diff, score] pairs
            l_diff is calculated as the length of a true positive polyline annotation
            minus the length of it's matched ground truth annotation. Only polyline
            annotations which matched to a polyline ground truth annotation are
            included.
            score is the prediction score of the automatic annotation

        true_positives: list of scores
            score is the prediction score of the automatic annotation

        false_positives: list of scores
            score is the prediction score of the automatic annotation

        false_negatives: int
            number of false negative annotations

        Parameters
        ----------

        ground_truth: str
            Path to ground truth VIA annotations file. Should contain annotations for
            all images in input annotation file.

        iou_thresh: float
            Threshold of intersection-over-union of bounding boxes to be considered a
            match.
        """
        annotations = self._load()
        with open(ground_truth, "r") as f:
            gt_annotations = json.load(f)

        all_ious = []
        polyline_hausdorff_distances = []
        length_differences = []
        true_positives = []
        false_positives = []
        false_negatives = 0

        for img_key, annotation in tqdm(
            annotations["_via_img_metadata"].items(),
            desc="Validating annotations",
            unit="img",
        ):
            gt_annotation = gt_annotations["_via_img_metadata"][img_key]
            regions = annotation["regions"]
            gt_regions = gt_annotation["regions"]

            ious = sp.sparse.dok_matrix((len(regions), len(gt_regions)), dtype="f8")

            for i, j in itertools.product(range(len(regions)), range(len(gt_regions))):
                shape = regions[i]["shape_attributes"]
                bb = bb_from_via_shape_attributes(shape)
                gt_shape = gt_regions[j]["shape_attributes"]
                gt_bb = bb_from_via_shape_attributes(gt_shape)

                iou = bb_intersection_over_union(bb, gt_bb)
                if iou >= iou_thresh:
                    ious[i, j] = iou

            ious = ious.tocsr()
            matches = sp.sparse.csgraph.maximum_bipartite_matching(ious, "column")
            false_negatives += len(gt_regions) - np.count_nonzero(matches >= 0)

            for i, match in enumerate(matches):
                score = regions[i]["region_attributes"]["score"]
                if match >= 0:
                    true_positives.append(score)
                    all_ious.append((ious[i, match], score))
                    shape = regions[i]["shape_attributes"]
                    gt_shape = gt_regions[match]["shape_attributes"]
                    if shape["name"] == gt_shape["name"] == "polyline":
                        linestring = LineString(
                            zip(shape["all_points_x"], shape["all_points_y"])
                        )
                        gt_linestring = LineString(
                            zip(gt_shape["all_points_x"], gt_shape["all_points_y"])
                        )
                        h_dist = linestring.hausdorff_distance(gt_linestring)
                        polyline_hausdorff_distances.append((h_dist, score))
                        l_diff = linestring.length - gt_linestring.length
                        length_differences.append((l_diff, score))

                else:
                    false_positives.append(score)

        output_dict = {
            "all_ious": all_ious,
            "polyline_hausdorff_distances": polyline_hausdorff_distances,
            "length_differences": length_differences,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
        self._output(json.dumps(output_dict, separators=(",", ":")))

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

    def extract_wingbeats(
        self,
        line_rate=inf,
        scan_distance=100,
        max_dist=None,
        supplementary_figures=None,
    ):
        """
        Uses the camfi algorithm to measure the wingbeat frequency of annotated flying
        insect motion blurs in still images.

        Parameters
        ---------

        line_rate : int
            The line rate of the rolling shutter

        scan_distance : int
            Half width of analysis windows (half width of blurs)

        max_dist : int
            Maximum number of columns to calculate autocorrelation over. Defaults to a
            half of the length of the image

        processes : int
            Number of worker processes to spawn. Default 1

        supplementary_figures : str
            Directory in which to put supplementary figures (optional)
        """
        # Print header
        self._output(
            "\t".join(
                [
                    "image_path",
                    "capture_time",
                    "annotation_idx",
                    "best_peak",
                    "blur_length",
                    "snr",
                    "wb_freq_up",
                    "wb_freq_dn",
                    "et_up",
                    "et_dn",
                    "period_up",
                    "period_dn",
                    "spec_dens",
                ]
            )
        )

        annotations = self._load()

        pool = Pool(self.processes)

        tot_annotations = 0
        pb = tqdm(
            pool.imap(
                process_annotations,
                zip(
                    annotations["_via_img_metadata"].values(),
                    itertools.repeat(scan_distance),
                    itertools.repeat(line_rate),
                    itertools.repeat(max_dist),
                    itertools.repeat(supplementary_figures),
                ),
                5,
            ),
            desc="Processing annotations",
            total=len(annotations["_via_img_metadata"]),
            unit="img",
            smoothing=0.0,
        )

        for results in pb:
            for result in results:
                if result[2] is not np.nan:  # Check if snr is not np.nan
                    self._output("\t".join(str(val) for val in result), mode="a")
                    tot_annotations += 1
            pb.set_postfix(refresh=False, tot_annotations=tot_annotations)

        pool.close()
        pool.join()

    def filelist(self, sort=True, shuffle=False):
        """
        Lists the images in the input VIA project

        Parameters
        ----------

        sort: bool
            If True, output is sorted lexigraphically. If False, order is arbitrary

        shuffle: bool or int
            If int, then the output is shuffled using `shuffle` as the seed.
            If True, then the output is shuffled using the system time as the seed.
            If False (default), do not shuffle.
            Shuffling occurs after sorting. For reproducability, set `sort=True`.
        """
        annotations = self._load()
        filenames = [a["filename"] for a in annotations["_via_img_metadata"].values()]
        if sort:
            filenames.sort()

        if isinstance(shuffle, int):
            random.seed(shuffle)
            random.shuffle(filenames)
        elif shuffle is True:
            random.seed()
            random.shuffle(filenames)

        self._output("\n".join(filenames))


def _cli_train():
    fire.Fire(train_model)


def _cli_annotate():
    fire.Fire(annotate)


def main():
    fire.Fire(AnnotationUtils)


if __name__ == "__main__":
    main()
