"""Defines procedures for training, and evaluation automatic camfi annotation models,
and for using them for making automatic annotations (inference). Depends on camfi.util,
camfi.datamodel.autoannotation, camfi.datamodel.geometry, camfi.datamode.via, as well
as ._torchutils and ._models."""

from datetime import datetime
import itertools
from math import pi
from pathlib import Path
from typing import Any, Callable, Optional, Union
from sys import stderr

import numpy as np
from pydantic import (
    BaseModel,
    DirectoryPath,
    NonNegativeInt,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    validator,
)
from scipy import sparse
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.mask_rcnn import MaskRCNN
from tqdm import tqdm, trange

from camfi.datamodel.autoannotation import CamfiDataset, Prediction
from camfi.datamodel.geometry import (
    BoundingBox,
    CircleShapeAttributes,
    PolylineShapeAttributes,
)
from camfi.datamodel.via import (
    ViaFileAttributes,
    ViaMetadata,
    ViaProject,
    ViaRegion,
    ViaRegionAttributes,
)
from camfi.models import model_urls
from camfi.util import (
    endpoint_truncate,
    smallest_enclosing_circle,
    weighted_intersection_over_minimum,
    Field,
)
from ._torchutils import collate_fn, get_model_instance_segmentation, train_one_epoch


def load_annotation_model(model_path_or_url: Union[Path, str]) -> MaskRCNN:
    """Loads a camfi annotation model. Accepts any model key provided in
    camfi.models, a Path object, or a URL str.

    Parameters
    ----------
    model_path_or_url : Union[Path, str]
        Path to .pth file specifying model parameters, model name defined in
        camfi.models.model_urls, or url to model to download from the internet.

    Returns
    -------
    model : MaskRCNN
        Instance segmentation model used for automatic annotation.
    """
    print(f"Loading model: {model_path_or_url}", file=stderr)
    model = get_model_instance_segmentation(2, pretrained=False)
    if isinstance(model_path_or_url, Path):
        state_dict = torch.load(model_path_or_url)
    elif model_path_or_url in model_urls:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[model_path_or_url])
    else:
        state_dict = torch.hub.load_state_dict_from_url(model_path_or_url)
    model.load_state_dict(state_dict)
    return model


def copy_annotation_model(model: MaskRCNN) -> MaskRCNN:
    """Copies a camfi annotation model.

    Parameters
    ----------
    model : MaskRCNN
        Model to copy.

    Returns
    -------
    model_copy : MaskRCNN
        Copy of model.
    """
    model_copy = get_model_instance_segmentation(2, pretrained=False)
    model_copy.load_state_dict(model.state_dict())
    return model_copy


def train_model(
    dataset: CamfiDataset,
    load_pretrained_model: Optional[Union[Path, str]] = None,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 5,
    num_workers: int = 2,
    num_epochs: int = 10,
    outdir: DirectoryPath = Path(),
    model_name: Optional[str] = None,
    save_intermediate: bool = False,
) -> Path:
    """Trains a camfi instance segmentation annotation model on specified dataset,
    saving to trained model to outdir.

    Parameters
    ----------
    dataset : CamfiDataset
        Dataset on which to train the model.
    load_pretrained_model : Optional[Union[Path, str]]
        Path or url to model parameters file. If set, will load the pretrained
        parameters. By default, will start with a model pre-trained on the Microsoft
        COCO dataset.
    device : Union[str, torch.device]
        E.g. "cpu" or "cuda". Training is typically much faster on a GPU. Use "cuda" for
        Nvidia GPUs.
    batch_size : int
        Number of images to load at once.
    num_workers : int
        Number of worker processes for data loader to spawn.
    num_epochs : int
        Number of epochs to train.
    outdir : DirectoryPath
        Path to directory where to save model(s).
    model_name : Optional[str]
        Identifier to include in model save file. By default the current date in
        YYYYmmdd format.
    save_intermediate : bool
        If True, model is saved after each epoch, not just after all epoch are complete.
        This is recommended, especially if training on a service which could terminate
        unpredicatbly (e.g. Google Colab).

    Returns
    -------
    model_path : Path
        Path to saved model.
    """
    # Parameter setting
    device = torch.device(device)
    if model_name is None:
        model_name = datetime.now().strftime("%Y%m%d")

    # Initialise data_loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # Initialise model
    if load_pretrained_model is not None:
        model = load_annotation_model(load_pretrained_model)
    else:
        model = get_model_instance_segmentation(2)
    model.to(device)

    # Initialise optimiser and lr_scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=3, gamma=0.1)

    # Train the model
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimiser, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

        if save_intermediate or epoch == num_epochs - 1:
            save_path = outdir / f"{model_name}_{epoch}_model.pth"
            torch.save(model.state_dict(), save_path)

    print(f"Training complete. Model saved at {save_path}")
    return save_path


class Annotator(BaseModel):
    """Provides methods for automatically annotating images of flying insects using a
    pre-trained instance segmentation model.

    Parameters
    ----------
    dataset : CamfiDataset
        Dataset to annotate.
    model : Union[str, Path, MaskRCNN]
        Either a path to state dict file which defines the segmentation model, or a url
        pointing to a model to download, or one of the model names defined in
        camfi.models.model_urls.
        Alternatively, a MaskRCNN instance can be given directly.
    device : Union[str, torch.device]
        Specifies device to run inference on. E.g. set to "cuda" to use an Nvidia GPU.
    backup_device : Optional[Union[str, torch.device]]
        Specifies device to run inference on when a runtime error occurs while using
        device. Probably only makes sense to set this to "cpu" if device="cuda". This
        option enables the annotator to leverage a GPU with limited memory capacity
        without crashing if a difficult image is encountered.
    backup_model: Optional[MaskRCNN]
        Defines the backup model. Will be automatically generated if backup_device is
        set. Should not be set manually.
    split_angle : PositiveFloat
        Approximate maximum angle between polyline segments in degrees. Note that this
        will immediately be converted to radians upon instantiation of Annotator.
    poly_order : PositiveInt
        Order of polynomial used for fitting motion blur paths.
    endpoint_method : Callable[[np.ndarray, ...], tuple[NonNegativeInt, NonNegativeInt]]
        Method to find endpoints of motion blurs. The first argument to this method
        should be a cropped mask np.ndarray.
    endpoint_extra_args : list[Any]
        Extra arguments to pass to endpoint_method.
    score_thresh : float
        Score threshold between 0.0 and 1.0 for automatic annotations to be kept.
    overlap_thresh : float
        Minimum proportion of overlap (weighted intersection over minimum) between two
        instance segmentation masks to infer that one of the masks should be discarded.
    edge_thresh : NonNegativeInt
        Minimum distance an annotation has to be from the edge of the image before it is
        converted from a polyline annotation to a circle annotation.
    """

    dataset: CamfiDataset
    model: MaskRCNN = "release"
    device: Union[str, torch.device] = "cpu"
    backup_device: Optional[Union[str, torch.device]] = None
    backup_model: Optional[MaskRCNN] = None
    split_angle: PositiveFloat = 15.0
    poly_order: PositiveInt = 2
    endpoint_method: Callable[
        ..., tuple[NonNegativeInt, NonNegativeInt]
    ] = endpoint_truncate
    endpoint_extra_args: list[Any] = [10]
    score_thresh: float = 0.4
    overlap_thresh: float = 0.4
    edge_thresh: NonNegativeInt = 20
    backup_model_used: int = 0

    class Config:
        arbitrary_types_allowed = True

    @validator("model", pre=True, always=True)
    def get_model(cls, v):
        if isinstance(v, MaskRCNN):
            return v
        else:
            return load_annotation_model(v)

    @validator("device", always=True)
    def put_model_on_device_and_set_to_eval(cls, v, values):
        print(f"Putting model on device: {v}", file=stderr)
        v = torch.device(v)
        values["model"].to(v)
        values["model"].eval()
        return v

    @validator("backup_model", pre=True, always=True)
    def copy_model_to_backup_device(cls, v, values):
        assert v is None, "Should not set 'backup_model'. It will be set automatically"
        if "backup_device" in values and values["backup_device"] is not None:
            v = copy_annotation_model(values["model"])
            v.to(values["backup_device"])
            v.eval()

        return v

    @validator("split_angle", always=True)
    def convert_split_angle_to_radians(cls, v):
        return v * pi / 180.0

    def get_prediction(self, img_idx: NonNegativeInt) -> Prediction:
        """Run predicion on a single image. First tries to use the model on self.device,
        and falls back to the model on self.backup_device if a RuntimeError is caught
        (if set).

        Parameters
        ----------
        img_idx: int
            Index of image in via project.

        Returns
        -------
        prediction: Prediction
            Output of model prediction.
        """
        try:
            img, _ = self.dataset[img_idx]
        except (OSError, RuntimeError) as e:
            print(
                f"Error loading {self.dataset.metadata(img_idx).filename}. {e!r}. Skipping.",
                file=stderr,
            )
            return Prediction.empty()

        with torch.no_grad():
            try:
                prediction = self.model([img.to(self.device)])[0]
            except RuntimeError:
                if self.backup_model:
                    prediction = self.backup_model([img.to(self.backup_device)])[0]
                    self.backup_model_used += 1
                else:
                    raise

        del img

        return Prediction.from_tensor_dict(prediction)

    def filter_annotations(self, prediction: Prediction) -> Prediction:
        """Applies self.score_thresh and self.overlap_thresh to filter out poor quality
        annotations.

        Parameters
        ----------
        prediction : Prediction
            Output of model prediction.

        Returns
        -------
        filtered_prediction : Prediction
            Filtered prediction.
        """
        # Remove predictions with below-threshold score
        prediction = prediction.filter_by_score(self.score_thresh)
        n_predictions = len(prediction)

        if n_predictions == 0:
            return prediction

        # Calculate mask overlaps for all pairs of predicted instances
        mask_overlaps = np.zeros((n_predictions, n_predictions), dtype="f4")

        for i, j in itertools.combinations(range(n_predictions), 2):
            if prediction.boxes[i].overlaps(prediction.boxes[j]):
                mask_overlaps[i, j] = weighted_intersection_over_minimum(
                    prediction.masks[i], prediction.masks[j]
                )
                mask_overlaps[j, i] = mask_overlaps[i, j]

        # Remove worst overlapping instances until there are no above-threshold overlaps
        keep = set(range(n_predictions))

        overlap_mask = mask_overlaps.max(axis=1) >= self.overlap_thresh
        while np.any(overlap_mask):
            # Figure out which overlapping annotation has the worst score
            overlap_annotations = np.where(overlap_mask)[0]
            to_discard = overlap_annotations[
                np.argmin(np.array(prediction.scores)[overlap_annotations])
            ]
            # Remove the annotation
            keep.remove(to_discard)
            mask_overlaps[to_discard, :] = 0.0
            mask_overlaps[:, to_discard] = 0.0

            overlap_mask = mask_overlaps.max(axis=1) >= self.overlap_thresh

        return prediction.get_subset_from_index(list(keep))

    def fit_poly(
        self,
        box: BoundingBox,
        mask: torch.Tensor,
    ) -> Union[PolylineShapeAttributes, CircleShapeAttributes, None]:
        """Uses polynomial regression to fit a polyline annotation to the provided
        segmentation mask.

        Parameters
        ----------
        box : BoundingBox
            Fully contains the object to be annotated.
        mask : tensor or array
            Segmentation mask of instance with shape (image_width, image_height).

        Returns
        -------
        shape_attributes : Union[PolylineShapeAttributes, CircleShapeAttributes, None]
            Geometry of automatic annotation.
        """
        portrait = box.is_portrait()
        crop_mask = box.crop_image(mask).cpu().numpy().reshape(box.shape)

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
        fit_mask_vals = crop_mask[y_vals.astype("i4"), x_vals.astype("i4")]

        endpoints = ind_vals[
            list(self.endpoint_method(fit_mask_vals, *self.endpoint_extra_args))
        ]

        # Approximate polynomial segment with polyline
        end_gradients = poly_fit.deriv()(endpoints)
        end_angles = np.arctan(end_gradients)
        angle_diff = abs(end_angles[1] - end_angles[0])
        all_points_ind, all_points_dep = poly_fit.linspace(
            n=int(np.ceil(angle_diff / self.split_angle) + 2), domain=endpoints
        )
        all_points_x = list((all_points_ind, all_points_dep)[portrait] + box.x0)
        all_points_y = list((all_points_dep, all_points_ind)[portrait] + box.y0)
        shape_attributes: Union[PolylineShapeAttributes, CircleShapeAttributes, None]
        try:
            shape_attributes = PolylineShapeAttributes(
                all_points_x=all_points_x, all_points_y=all_points_y
            )
        except ValidationError:
            try:
                cx, cy, r = smallest_enclosing_circle(zip(all_points_x, all_points_y))
                shape_attributes = CircleShapeAttributes(cx=cx, cy=cy, r=r)
            except ValidationError:
                shape_attributes = None

        return shape_attributes

    def convert_to_circle(
        self,
        polyline: PolylineShapeAttributes,
        img_shape: tuple[PositiveInt, PositiveInt],
    ) -> Union[PolylineShapeAttributes, CircleShapeAttributes]:
        """Checks if a polyline annotation is close to the edge of an image, and if so,
        converts it to a circle annotation by computing the smallest enclosing circle of
        all points in the polyline.

        Parameters
        ----------
        polyline : PolylineShapeAttributes
            Shape to convert if too close to edge.
        img_shape: tuple[int, int]
            Height and width of image.

        Returns
        -------
        shape_attributes : Union[PolylineShapeAttributes, CircleShapeAttributes]
            Geometry of annotation after (possible) conversion. If polyline does not
            go too close to the edge of the image, then polyline is returned unchanged.
            Else, a circle annotation is returned.
        """
        polyline_accepted_region = BoundingBox.from_shape(
            img_shape, border=self.edge_thresh
        )
        if polyline.in_box(polyline_accepted_region):
            return polyline

        return polyline.as_circle()

    def annotate_img(self, img_idx: int) -> list[ViaRegion]:
        """Calls self.get_prediction, self.filter_annotations, and self.fit_poly to
        produce annotations for an image specified with img_idx.

        Parameters
        ----------
        img_idx: int
            Index of image in via project.

        Returns
        -------
        regions : list[ViaRegion]
            list of annotations for image.
        """
        prediction = self.get_prediction(img_idx)
        prediction = self.filter_annotations(prediction)

        regions = []

        for i in range(len(prediction)):
            box = prediction.boxes[i]
            mask = prediction.masks[i]
            score = prediction.scores[i]
            shape_attributes = self.fit_poly(box, mask)
            if shape_attributes is None:
                continue
            if shape_attributes.name == "polyline":
                assert isinstance(shape_attributes, PolylineShapeAttributes)
                shape_attributes = self.convert_to_circle(
                    shape_attributes, (mask.shape[-2], mask.shape[-1])
                )
            region_attributes = ViaRegionAttributes(score=score)
            regions.append(
                ViaRegion(
                    region_attributes=region_attributes,
                    shape_attributes=shape_attributes,
                )
            )

        return regions

    def annotate(self, disable_progress_bar: Optional[bool] = True) -> ViaProject:
        """Calls self.annotate_img on all images and returns a ViaProject instance.
        Copies the `via_attributes` and `via_settings` fields from
        `self.dataset.via_project`, and just replaces the `via_img_metadata` field.

        Parameters
        ----------
        disable_progress_bar : Optional[bool]
            If True (default), progress bar is disabled.
            If set to None, disable on non-TTY.

        Returns
        -------
        project : ViaProject
            With automatic annotations made.
        """
        via_img_metadata: dict[str, ViaMetadata] = {}

        postfix = {"tot_annotations": 0}
        if self.backup_device:
            postfix["backup_device_used"] = self.backup_model_used
        pb = trange(
            len(self.dataset),
            disable=disable_progress_bar,
            desc="Annotating images",
            unit="img",
            dynamic_ncols=True,
            ascii=True,
            postfix=postfix,
        )
        for img_idx in pb:
            img_key = self.dataset.keys[img_idx]
            regions = self.annotate_img(img_idx)
            in_metadata = self.dataset.metadata(img_idx)
            out_metadata = ViaMetadata.construct(
                file_attributes=in_metadata.file_attributes.copy(),
                filename=in_metadata.filename,
                regions=regions,
                size=in_metadata.size,
            )
            via_img_metadata[img_key] = out_metadata
            postfix["tot_annotations"] += len(regions)
            if self.backup_device:
                postfix["backup_device_used"] = self.backup_model_used
            pb.set_postfix(postfix, refresh=False)

        print(f"Annotation complete.", file=stderr)
        return ViaProject.construct(
            via_attributes=self.dataset.via_project.via_attributes,
            via_img_metadata=via_img_metadata,
            via_settings=self.dataset.via_project.via_settings,
        )


class AnnotationValidationResult(BaseModel):
    """Contains various metrics for assessing the quality of a set of automatically
    obtained annotations of flying insects.

    Parameters
    ----------
    ious : list[tuple[NonNegativeFloat, NonNegativeFloat]]
        list of (iou, score) pairs.
        iou is the Intersection over Union of the bounding boxes of true positives
        to their matched ground truth annotation. All matched annotations are
        included.
    polyline_hausdorff_distances : list[tuple[NonNegativeFloat, NonNegativeFloat]]
        list of (h_dist, score) pairs.
        h_dist is the hausdorff distance of a true positive polyline annotation,
        where the annotation is matched to a polyline ground truth annotation. Only
        polyline annotations which matched to a polyline ground truth annotation are
        included.
    length_differences : list[tuple[float, NonNegativeFloat]]
        list of (l_diff, score) pairs.
        l_diff is calculated as the length of a true positive polyline annotation
        minus the length of it's matched ground truth annotation. Only polyline
        annotations which matched to a polyline ground truth annotation are
        included.
    true_positives : list[NonNegativeFloat]
        list of scores.
    false_positives : list[NonNegativeFloat]
        list of scores. Score is the prediction score of the automatic annotation.
    false_negatives : int
        Number of false negative annotations.
    """

    ious: list[tuple[NonNegativeFloat, NonNegativeFloat]] = []
    polyline_hausdorff_distances: list[tuple[NonNegativeFloat, NonNegativeFloat]] = []
    length_differences: list[tuple[float, NonNegativeFloat]] = []
    true_positives: list[NonNegativeFloat] = []
    false_positives: list[NonNegativeFloat] = []
    false_negatives: NonNegativeInt = 0


def validate_annotations(
    auto_annotations: ViaProject,
    ground_truth: ViaProject,
    iou_thresh: float = 0.5,
    subset_functions: Optional[dict[str, Callable[[ViaMetadata], bool]]] = None,
    disable_progress_bar: Optional[bool] = True,
) -> list[AnnotationValidationResult]:
    """Compares automatic annotations against a ground-truth annotations for validation
    puposes. Validation data is stored in an AnnotationValidationResult object.

    Parameters
    ----------
    auto_annotations : ViaProject
        Automatically obtained annotations to assess.
    ground_truth : ViaProject
        Manually created ground-truth annotations.
    iou_thresh : float
        Threshold of intersection-over-union of bounding boxes to be considered a
        match. Typically, this is 0.5.
    subset_functions : Optional[dict[str, Callable[[ViaMetadata], bool]]]
        Mapping from subset name to subset function. If set, validation will be repeated
        multiple times with different subsets, once for each element.
    disable_progress_bar : Optional[bool]
        If True (default), progress bar is disabled.
        If set to None, disable on non-TTY.

    Returns
    -------
    validation_results : list[AnnotationValidationResult]
        list containing instances of AnnotationValidationResult. If subset_functions is
        set, then validation_results will have len(subset_functions) elements. By
        default it will just contain one element.
    """
    if subset_functions is None:
        subset_functions = {"all": lambda x: True}

    results: list[AnnotationValidationResult] = []

    for name, subset_function in subset_functions.items():
        gt_annotations = ground_truth.filtered_copy(subset_function)

        result = AnnotationValidationResult()

        for img_key, gt_metadata in tqdm(
            gt_annotations.via_img_metadata.items(),
            disable=disable_progress_bar,
            desc=f"Validating {name} annotations",
            unit="img",
            dynamic_ncols=True,
            ascii=True,
        ):
            metadata = auto_annotations.via_img_metadata[img_key]
            ious = sparse.dok_matrix(
                (len(metadata.regions), len(gt_metadata.regions)), dtype="f8"
            )

            for i, j in itertools.product(
                range(len(metadata.regions)), range(len(gt_metadata.regions))
            ):
                iou = metadata.regions[i].shape_attributes.intersection_over_union(
                    gt_metadata.regions[j].shape_attributes
                )
                if iou >= iou_thresh:
                    ious[i, j] = iou

            ious = ious.tocsr()
            matches = sparse.csgraph.maximum_bipartite_matching(ious, "column")
            result.false_negatives += len(gt_metadata.regions) - np.count_nonzero(
                matches >= 0
            )

            for i, match in enumerate(matches):
                score = metadata.regions[i].region_attributes.score
                if score is None:
                    raise ValueError(
                        "Invalid automatically obtained annotation. "
                        "Ensure that auto_annotations were obtained automatically "
                        f"(region {i} of {img_key} missing 'score' region_attribute)."
                    )
                elif match >= 0:
                    result.true_positives.append(score)
                    result.ious.append((ious[i, match], score))
                    shape = metadata.regions[i].shape_attributes
                    gt_shape = gt_metadata.regions[match].shape_attributes
                    if shape.name == gt_shape.name == "polyline":
                        assert isinstance(shape, PolylineShapeAttributes)
                        h_dist = shape.hausdorff_distance(gt_shape)
                        result.polyline_hausdorff_distances.append((h_dist, score))
                        l_diff = shape.length() - gt_shape.length()
                        result.length_differences.append((l_diff, score))

                else:
                    result.false_positives.append(score)

        results.append(result)

    return results
