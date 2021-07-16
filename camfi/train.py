from torch.utils.data import DataLoader

from camfi.data import CamfiDataset
from ._torchutils import get_model_instance_segmentation, train_one_epoch


def train_model_v2(
    dataset: CamfiDataset,
    device: str = "cpu",
    backup_device: Optional[str] = None,
    batch_size: int = 5,
    num_workers: int = 2,
    num_epochs: int = 5,
    outdir: Union[str, os.PathLike] = ".",
    model_name: Optional[str] = None,
    model_name: Optional[str] = None,
    save_intermediate: bool = False,
) -> None:
    pass


def train_model(
    *via_projects: Union[str, os.PathLike],
    load_pretrained_model: Optional[Union[str, os.PathLike]] = None,
    img_dir: Optional[Union[str, os.PathLike]] = None,
    crop: Optional[Tuple[int, int, int, int]] = None,
    point_r: int = 10,
    mask_dilate: int = 5,
    min_annotations: int = 0,
    max_annotations: float = np.inf,
    exclude: Optional[Union[str, os.PathLike]] = None,
    device: str = "cpu",
    num_classes: int = 2,
    batch_size: int = 5,
    num_workers: int = 2,
    num_epochs: int = 5,
    outdir: Union[str, os.PathLike] = ".",
    model_name: Optional[str] = None,
    save_intermediate: bool = False,
) -> None:
    """
    Parameters
    ----------
    via_projects: list of paths
        Path(s) to VIA project files with annotated motion blurs for training on
    load_pretrained_model: path-like
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
    max_annotations: float
        Skip images which have more than max_annotations annotations. Set this if you
        are running into memory issues when training on a GPU.
    exclude: path-like
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
    torchdevice = torch.device(device)
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
        torchutils.get_transform(train=True),
        *via_projects,
        crop=crop,
        point_r=point_r,
        mask_dilate=mask_dilate,
        min_annotations=min_annotations,
        max_annotations=max_annotations,
        exclude=exclude_set,
    )
    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=torchutils.collate_fn,
    )

    # Initialise model
    model = torchutils.get_model_instance_segmentation(num_classes)
    if load_pretrained_model is not None:
        model.load_state_dict(torch.load(load_pretrained_model))
    model.to(torchdevice)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        torchutils.train_one_epoch(
            model, optimizer, data_loader, torchdevice, epoch, print_freq=10
        )
        # update the learning rate
        lr_scheduler.step()

        if save_intermediate or epoch == num_epochs - 1:
            save_path = os.path.join(outdir, f"{model_name}_{epoch}_model.pth")
            torch.save(model.state_dict(), save_path)

    print(f"Training complete. Model saved at {save_path}")
