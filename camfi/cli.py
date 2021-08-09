"""Defines command-line interface to Camfi.
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter as Formatter
from datetime import datetime
from functools import wraps
from inspect import getdoc
from pathlib import Path
from shutil import get_terminal_size
from sys import exit, stderr
from textwrap import fill
from typing import Callable, Optional

from camfi import __version__

CONFIG_URL = "https://camfi.readthedocs.io/en/latest/usage/configuration.html"


def _qprint(*args, **kwargs) -> None:
    pass


def _vprint(*args, **kwargs) -> None:
    print(f"[{datetime.now().astimezone().isoformat()}]", *args, file=stderr, **kwargs)


def _fill_paras(paras: list[str], sep: str = "\n \n", **kwargs) -> str:
    return sep.join(
        para if para.startswith("  ") else fill(para, **kwargs) for para in paras
    )


class ConfigParseError(Exception):
    """Raised if config parsing fails."""


class Commander:
    """Defines commands for camfi cli. Any command defined on this class can be called
    by providing them as command-line arguments to ``camfi``.
    """

    _to_sh = {ord("_"): ord("-")}
    _to_py = {ord("-"): ord("_")}

    def __init__(
        self,
        config_path: Optional[Path],
        input_file: Optional[Path] = None,
        output: Optional[Path] = None,
        disable_progress_bar: Optional[bool] = None,
        vprint: Callable = _vprint,
        vvprint: Callable = _qprint,
    ):
        """Parses config_path file into self.config : camfi.projectconfig.CamfiConfig.

        Parameters
        ----------
        config_path : Optional[Path]
            Path to configuration file. Can be JSON (.json) or StrictYAML (.yaml|.yml).
        output : Optional[Path]
            If set, ``self.default_output`` will be overwritten with this value.
        """
        self._vprint, self._vvprint = vprint, vvprint

        # Import only done after command line parsing to make it faster and more robust.
        self._vvprint("Importing camfi...")
        from camfi.projectconfig import CamfiConfig

        self._vvprint("Done.")

        self._vprint(
            f"Parsing configuration file: {config_path}..."
            if config_path
            else "Creating configuration."
        )
        if config_path is None:
            self.config = CamfiConfig()
        elif config_path.suffix.startswith(".j"):
            self.config = CamfiConfig.parse_file(config_path)
        elif config_path.suffix.startswith(".y"):
            self.config = CamfiConfig.parse_yaml_file(config_path)
        else:
            raise ConfigParseError(
                "Could not determine config format from file suffix. "
                f"Expected one of (.json|.yaml|.yml). Got {config_path.suffix}."
            )
        self._vprint("Done.")

        self._vvprint("Updating command-line configurable config params...")
        if input_file:
            self._vvprint(f"Setting config.via_project_file = {input_file}")
            self.config.via_project_file = input_file
        if output:
            self._vvprint(f"Setting config.default_output = {output}")
            self.config.default_output = output
        if disable_progress_bar is not None:  # Could be True or False
            self._vvprint(
                f"Setting config.disable_progress_bar = {disable_progress_bar}"
            )
            self.config.disable_progress_bar = disable_progress_bar
        self._vvprint("Done updating params.")

    @classmethod
    def _get_command(cls, command: str) -> Callable[[], None]:
        return getattr(cls, command.translate(cls._to_py))

    def __call__(self, command: str) -> None:
        """Calls method of self corresponding to command.

        Parameters
        ----------
        command : str
            Name of command to call. These are the same as the methods defined on this
            class. Any "-" characters are converted to "_" charaters.
        """
        return self._get_command(command)(self)  # type: ignore[call-arg]

    def _write(self, s: str):
        if self.config.default_output:
            with open(self.config.default_output, "w") as f:
                f.write(s)
        else:
            print(s)

    @classmethod
    def cmds(cls) -> dict[str, Optional[str]]:
        """Returns the commands defined by this class.

        Returns
        -------
        commands : dict[str, str]
            Dictionary with command names mapped to command docstrings.
        """
        return {
            choice.translate(cls._to_sh): getdoc(cls._get_command(choice))
            for choice in filter(
                lambda x: not (x.startswith("_") or x == "cmds"), dir(cls)
            )
        }

    def annotate(self) -> None:
        """Performs automatic annotation on all the images in via_project, outputting
        the resulting annotated VIA project file to the configured ``output_path``
        specified under ``annotator.inference`` in the configuration file.
        Requires ``annotator.inference`` to be configured.
        While not strictly required, ``annotator.inference.output_path`` should be
        configured, otherwise the result of annotation will not be saved before the
        program terminates (and this is *probably* not what you want).
        Alternatively, you can configure ``default_output`` either in the configuration
        file or by using the ``-o``/``--output`` flag.
        """
        self.config.annotate()

    def train(self) -> None:
        """Trains a camfi instance segmentation annotation model on manually annotated
        dataset, saving to trained model to the ``outdir`` configured under
        ``annotator.training``. Requires ``annotator.training`` to be configured.
        If ``annotator.inference`` is configured, but under it ``model`` has not been
        explicitely set, then after the model is trained ``model`` of
        ``annotator.inference`` will be set to the newly trained model.
        This means that with proper configuration, ``train`` and ``annotate``
        can be strung together in one command (i.e. ``camfi train annotate``).
        """
        model_path = self.config.train_model()

        # Update model used for annotation
        if self.config.annotator and self.config.annotator.inference:
            if "model" not in self.config.annotator.inference.__fields_set__:
                self._vprint(f"Setting config.annotator.inference.model = {model_path}")
                self.config.annotator.inference.model = model_path
            else:
                self._vprint(
                    "Inference model already set to "
                    f"{self.config.annotator.inference.model} "
                    "Not changing."
                )

    def validate(self) -> None:
        """Validates automatically aquired annotations against ground-truth annotations,
        saving the results to the ``output_dir``
        configured under ``annotator.validation``.
        Requires ``annotator.validation`` to be configured.
        While not strictly required, ``annotator.validation.output_dir``
        should be configured, otherwise the result of validation will not be saved
        before the program terminates (and this is *probably* not what you want).
        If ``image_set`` under ``annotator.validation`` contains
        "train" or "test",
        then ``test_set`` (or ``test_set_file``) under ``annotator.training`` should
        also be configured.
        If ``annotator.training`` is not set,
        then the "train" ``image_set`` will be equivalent to "all",
        and "test" will be an empty set of images.
        It is also possible to leave ``autoannotated_via_project_file`` under
        ``annotator.validation`` unconfigured. In this case, the ``output_path`` from
        ``annotation.inference`` will be validated
        (so at least one of these must be configured).
        Alternatively, you can configure ``default_output`` either in the configuration
        file or by using the ``-o``/``--output`` flag.
        """
        self.config.validate_annotations()

    def load_exif(self) -> None:
        """Loads EXIF metadata into VIA project in-place after reading it from file.
        If ``time`` (and optionally ``camera``) are configured,
        then this will also insert location and corrected timestamp metadata.
        """
        self.config.load_all_exif_metadata()

    def extract_wingbeats(self) -> None:
        """Runs the Camfi algorithm to
        extract wingbeat data from all images in the VIA project,
        inserting that data into the project in-place.
        Requires ``camera`` and ``wingbeat_extraction`` to be configured.
        """
        self.config.extract_all_wingbeats()

    def filter_images(self) -> None:
        """Applies filters to exclude images from VIA project.
        Operates in-place on the VIA project.
        Does nothing if ``filters.image_filters`` isn't configured.
        """
        if self.config.filters is None or self.config.filters.image_filters is None:
            self._vprint("No image filters configured. Skipping.")
            return None
        self.config.apply_image_filters()

    def filter_regions(self) -> None:
        """Applies filters to exclude regions (annotations) from VIA project.
        Operates in-place on the VIA project.
        Does nothing if ``filters.region_filters`` isn't configured.
        """
        if self.config.filters is None or self.config.filters.region_filters is None:
            self._vprint("No region filters configured. Skipping.")
            return None
        self.config.apply_region_filters()

    def write(self) -> None:
        """Writes VIA project to stdout or file
        (set using ``default_output`` configuration parameter or ``-o``/``--output``).
        Can be used after other commands which act in-place on the VIA project
        (e.g. ``load-exif``, ``extract-wingbeats`` and ``apply-filters``).
        Prints to stdout if no output is given.
        """
        self.config.write_project()

    def do_nothing(self) -> None:
        """Does nothing, except parse options and configuration.
        This can be useful if all you want to do is validate and/or convert the
        configuration file.
        This is the default command which is run when ``camfi`` is called.
        """
        pass

    def filelist(self) -> None:
        """Lists the images in the VIA project to stdout or file
        (set using ``default_output`` configuration parameter or ``-o``/``--output``).
        """
        image_files = self.config.filelist()
        self._write("\n".join(str(image_file) for image_file in image_files))

    def region_table(self) -> None:
        """Produces a table with one row per region (annotation).
        Table is written to stout or file
        (set using ``default_output`` configuration parameter or ``-o``/``--output``).
        """
        region_df = self.config.via_project.to_region_dataframe()
        self._write(region_df.to_csv(sep="\t", index=False))

    def image_table(self) -> None:
        """Produces a table with one row per image, with various image metadata columns,
        including n_annotations (the number of annotations in the image).
        Table is written to stout or file
        (set using ``default_output`` configuration parameter or ``-o``/``--output``).
        """
        image_df = self.config.get_image_dataframe()
        self._write(image_df.to_csv(sep="\t", index=False))

    def table(self) -> None:
        """Similar to ``image-table``, but includes weather and sun time columns.
        Produces a table with one row per image, with various image metadata columns,
        including n_annotations (the number of annotations in the image).
        Table is written to stout or file
        (set using ``default_output`` configuration parameter or ``-o``/``--output``).
        """
        image_df = self.config.get_merged_dataframe()
        self._write(image_df.to_csv(sep="\t"))

    def zip_images(self) -> None:
        """Makes a zip archive of all the images in the VIA project file
        (``default_output`` configuration parameter or ``-o``/``--output`` must be set).
        """
        self.config.zip_images()


def _get_description(show_rst: bool, terminal_width: int) -> str:
    description = [
        (
            f"Camfi v{__version__}. "
            "Copyright 2021 Jesse Rudolf Amenuvegbe Wallace "
            "and contributors. "
            "Licenced under the MIT Licence. "
            "Full documentation available at "
            "https://camfi.readthedocs.io/en/latest/. "
            "Source code available from "
            "https://github.com/J-Wall/camfi. "
            "Cite as "
        )
        if not show_rst
        else "",
        (
            "    Wallace (2021). J-Wall/camfi. Zenodo.\n"
            "    https://doi.org/10.5281/zenodo.4971144."
        )
        if not show_rst
        else "",
        (
            "Camfi is a method "
            "for the long-term non-invasive monitoring "
            "of the activity "
            "and abundance "
            "of low-flying insects "
            "using inexpensive wildlife cameras. "
            "It provides utilities "
            "for measuring "
            "the wingbeat frequency "
            "of insects "
            "in still images, "
            "based on the motion blurs "
            "drawn on the image sensor "
            "by the insect "
            "moving through the air. "
            "For large-scale monitoring projects, "
            "camfi enables "
            "automatic annotation "
            "of flying insects "
            "using the Mask R-CNN framework."
        )
        if not show_rst
        else "",
        (
            "Most configuration "
            "for camfi is done with a "
            "configuration file "
            "rather than with "
            "command-line arguments and options. "
            "Documentation for the configuration can be found here: "
            f"{':doc:`configuration`' if show_rst else CONFIG_URL}. "
        ),
        (
            "Camfi makes a bunch of sub-commands available to the command line "
            "(listed below). "
            "If multiple commands are given to a single camfi command, "
            "they will be executed in sequence. "
            "For example, "
            f"the command::"
        ),
        (
            "    $ camfi \\\n"
            "          --input project.json --root datadir \\\n"
            "          --output project_with_wingbeats.json \\\n"
            "          load-exif extract-wingbeats write"
        ),
        (
            "will first load a VIA project from a file, "
            "then load EXIF metadata from the image files (in datadir) "
            "into the project (``load-exif``), "
            "then the camfi wingbeat extraction algoritm will be run to obtain "
            "wingbeat frequencies (``extract-wingbeats``), "
            "and finally the project will be "
            "written to ``project_with_wingbeats.json`` (``write``). "
        ),
        ("Another command::"),
        ("    $ camfi train annotate validate"),
        (
            "would load a VIA project from a file, "
            "train an instance segmentation model "
            "using the images and (manually obtained) annotations "
            "in the project, "
            "and save the new model to disk (``train``). "
            "Then camfi would re-annotate (``annotate``) the images in the project "
            "using the newly trained model, "
            "saving the newly obtained annotations to a new file "
            "(assuming ``annotator`` is properly configured). "
            "Finally, camfi would validate the automatically-obtained annotations "
            "against the manually-obtained ones (``validate``). "
            "All this, while properly handling keeping "
            "training and test image sets separate "
            "where required (and as per the configuration)."
        ),
    ]

    description_str = _fill_paras(description, width=terminal_width)
    return "".join(description_str.split("``"))


def _get_epilog(
    commands: dict[str, Optional[str]], show_rst: bool, terminal_width: int
) -> str:
    epilog = [
        ("Below are the list of commands available to Camfi.\n")
        if show_rst
        else "available commands: "
    ]
    for command, docs in commands.items():
        if not docs:
            docs = "Undocumented."
        if show_rst:
            epilog.append(f"\n{command}\n    {fill(docs, subsequent_indent='    ')}")
        else:
            _command = (
                f"  {command:7}"
                if len(command) < 7
                else f"  {command:{terminal_width - 2}}"
            )
            epilog.append(
                fill(
                    f"{_command} {''.join(docs.split('``'))}",
                    width=terminal_width,
                    subsequent_indent=f"{'':10}",
                )
            )

    return "\n".join(epilog)


def get_argument_parser(show_rst: bool = True) -> ArgumentParser:
    """Defines arguments to the ``camfi`` command.

    Parameters
    ----------
    show_rst : bool
       If False, reStructuredText will be ommitted from description and epilog.

    Returns
    -------
    parser : ArgumentParser
        Command-line argument parser for ``camfi``.
    """
    commands = Commander.cmds()
    terminal_width = get_terminal_size().columns

    parser = ArgumentParser(
        prog="camfi",
        description=_get_description(show_rst, terminal_width),
        epilog=_get_epilog(commands, show_rst, terminal_width),
        formatter_class=Formatter,
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit.",
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="path",
        type=Path,
        help=(
            "Path to configuration file. "
            "Can be JSON (.json) or StrictYAML (.yaml|.yml). "
            "If no configuration file is supplied, "
            "a default (empty) configuration is used. "
            "Most commands require at least some configuration. "
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        metavar="path",
        type=Path,
        help=(
            "Path to input VIA project file. "
            "Replaces ``via_project_file`` in configuration file, "
            "providing an alternative to setting it in the configuration file. "
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="path",
        type=Path,
        help=(
            "Path to output file. "
            "Replaces ``default_output`` in configuration file, "
            "providing an alternative to setting specific outputs in the configuration "
            "file. "
        ),
    )
    parser.add_argument(
        "-r",
        "--root",
        metavar="dir",
        type=Path,
        help=(
            "Directory containing all images for the project. "
            "Replaces ``root`` in configuration file. "
        ),
    )
    parser.add_argument(
        "-d",
        "--disable-progress-bar",
        action="store_const",
        const=True,
        help="Disables progress bars. By default, disable on non-TTY. ",
    )
    parser.add_argument(
        "-p",
        "--progress-bar",
        action="store_const",
        const=True,
        help="Forces progress bars. By default, disable on non-TTY. ",
    )
    parser.add_argument(
        "-j",
        "--json-conf-out",
        metavar="path",
        type=Path,
        help=(
            "If set, configuration will be written "
            "to file in JSON format after it is parsed. "
            "Set to - to have config written to stdout. "
        ),
    )
    parser.add_argument(
        "-y",
        "--yaml-conf-out",
        metavar="path",
        type=Path,
        help=(
            "If set, configuration will be written "
            "to file in StrictYAML format after it is parsed. "
            "Set to - to have config written to stdout. "
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress cli info. You may also like to use ``-d``. ",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Extra verbose cli info. "
    )
    parser.add_argument(
        "commands",
        nargs="*",
        type=str,
        choices=commands.keys(),
        metavar="command",
        default="do-nothing",
        help=(
            "One or more commands to be executed sequentially by camfi. "
            f"{'' if show_rst else f'Can be one of {set(commands.keys())}. '}"
            "Each command uses the same configuration, "
            "as specified by "
            f"{'``-c``/``--config``' if show_rst else '-c/--config'}. "
        ),
    )

    return parser


def main():
    parser = get_argument_parser(show_rst=False)
    args = parser.parse_args()

    if args.version:
        print(f"Camfi v{__version__}")
        exit(0)

    # Set vprint and vvprint
    if args.verbose:
        vprint, vvprint = _vprint, _vprint
    elif args.quiet:
        vprint, vvprint = _qprint, _qprint
    else:
        vprint, vvprint = _vprint, _qprint

    disable_progress_bar = args.disable_progress_bar
    if args.progress_bar:
        disable_progress_bar = False

    commander = Commander(
        args.config,
        input_file=args.input,
        output=args.output,
        disable_progress_bar=disable_progress_bar,
        vprint=vprint,
        vvprint=vvprint,
    )

    if str(args.json_conf_out) == "-":
        vprint("Writing config JSON to stdout...")
        print(commander.config.json(indent=2, exclude_unset=True))
        vprint("Done.")
    elif args.json_conf_out:
        vprint(f"Writing config JSON to {args.json_conf_out}...")
        with open(args.json_conf_out, "w") as f:
            f.write(commander.config.json(indent=2, exclude_unset=True))
        vprint("Done.")

    # Output config YAML
    if str(args.yaml_conf_out) == "-":
        vprint("Writing config YAML to stdout...")
        print(commander.config.yaml())
        vprint("Done.")
    elif args.yaml_conf_out:
        vprint(f"Writing config YAML to {args.yaml_conf_out}...")
        with open(args.yaml_conf_out, "w") as f:
            f.write(commander.config.yaml())
        vprint("Done.")

    # Run commands
    commands = args.commands if isinstance(args.commands, list) else [args.commands]
    for command in commands:
        vprint(f"Running command: {command}")
        commander(command)
        vprint("Done.")


if __name__ == "__main__":
    main()
