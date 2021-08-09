"""This module can be ignored unless you are developing camfi.

The code in this file is not directly used by camfi. It is used for generating
_region_filter_config_static.py which is imported by via.py.

Running ``python camf/datamodel/_region_filter_config_dynamic.py`` from the camfi
root dir will generate region_filter_config.py. The funky thing about this
is this script actually depends on camfi being installed already. Make of that what you
will. Basically, this script can be ignored unless you are developing the
``ViaRegionAttributes`` class specifically, in which case you should run this script
after you make any changes.

region_filter_config.py should not be edited manually.
"""

from numbers import Real
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, create_model

from camfi.datamodel.via_region_attributes import ViaRegionAttributes
from camfi.util import Field


class FloatFilter(BaseModel):
    ge: float = Field(
        ...,
        title="Greater-than or Equal-to",
        description="Only include region if attribute >= this value.",
    )
    le: float = Field(
        ...,
        title="Less-than or Equal-to",
        description="Only include region if attribute <= this value.",
    )
    exclude_none: bool = Field(
        False, description="Whether to exclude region if attribute is not set."
    )


_region_filter_config_fields = {
    name: (
        Optional[FloatFilter],
        Field(None, description=f"Sets filters for the {name} region attribute."),
    )
    for name in ViaRegionAttributes.__fields__.keys()
    if issubclass(ViaRegionAttributes.__fields__[name].type_, Real)
}

RegionFilterConfig = create_model(  # type: ignore[var-annotated]
    "RegionFilterConfig",
    **_region_filter_config_fields,  # type: ignore[arg-type]
)


if __name__ == "__main__":

    # Deferred imports
    from datamodel_code_generator import InputFileType, generate

    # Generate schema
    json_schema = RegionFilterConfig.schema_json()

    # Generate _region_filter_config_static.py
    output = Path(__file__).parent / "region_filter_config.py"
    generate(
        json_schema,
        input_file_type=InputFileType.JsonSchema,
        input_filename=Path(__file__).name,
        output=output,
    )

    # Add config to RegionFilterConfig
    desc = (
        "Contains options for filtering regions (annotations) from images, "
        "based on the values of region attributes. "
    )
    with open(output, "a") as f:
        f.write(
            f"""
    class Config:
        schema_extra = {{
            "description": {desc!r}
        }}"""
        )
