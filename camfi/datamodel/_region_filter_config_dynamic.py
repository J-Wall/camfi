"""This module can be ignored unless you are developing camfi.

The code in this file is not directly used by camfi. It is used for generating
_region_filter_config_static.py which is imported by via.py.

Running ``python camf/datamodel/_region_filter_config_fields.py`` from the camfi
root dir will generate _region_filter_config_static.py. The funky thing about this
is this script actually depends on camfi being installed already. Make of that what you
will. Basically, this script can be ignored unless you are developing the
``ViaRegionAttributes`` class specifically, in which case you should run this script
after you make any changes.

_region_filter_config_static.py should not be edited manually.
"""

from numbers import Real
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, create_model

from camfi.datamodel._via_region_attributes import ViaRegionAttributes


class FloatFilter(BaseModel):
    ge: float = Field(
        ...,
        title="Greater or Equal",
        description="Only include region if attribute >= this value.",
    )
    le: float = Field(
        ...,
        title="Less or Equal",
        description="Only include region if attribute <= this value.",
    )
    exclude_none: bool = Field(
        False, description="Whether to exclude region if attribute is not set."
    )


_region_filter_config_fields = {
    name: (
        Optional[FloatFilter],
        Field(None, description=f"Sets threhsolds for the {name} region attribute."),
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
    output = Path(__file__).parent / "_region_filter_config_static.py"
    generate(
        json_schema,
        input_file_type=InputFileType.JsonSchema,
        input_filename=Path(__file__).name,
        output=output,
    )

    # Add config to RegionFilterConfig
    with open(output, "a") as f:
        f.write(
            """
    class Config:
        schema_extra = {
            "description": "Contains options for filtering regions (annotations)."
        }"""
        )
