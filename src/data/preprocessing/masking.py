"""
Stage 5 — masking.

By this point every gap that could responsibly be filled has been filled,
and every cell carries a companion `<var>_observed` flag recording
whether the number in it was measured or inferred. Masking turns that
into the explicit, consistent set of masks the model and the loss
function will use.

Three things happen here:

* **The ocean mask is established.** NEER's `land_mask` variable is True
  over ocean (the name comes from the source products; the semantics
  come from `schema.VARIABLE_SPECS`). This step publishes it under the
  unambiguous name `ocean_mask` and treats it as authoritative.
* **Land is cleared.** Every geophysical variable is set to
  `land_fill_value` (NaN by default) over land, and every `_observed`
  flag is set False there. Interpolation across a coastline is the
  classic way to get a plausible-looking temperature two kilometres
  inland, and this is where it gets undone.
* **Validity is recorded.** Per variable, the step reports ocean
  coverage and the observed fraction over ocean — the numbers that say
  whether a variable is usable at all, and that land in the saved
  preprocessing metadata.

The step deliberately does not drop poorly-covered variables. It reports
the coverage and leaves the decision to the caller, in the same spirit as
`src/data/validation`: this layer describes, the human decides.

Nothing here is learned from the data — a coastline is a coastline in
every split.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional

import numpy as np

from src.data.loaders.representation import OceanDataset, build_variable
from src.data.preprocessing._utils import (
    broadcast_spatial,
    data_variables,
    is_mask_variable,
    ocean_mask_of,
    with_values,
    with_variables,
)
from src.data.preprocessing.base import StatelessStep
from src.data.preprocessing.missing import OBSERVED_SUFFIX

#: Name this step publishes the ocean/land mask under (True = ocean).
OCEAN_MASK = "ocean_mask"


class Masker(StatelessStep):
    """Apply the land/ocean mask and publish the mask set consistently."""

    name: ClassVar[str] = "masking"
    title: ClassVar[str] = "Masking"

    def __init__(
        self,
        *,
        mask_variable: str = "land_mask",
        land_fill_value: float = float("nan"),
        apply_to_variables: bool = True,
        publish_ocean_mask: bool = True,
        keep_source_mask: bool = True,
    ) -> None:
        super().__init__()
        self.mask_variable = mask_variable
        self.land_fill_value = land_fill_value
        self.apply_to_variables = apply_to_variables
        self.publish_ocean_mask = publish_ocean_mask
        self.keep_source_mask = keep_source_mask

    def config(self) -> Dict[str, Any]:
        return {
            "mask_variable": self.mask_variable,
            "land_fill_value": (
                None if np.isnan(self.land_fill_value) else float(self.land_fill_value)
            ),
            "apply_to_variables": self.apply_to_variables,
            "publish_ocean_mask": self.publish_ocean_mask,
            "keep_source_mask": self.keep_source_mask,
        }

    # -- transform ---------------------------------------------------------

    def _transform(self, dataset: OceanDataset) -> OceanDataset:
        ocean = ocean_mask_of(dataset, self.mask_variable)
        variables = dict(dataset.variables)
        report: Dict[str, Any] = {"variables": {}}

        if ocean is None:
            report["skipped"] = (
                f"no '{self.mask_variable}' variable; every cell is treated as ocean"
            )
            n_ocean = None
        else:
            n_ocean = int(ocean.sum())
            report["ocean_cells"] = n_ocean
            report["land_cells"] = int(ocean.size - n_ocean)
            report["ocean_fraction"] = round(float(ocean.mean()), 6)

            if self.publish_ocean_mask:
                variables[OCEAN_MASK] = build_variable(
                    OCEAN_MASK,
                    ocean,
                    ("lat", "lon"),
                    units="bool",
                    attrs={
                        "description": "True = ocean, False = land",
                        "source_variable": self.mask_variable,
                    },
                )
            if not self.keep_source_mask and self.mask_variable != OCEAN_MASK:
                variables.pop(self.mask_variable, None)

            if self.apply_to_variables:
                for name, variable in list(variables.items()):
                    if name == OCEAN_MASK or name == self.mask_variable:
                        continue
                    if not {"lat", "lon"} <= set(variable.dims):
                        continue
                    broadcast = broadcast_spatial(ocean, variable.dims, ("lat", "lon"))
                    values = variable.values
                    if is_mask_variable(name, variable):
                        masked = np.asarray(values, dtype=bool) & broadcast
                    else:
                        masked = np.where(
                            broadcast, np.asarray(values, dtype=float), self.land_fill_value
                        ).astype(np.float32)
                    variables[name] = with_values(variable, masked)

        # Coverage report, computed after masking so the numbers describe
        # what downstream code will actually receive.
        masked_dataset = with_variables(dataset, variables)
        for name, variable in data_variables(masked_dataset).items():
            report["variables"][name] = self._coverage(masked_dataset, name, ocean)

        self._report = report
        return masked_dataset

    # -- internals ---------------------------------------------------------

    def _coverage(
        self, dataset: OceanDataset, name: str, ocean: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        variable = dataset[name]
        values = np.asarray(variable.values, dtype=float)
        finite = np.isfinite(values)

        if ocean is not None and {"lat", "lon"} <= set(variable.dims):
            in_domain = np.broadcast_to(
                broadcast_spatial(ocean, variable.dims, ("lat", "lon")), values.shape
            )
        else:
            in_domain = np.ones(values.shape, dtype=bool)

        n_domain = int(in_domain.sum())
        coverage = {
            "n_cells": int(values.size),
            "n_ocean_cells": n_domain,
            "valid_fraction_over_ocean": (
                None if n_domain == 0 else round(float((finite & in_domain).sum() / n_domain), 6)
            ),
        }

        observed = dataset.variables.get(f"{name}{OBSERVED_SUFFIX}")
        if observed is not None and n_domain:
            observed_values = np.asarray(observed.values, dtype=bool)
            coverage["observed_fraction_over_ocean"] = round(
                float((observed_values & in_domain).sum() / n_domain), 6
            )
        return coverage
