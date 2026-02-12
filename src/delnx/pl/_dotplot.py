import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import marsilea as ma
import pandas as pd

from delnx.pp._utils import group_by_max

from ._matrixplot import MatrixPlot


@dataclass
class DotPlot(MatrixPlot):
    """
    DotPlot visualizes mean expression and fraction of cells expressing markers per group.

    Inherits from MatrixPlot and uses marsilea's SizedHeatmap for visualization.
    """

    scale_dots: float = 1.0

    def _build_size(self) -> pd.DataFrame:
        """
        Computes group-level detection rate (fraction of cells with non-zero expression).

        Returns
        -------
            pd.DataFrame: Detection rate matrix (groups x markers).
        """
        group_col = self.adata.obs["_group"].astype(str)

        # Flatten markers if given as dict
        if isinstance(self.markers, dict):
            flat_markers = list(itertools.chain.from_iterable(self.markers.values()))
        else:
            flat_markers = self.markers

        # Extract data matrix from layer or X
        if getattr(self, "layer", None):
            if self.layer not in self.adata.layers:
                raise ValueError(f"Layer '{self.layer}' not found in adata.layers.")
            mat = self.adata[:, flat_markers].layers[self.layer]
        else:
            mat = self.adata[:, flat_markers].X

        # Convert to dense if sparse
        if hasattr(mat, "toarray"):
            mat = mat.toarray()

        # Create DataFrame with group index
        df = pd.DataFrame(mat, index=group_col, columns=flat_markers)

        # Compute detection: non-zero → 1, else 0
        detection = df.gt(0).astype(int)

        # Compute detection rate: mean across cells in each group
        detection_rate = detection.groupby(detection.index).mean()

        # Reorder to match group category order if categorical
        if hasattr(self.adata.obs["_group"], "cat"):
            detection_rate = detection_rate.reindex(self.adata.obs["_group"].cat.categories)

        return detection_rate

    def _build_plot(self):
        """
        Build the plot.

        Returns
        -------
        marsilea.SizedHeatmap
            The build dot plot object.
        """
        data = self._build_data()
        size = self._build_size()

        # Resolve row grouping
        self.row_group, self.order = self._resolve_row_grouping(self.mean_df.index.astype(str))

        # Check if dendrogram is specified
        # If yes, we have to precompute the dendrogram
        # since the dendrograms are computed on the fly, we will have to
        # change the ordering of the markers & reextract the matrix
        # needed if column grouping is enabled
        if self.dendrograms and self.column_grouping:
            for pos in self.dendrograms:
                if pos in ["left", "right"]:
                    cb = ma.Heatmap(data)
                    cb.add_dendrogram(pos, add_base=False)

                    deform_order = cb.get_deform()
                    deform_order._run_cluster()

                    row_order = deform_order.row_reorder_index
                    data_reordered = data.iloc[row_order, :]
                    self.markers = group_by_max(data_reordered.T)

                    data = self._build_data()

        # Scale the data if scaling is enabled
        data = self._scale_data(data, self.scaling_keys)

        m = ma.SizedHeatmap(
            size=size,
            color=data,
            sizes=(1, self.scale_dots * 100),
            width=self.width,
            height=self.height,
            cmap=self.cmap,
            edgecolor=None,
            linewidth=0,
            color_legend_kws={"title": "Expression\nin group"},
            size_legend_kws={
                "title": "Fraction of cells\nin group (%)",
                "labels": ["20%", "40%", "60%", "80%", "100%"],
                "show_at": [0.2, 0.4, 0.6, 0.8, 1.0],
            },
        )

        m = self._add_extras(m)
        return m


def dotplot(
    adata: Any,
    markers: Sequence[str],
    groupby: str | list[str],
    save: str | None = None,
    **kwargs,
):
    """
    Create a dot plot showing mean expression and fraction of cells expressing markers per group.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    markers : sequence of str
        Marker genes/features to plot.
    groupby: Key(s) in adata.obs to group by.
    **kwargs
        Additional arguments passed to DotPlot.
    """
    plot = DotPlot(adata=adata, markers=markers, groupby_keys=groupby, **kwargs)
    if save:
        plot.save(save, bbox_inches="tight")
    else:
        plot.show()
