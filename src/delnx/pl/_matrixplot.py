import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import marsilea as ma
import marsilea.plotter as mp
import pandas as pd

from delnx.pp._utils import group_by_max

from ._baseplot import BasePlot
from ._palettes import default_palette


@dataclass
class MatrixPlot(BasePlot):
    """
    MatrixPlot visualizes group-level mean expression data as a heatmap with support for group annotations and flexible row grouping.
    """

    def _build_data(self) -> pd.DataFrame:
        """
        Computes group-level mean expression and prepares group metadata.

        Returns
        -------
            pd.DataFrame: Mean expression matrix (groups x markers).
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

        # Compute group-averaged expression
        df = pd.DataFrame(mat, index=group_col)
        self.mean_df = df.groupby(df.index).mean()
        self.mean_df.columns = flat_markers

        # Rearrange self mean df based on factors in _group
        self.mean_df = self.mean_df.reindex(self.adata.obs["_group"].cat.categories)

        group_meta = (
            self.adata.obs[self.groupby_keys]
            .copy()
            .assign(_group=group_col)
            .drop_duplicates("_group")
            .set_index("_group")
        )
        self.group_metadata = group_meta.loc[list(self.mean_df.index)]

        if self.group_metadata.isnull().any().any():
            missing = self.group_metadata[
                self.group_metadata.isnull().any(axis=1)
            ].index.tolist()
            raise ValueError(f"Missing group metadata for: {missing}")

        return self.mean_df

    def _resolve_row_grouping(
        self, index_source: Any | None = None
    ) -> tuple[pd.Categorical | None, list[str] | None]:
        """
        Determines row grouping for the heatmap.

        Parameters
        ----------
        index_source: Any | None
            Optional source for row indices, defaults to mean_df index.

        Returns
        -------
        Tuple of (group labels, group categories) or (None, None).
        """
        # Fallback to mean_df index if not specified
        if index_source is None:
            index_source = self.mean_df.index

        # Auto: treat each row as its own group
        if self.row_grouping == "auto":
            group = pd.Categorical(
                index_source, categories=list(index_source), ordered=True
            )
            return group, list(index_source)

        # No grouping
        elif self.row_grouping is None:
            return None, None

        # Single column from group_metadata
        elif isinstance(self.row_grouping, str):
            values = self.group_metadata.loc[index_source, self.row_grouping]
            categories = (
                values.drop_duplicates().tolist()
            )  # preserve order of appearance
            group = pd.Categorical(values, categories=categories, ordered=True)
            return group, categories

        # Multiple columns → compound grouping
        elif isinstance(self.row_grouping, list):
            df = self.group_metadata.loc[index_source, self.row_grouping].astype(str)
            compound = df.agg("_".join, axis=1)
            categories = compound.drop_duplicates().tolist()
            group = pd.Categorical(compound, categories=categories, ordered=True)
            return group, categories

        # Provided Series or Categorical
        elif isinstance(self.row_grouping, pd.Series | pd.Categorical):
            if isinstance(self.row_grouping, pd.Series):
                values = self.row_grouping.loc[index_source]
            else:
                values = pd.Series(self.row_grouping, index=self.mean_df.index).loc[
                    index_source
                ]
            categories = values.drop_duplicates().tolist()
            group = pd.Categorical(values, categories=categories, ordered=True)
            return group, categories

        else:
            raise ValueError("Invalid value for row_grouping in MatrixPlot.")

    def _add_group_colorbar(self, m: ma.Heatmap, key: str):
        """
        Add a colorbar for a specific group key.

        Parameters
        ----------
        m : ma.Heatmap
            The heatmap object to which the colorbar will be added.
        key : str
            The key in `adata.obs` for which to add the colorbar.

        Raises
        ------
        ValueError
            If the key is not found in `adata.obs`.
        """
        values = self.group_metadata[key]

        # Extract category names and check for custom color palette
        categories = list(self.adata.obs[key].cat.categories)
        uns_key = f"{key}_colors"
        raw_colors = self.adata.uns.get(uns_key)

        # Create color mapping from either .uns or fallback palette
        if raw_colors is None:
            colors = default_palette(len(categories))
        else:
            colors = raw_colors

        palette = dict(zip(categories, colors, strict=False))

        # Restrict palette to the relevant group values
        filtered_palette = {val: palette[val] for val in values}

        label = self.group_names[self.groupby_keys.index(key)]
        colorbar = mp.Colors(list(values), palette=filtered_palette, label=label)
        m.add_left(colorbar, size=self.groupbar_size, pad=self.groupbar_pad)

    def _build_plot(self):
        """
        Build the plot

        Returns
        -------
        marsilea.SizedHeatmap
            The build matrix plot object.
        """
        data = self._build_data()

        # Resolve row grouping
        self.row_group, self.order = self._resolve_row_grouping(
            self.mean_df.index.astype(str)
        )

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

        m = ma.Heatmap(
            data,
            cmap=self.cmap,
            height=self.height,
            width=self.width,
            cbar_kws={"title": "Expression\nin group"},
        )

        m = self._add_extras(m)
        return m


def matrixplot(
    adata: Any,
    markers: Sequence[str],
    groupby: str | list[str],
    save: str | None = None,
    **kwargs,
):
    """
    Create a matrix plot showing mean expression of markers per group.

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
    plot = MatrixPlot(adata=adata, markers=markers, groupby_keys=groupby, **kwargs)
    if save:
        plot.save(save, bbox_inches="tight")
    else:
        plot.show()
