import itertools
from dataclasses import dataclass, field

import anndata as ad
import marsilea as ma
import marsilea.plotter as mp
import matplotlib.pyplot as plt
import pandas as pd

from ._palettes import default_palette


@dataclass
class BasePlot:
    """
    Base class for plotting annotated data matrices with AnnData and Marsilea.

    This class provides a flexible interface for generating annotated heatmaps and related
    visualizations from single-cell or other high-dimensional data stored in AnnData objects.
    It supports grouping, annotation, and customization of plot appearance, and is designed
    to be extended for specific plot types.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing the data to be plotted.
    markers : list[str]
        List of feature names (e.g., genes) to be plotted. Can also be a dict mapping group names to lists of features.
    groupby_keys : str | list[str]
        Key(s) in `adata.obs` used for grouping data. If multiple keys are provided, they will be combined into a single group label.
    layer : str | None
        Layer in `adata.layers` to use for the data matrix. If None, uses the default `.X` matrix.
    scaling_keys : list[str] | None
        Keys in `adata.obs` for group-wise scaling of the data. If None, no scaling is applied. If an empty list, global scaling is applied.
    row_grouping : str | list[str] | pd.Series | pd.Categorical | None
        Defines how to group rows in the heatmap. Can be a column name, list of names, or a Series/Categorical. If "auto", uses the group labels.
    column_grouping : bool
        Whether to group columns based on the markers. If True, markers will be grouped by their keys if provided as a dict.
    cmap : str
        Colormap to use for the heatmap. Default is "viridis".
    height : float
        Height of the heatmap in inches.
    width : float
        Width of the heatmap in inches.
    scale_render : float
        Scaling factor for rendering the plot. Default is 1.0.
    show_column_names : bool
        Whether to display column names (markers) in the heatmap.
    show_row_names : bool
        Whether to display row names (group labels) in the heatmap.
    show_legends : bool
        Whether to show legends for annotations and colorbars.
    groupbar_size : float
        Size of the group colorbars in the heatmap.
    groupbar_pad : float
        Padding around the group colorbars.
    chunk_rotation : int
        Rotation angle for chunk labels in the heatmap.
    chunk_align : str
        Alignment of chunk labels in the heatmap. Options are "left", "center", or "right".
    chunk_pad : float
        Padding around chunk labels in the heatmap.
    chunk_fontsize : int
        Font size for chunk labels in the heatmap.
    dendrograms : list[str] | None
        Positions for dendrograms in the heatmap. Can include "left", "right", "top", or "bottom". If None, no dendrograms are added.
    group_names : str | list[str] | None
        Names for the groups used in annotations. If None, uses `groupby_keys` as default.
    min_group_size : int
        Minimum number of cells required in a group to be included in the plot. Groups with fewer cells will be filtered out.
    """

    adata: ad.AnnData
    markers: list[str]
    groupby_keys: str | list[str]

    # Layer to use for the data matrix
    # If None, uses the default `.X` matrix from `adata`
    layer: str | None = None

    # Whether to scale the data before plotting
    scaling_keys: list[str] | None = None

    # Row grouping options
    row_grouping: str | list[str] | pd.Series | pd.Categorical | None = "auto"

    # Column grouping options
    column_grouping: bool = False

    # Plotting parameters for heatmap
    cmap: str = "viridis"

    # Layout and appearance parameters
    height: float = 3.5
    width: float = 3
    scale_render: float = 1.0

    # Annotations and labels
    show_column_names: bool = True
    show_row_names: bool = True
    show_legends: bool = True

    # Grouping and annotation parameters
    groupbar_size: float = 0.1
    groupbar_pad: float = 0.05
    chunk_rotation: int = 0
    chunk_align: str = "center"
    chunk_pad: float = 0.1
    chunk_fontsize: int = 10

    # Dendrograms
    dendrograms: list[str] | None = None

    # Group names for annotations
    group_names: str | list[str] | None = None

    # Internal attributes initialized in __post_init__
    group_labels: pd.Categorical = field(init=False)
    group_metadata: pd.DataFrame = field(init=False)

    # Keep groups with more than this many cells
    min_group_size: int = 10

    def __post_init__(self):
        """Initialize group labels and add them to adata.obs as '_group'."""
        # Make category dtype for groupby_keys if not already categorical
        if isinstance(self.groupby_keys, str):
            if self.groupby_keys not in self.adata.obs.columns:
                raise ValueError(f"Key '{self.groupby_keys}' not found in adata.obs.")
            if not pd.api.types.is_categorical_dtype(self.adata.obs[self.groupby_keys]):
                self.adata.obs[self.groupby_keys] = self.adata.obs[self.groupby_keys].astype("category")
        elif isinstance(self.groupby_keys, list):
            for key in self.groupby_keys:
                if key not in self.adata.obs.columns:
                    raise ValueError(f"Key '{key}' not found in adata.obs.")
                if not pd.api.types.is_categorical_dtype(self.adata.obs[key]):
                    self.adata.obs[key] = self.adata.obs[key].astype("category")

        if isinstance(self.groupby_keys, str):
            self.groupby_keys = [self.groupby_keys]

        # If group_names is not provided, use groupby_keys as default
        if self.group_names is None:
            self.group_names = self.groupby_keys
        else:
            if isinstance(self.group_names, str):
                self.group_names = [self.group_names]

        # Create _group labels
        group_labels = self.adata.obs[self.groupby_keys].astype(str).agg("_".join, axis=1)

        # Build all possible combinations from category levels in order
        category_levels = [self.adata.obs[k].cat.categories for k in self.groupby_keys]
        ordered_combinations = ["_".join(tup) for tup in itertools.product(*category_levels)]

        # Create ordered categorical
        self.group_labels = pd.Categorical(
            group_labels, categories=[x for x in ordered_combinations if x in group_labels.unique()], ordered=True
        )
        self.adata.obs["_group"] = self.group_labels

        # Filter groups based on minimum size
        group_counts = self.adata.obs["_group"].value_counts()
        valid_groups = group_counts[group_counts > self.min_group_size].index

        self.adata = self.adata[self.adata.obs["_group"].isin(valid_groups)].copy()
        self.adata.obs["_group"] = self.adata.obs["_group"].cat.remove_unused_categories()

        self.group_metadata = self.adata.obs.copy()

    def _resolve_row_grouping(self, index_source=None) -> tuple[pd.Categorical | None, pd.Index | None]:
        """
        Resolve row grouping for the heatmap.

        Parameters
        ----------
        index_source : pd.Index or None, optional
            If provided, the grouping will be based on this index. If None, uses the full
            group labels from `adata.obs`.

        Returns
        -------
        tuple[pd.Categorical | None, pd.Index | None]
            A tuple containing:
            - pd.Categorical or None: The resolved row grouping.
            - pd.Index or None: The categories of the grouping.
        """
        if self.row_grouping == "auto":
            group = index_source if index_source is not None else self.group_labels
            return group, getattr(group, "categories", list(group))
        elif self.row_grouping is None:
            return None, None
        elif isinstance(self.row_grouping, str):
            group = (
                self.adata.obs[self.row_grouping].loc[index_source]
                if index_source is not None
                else self.adata.obs[self.row_grouping]
            )
            group = pd.Categorical(group)
            return group, group.categories
        elif isinstance(self.row_grouping, list):
            group = self.adata.obs[self.row_grouping].astype(str).agg("_".join, axis=1)
            group = group.loc[index_source] if index_source is not None else group
            group = pd.Categorical(group)
            return group, group.categories
        elif isinstance(self.row_grouping, pd.Series | pd.Categorical):
            group = (
                self.row_grouping.loc[index_source] if isinstance(self.row_grouping, pd.Series) else self.row_grouping
            )
            group = pd.Categorical(group)
            return group, group.categories
        else:
            raise ValueError("Invalid value for row_grouping")

    def _build_data(self) -> pd.DataFrame:
        """Extracts the data matrix for the selected markers from .X or a specified layer."""
        # Flatten markers if given as a dict
        if isinstance(self.markers, dict):
            flat_markers = list(itertools.chain.from_iterable(self.markers.values()))
        else:
            flat_markers = self.markers

        # Extract matrix from specified layer if provided
        if getattr(self, "layer", None):
            if self.layer not in self.adata.layers:
                raise ValueError(f"Layer '{self.layer}' not found in adata.layers.")
            mat = self.adata[:, flat_markers].layers[self.layer]
        else:
            mat = self.adata[:, flat_markers].X

        # Convert to dense array if sparse
        if hasattr(mat, "toarray"):
            mat = mat.toarray()

        return pd.DataFrame(mat, index=list(self.adata.obs.index), columns=flat_markers)

    def _add_row_labels(self, m: ma.Heatmap):
        """
        Add row labels to the heatmap

        - If self.order is defined, use `Chunk` to show grouped labels.
        - If self.order is None, use `Labels` to show individual row names.

        Parameters
        ----------
        m : ma.Heatmap
            The heatmap object to which row labels will be added.
        """
        if self.row_group is not None:
            # Create chunked row labels using the order
            chunk = mp.Chunk(
                self.order,
                rotation=self.chunk_rotation,
                align=self.chunk_align,
                fontsize=self.chunk_fontsize,
            )
            m.group_rows(self.row_group, order=self.order)
            m.add_left(chunk)
        else:
            # Use index from the data matrix as row labels
            labels = mp.Labels(
                list(self.mean_df.index),  # or m.data.index
                rotation=self.chunk_rotation,
                align=self.chunk_align,
                fontsize=self.chunk_fontsize,
            )
            m.add_left(labels, pad=self.chunk_pad)

    def _add_column_labels(self, m: ma.Heatmap):
        """
        Add column labels to the heatmap.

        - If `self.markers` is a dict, create grouped chunks using keys as categories.
        - Otherwise, show all markers using `mp.Labels`.

        Parameters
        ----------
        m : ma.Heatmap
            The heatmap object to which column labels will be added.
        """
        if isinstance(self.markers, dict):
            # Build matching group labels for each column
            chunk_labels = list(itertools.chain.from_iterable([key] * len(vals) for key, vals in self.markers.items()))

            # Create Categorical with explicit order
            group_labels = pd.Categorical(chunk_labels, categories=list(self.markers.keys()), ordered=True)

            # Apply column grouping
            m.group_cols(group_labels, order=list(self.markers.keys()))

            # Add chunked column annotations
            chunk = mp.Chunk(
                list(self.markers.keys()),
                rotation=90,
                align=self.chunk_align,
                fontsize=self.chunk_fontsize,
            )

            m.add_top(chunk)
        else:
            # Simple unchunked label case
            labels = mp.Labels(self.markers, fontsize=self.chunk_fontsize)
            m.add_top(labels, pad=self.chunk_pad)

    def _add_annotations(self, m: ma.Heatmap):
        """
        Add group colorbars and labels to the heatmap.

        Parameters
        ----------
        m : ma.Heatmap
            The heatmap object to which annotations will be added.
        """
        # Add an annotation colorbar for each group key
        for key in reversed(self.groupby_keys):
            self._add_group_colorbar(m, key)
        # Add column and gene labels if specified
        if self.show_column_names:
            self._add_column_labels(m)
        if self.show_row_names:
            self._add_row_labels(m)

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
        # Check if the key exists in adata.obs
        if key not in self.adata.obs:
            raise ValueError(f"Key '{key}' not found in adata.obs. Available keys: {self.adata.obs.columns.tolist()}")
        # Extract values and palette for the group key
        values = self.adata.obs[key]
        palette = self.adata.uns.get(f"{key}_colors")
        # If no palette is found, use a default palette
        if palette is None:
            categories = values.cat.categories
            palette = dict(zip(categories, default_palette(len(categories)), strict=False))
        # Create and add the colorbar
        label = self.group_names[self.groupby_keys.index(key)]
        colorbar = mp.Colors(list(values), palette=palette, label=label)
        m.add_left(colorbar, size=self.groupbar_size, pad=self.groupbar_pad)

    def _add_extras(self, m):
        """
        Add additional features to the heatmap.

        Parameters
        ----------
        m : ma.Heatmap
            The heatmap object to which extras will be added.
        """
        # Add categorical annotations
        self._add_annotations(m)
        # Add dendrograms if specified
        if self.dendrograms:
            for pos in self.dendrograms:
                m.add_dendrogram(pos, add_base=False)
        # Add legends if specified
        if self.show_legends:
            m.add_legends()
        return m

    def _scale_data(self, data: pd.DataFrame, scaling_keys: list = None) -> pd.DataFrame:
        """
        Conditionally apply min-max scaling to the data.

        Parameters
        ----------
        data : pd.DataFrame
            Data matrix (rows indexed by _group, columns are features like genes).
        scaling_keys : list of str, optional
            - If None: no scaling is applied.
            - If empty list ([]): global scaling is applied.
            - If list of metadata keys: scaling is done per group defined by those keys.

        Returns
        -------
        pd.DataFrame
            Scaled data with same shape and index/columns as input.
        """
        # Case 1: No scaling requested
        if scaling_keys is None:
            return data.copy()

        # Ensure the metadata and data are aligned
        if not set(scaling_keys).issubset(self.group_metadata.columns):
            raise ValueError(f"Scaling keys {scaling_keys} not found in group metadata columns.")

        df_metadata = self.group_metadata.copy()

        # Case 2: Global scaling
        if len(scaling_keys) == 0:
            data_min = data.min(axis=0)
            data_max = data.max(axis=0)
            return (data - data_min) / (data_max - data_min).replace(0, 1e-8)

        # Case 3: Group-wise scaling
        scaled_data = pd.DataFrame(index=data.index, columns=data.columns, dtype=float)
        groupby_obj = df_metadata.groupby(scaling_keys)

        for _, idx in groupby_obj.groups.items():
            group_data = data.loc[idx]
            data_min = group_data.min(axis=0)
            data_max = group_data.max(axis=0)
            scaled_group = (group_data - data_min) / (data_max - data_min).replace(0, 1e-8)
            scaled_data.loc[idx] = scaled_group

        return scaled_data

    def _build_plot(self):
        """Build the base heatmap plot."""
        # Build the data matrix for the heatmap
        data = self._build_data()

        # Scale the data if scaling is enabled
        data = self._scale_data(data, self.scaling_keys)

        # Create heatmap
        m = ma.Heatmap(
            data,
            cmap=self.cmap,
            height=self.height,
            width=self.width,
            cbar_kws={"title": "Expression\nin group"},
        )
        # Extract grouping information
        self.row_group, self.order = self._resolve_row_grouping()
        # Add extras to the heatmap
        m = self._add_extras(m)
        return m

    def show(self):
        """Display the plot using matplotlib's interactive mode."""
        # Build the plot
        m = self._build_plot()
        # Render the plot
        with plt.rc_context(rc={"axes.grid": False, "grid.color": ".8"}):
            m.render(scale=self.scale_render)

    def save(self, filename: str, **kwargs):
        """
        Save the plot to a file.

        Parameters
        ----------
        filename : str
            The path to save the plot.
        **kwargs
            Additional keyword arguments passed to the save method.
        """
        # Build the plot
        m = self._build_plot()
        # Render the plot
        with plt.rc_context(rc={"axes.grid": False, "grid.color": ".8"}):
            m.save(filename, **kwargs)
