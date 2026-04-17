import marsilea as ma
import matplotlib.pyplot as plt
import numpy as np

import delnx as dx


def test_plot_volcanoplot(adata_pb_counts):
    """Test plotting of volcanoplot."""

    def nb_de(adata, **kw):
        fit = dx.tl.nb_fit(adata, **kw)
        return dx.tl.nb_test(adata, fit)

    # Run grouped NB DE analysis
    results = dx.tl.grouped(
        nb_de,
        adata_pb_counts,
        group_key="cell_type",
        condition_key="condition",
        reference="control",
    )

    # Select one cell type at a time
    results = results[results["group"] == "cell_type_1"]

    # Plot volcano plot
    fig, _ = dx.pl.volcanoplot(
        results, thresh={"coef": 2, "-log10(pval)": -np.log10(0.05)}, label_top=5, return_fig=True, show=False
    )

    # Check if the figure is created
    assert fig is not None
    assert isinstance(fig, plt.Figure)


def test_plot_violinplot(adata_small):
    """Test plotting of violinplot."""

    plot = dx.pl.ViolinPlot(
        adata_small,
        genes=["gene_1", "gene_2", "gene_3"],
        groupby="cell_type",
        splitby="condition",
        include_groups=["cell_type_1"],
        include_splits=["ctr", "treatment"],
        figsize=(2, 4),
        use_raw=False,
        flip=True,
    )
    board = plot._build_plot(plot_type="violin")
    # Check if the figure is created
    assert board is not None
    assert isinstance(board, ma.base.StackBoard)


def test_matrixplot(adata_small):
    """Test plotting of matrixplot."""

    m = dx.pl.MatrixPlot(
        adata_small,
        markers=["gene_1", "gene_2", "gene_3"],
        groupby_keys=["cell_type", "condition"],
        group_names=["Cell type", "Condition"],
        row_grouping=["cell_type", "condition"],
        column_grouping=True,
        dendrograms=["left"],
        cmap="RdPu",
        scale=True,
        width=10,
        height=10,
        show_legends=True,
        show_column_names=False,
        show_row_names=True,
    )
    fig = m._build_plot()
    # Check if the figure is created
    assert fig is not None
    assert isinstance(fig, ma.heatmap.Heatmap)


def test_dotplot(adata_small):
    """Test plotting of matrixplot."""

    m = dx.pl.DotPlot(
        adata_small,
        markers=["gene_1", "gene_2", "gene_3"],
        groupby_keys=["cell_type", "condition"],
        group_names=["Cell type", "Condition"],
        row_grouping=["cell_type", "condition"],
        column_grouping=True,
        dendrograms=["left"],
        cmap="RdPu",
        width=10,
        height=10,
        scale=True,
        show_legends=True,
        show_column_names=False,
        show_row_names=True,
    )
    fig = m._build_plot()
    # Check if the figure is created
    assert fig is not None
    assert isinstance(fig, ma.heatmap.SizedHeatmap)
