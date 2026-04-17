from collections.abc import Sequence
from typing import Any

import gseapy as gp
import pandas as pd

from delnx.ds._gmt import get_gene_sets

MIN_GENESET_SIZE = 5
MAX_GENESET_SIZE = 500


def gsea(
    genes: Sequence[str],
    background: Sequence[str] | None = None,
    gene_sets: dict[str, list[str]] | None = None,
    collection: str = "all",
    url: str | None = None,
    filepath: str | None = None,
    geneset_key: str = "geneset",
    genesymbol_key: str = "genesymbol",
    method: str = "enrichr",
    return_object: bool = False,
    min_genes: int = MIN_GENESET_SIZE,
    max_genes: int = MAX_GENESET_SIZE,
) -> pd.DataFrame | Any:
    """
    Run enrichment analysis for a single gene list using Enrichr.

    Parameters
    ----------
    genes : Sequence[str]
        List of gene symbols to analyze.
    background : Sequence[str], optional
        Background gene list to use for enrichment analysis. If None, uses all genes in the gene sets.
    gene_sets : dict[str, list[str]], optional
        Pre-loaded gene sets as a dictionary where keys are gene set names and values are lists of gene symbols.
        If None, will load gene sets based on the provided collection, URL, or filepath.
    collection : str
        Name of the collection to load gene sets from. Default is "all".
    url : str, optional
        URL to load the GMT file from. If None, uses the default collection.
    filepath : str, optional
        Local file path to load the GMT file from. If None, uses the default collection.
    geneset_key : str
        Column name for the gene set name in the output dictionary. Default is "geneset".
    genesymbol_key : str
        Column name for the gene symbol in the output dictionary. Default is "genesymbol".
    method : str
        Method to use for enrichment analysis. Currently only "enrichr" is supported.
    return_object : bool
        If True, returns the gseapy Enrichr object. If False, returns a pandas DataFrame with results.
    min_genes : int
        Minimum number of genes in a gene set to include in the analysis. Default is 5.
    max_genes : int
        Maximum number of genes in a gene set to include in the analysis. Default is 500.
    """
    if method != "enrichr":
        raise ValueError(f"Unsupported method: {method}")

    if gene_sets is None:
        gene_sets = get_gene_sets(
            collection=collection,
            url=url,
            filepath=filepath,
            geneset_key=geneset_key,
            genesymbol_key=genesymbol_key,
            min_genes=min_genes,
            max_genes=max_genes,
        )

    enr = gp.enrichr(
        gene_list=list(genes),
        background=list(background) if background is not None else None,
        gene_sets=gene_sets,
        outdir=None,
        no_plot=True,
    )

    return enr if return_object else enr.res2d


