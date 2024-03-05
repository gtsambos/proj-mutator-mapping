import pandas as pd
import numpy as np
import argparse
import json
import sys
from utils import (
    compute_spectra,
    compute_manual_chisquare,
    compute_manual_cosine_distance,
    get_covariate_matrix,
    calculate_covariate_by_marker,
    get_sample_sizes,
    perform_spectral_scan,
)
from schema import IHDResultSchema, MutationSchema
import numba


def filter_mutation_data(
    mutations: pd.DataFrame,
    geno: pd.DataFrame,
) -> pd.DataFrame:
    # get unique samples in mutation dataframe
    samples = mutations["sample"].unique()
    # get the overlap between those and the sample names in the genotype data
    samples_overlap = list(set(samples).intersection(set(geno.columns)))

    if len(samples_overlap) == 0:
        print(
            """Sorry, no samples in common between mutation data
        and genotype matrix. Please ensure sample names are identical."""
        )
        sys.exit()

    # then subset the genotype and mutation data to include only those samples
    cols2use = ["marker"]
    cols2use.extend(samples_overlap)
    geno = geno[cols2use]

    mutations_filtered = mutations[mutations["sample"].isin(samples_overlap)]
    return mutations_filtered


def main(args):
    # read in JSON file with file paths
    config_dict = None
    with open(args.config, "rb") as config:
        config_dict = json.load(config)

    # read in genotype info
    geno = pd.read_csv(config_dict["geno"])
    markers = pd.read_csv(config_dict["markers"])

    markers2use = markers[markers["chromosome"] != "X"]["marker"].unique()
    geno = geno[geno["marker"].isin(markers2use)]

    # read in singleton data and validate with pandera
    mutations = pd.read_csv(args.mutations, dtype={"sample": str})
    MutationSchema.validate(mutations)

    mutations_filtered = filter_mutation_data(mutations, geno)

    # get a list of samples and their corresponding mutation spectra
    samples, mutation_types, spectra = compute_spectra(
        mutations_filtered,
        k=args.k,
        cpg=True,
    )
    print(
        f"""Using {len(samples)} samples
          and {int(np.sum(spectra))} total mutations."""
    )

    strata = np.ones(len(samples))
    if args.stratify_column is not None:
        sample2strata = dict(
            zip(
                mutations_filtered["sample"],
                mutations_filtered[args.stratify_column],
            )
        )
        strata = np.array([sample2strata[s] for s in samples])

    callable_kmer_arr = None
    if args.callable_kmers and args.k == 1:
        callable_kmer_arr = np.zeros(
            (len(samples), len(mutation_types)), dtype=np.int64
        )
        callable_kmers = pd.read_csv(args.callable_kmers)
        # NOTE: need to check schema of df
        for si, s in enumerate(samples):
            for mi, m in enumerate(mutation_types):
                base_nuc = m.split(">")[0] if m != "CpG>TpG" else "C"
                callable_k = callable_kmers[
                    (callable_kmers["sample"] == s)
                    & (callable_kmers["nucleotide"] == base_nuc)
                ]
                callable_kmer_arr[si, mi] = callable_k["count"].values[0]

    # convert string genotypes to integers based on config definition
    replace_dict = config_dict["genotypes"]
    geno_asint = geno.replace(replace_dict).replace({1: np.nan})

    if args.adj_region is not None:
        chrom = args.adj_region.split(":")[0]
        start, end = list(map(float, args.adj_region.split(":")[1].split("-")))
        # find markers within this region
        markers_to_filter = markers[
            (markers["chromosome"] == chrom)
            & (markers["Mb"] >= start)
            & (markers["Mb"] <= end)
        ]["marker"].unique()
        marker_idxs = geno_asint[
            geno_asint["marker"].isin(markers_to_filter)
        ].index.values
        geno_asint = geno_asint.iloc[geno_asint.index.difference(marker_idxs)]

    # convert genotype values to a matrix
    geno_asint_filtered_matrix = geno_asint[samples].values
    # get an array of marker names at the filtered genotyped loci
    markers_filtered = geno_asint["marker"].values

    # compute similarity between allele frequencies at each marker
    # genotype_similarity = compute_genotype_similarity(geno_asint_filtered_matrix)
    genotype_similarity = np.ones(geno_asint_filtered_matrix.shape[0])
    distance_method = compute_manual_cosine_distance
    if args.distance_method == "chisquare":
        distance_method = compute_manual_chisquare

    covariate_cols = []
    covariate_matrix = get_covariate_matrix(
        mutations_filtered,
        samples,
        covariate_cols=covariate_cols,
    )

    covariate_ratios = calculate_covariate_by_marker(
        covariate_matrix,
        geno_asint_filtered_matrix,
    )

    # compute overall mutation spectra between mice with B and D alleles
    # at each locus
    out_a, out_b = perform_spectral_scan(
        spectra,
        geno_asint_filtered_matrix,
        distance_method=distance_method,
    )

    # get sample sizes for each group
    sample_sizes = get_sample_sizes(geno_asint_filtered_matrix)
                                    
    # adjust overall mutation spectra by sample size
    adj_a = np.zeros(out_a.shape)
    adj_b = np.zeros(out_b.shape)
    for i in range(out_a.shape[0]):
        adj_a[i] = out_a[i] / sample_sizes[i, 0]
        adj_b[i] = out_b[i] / sample_sizes[i, 1]


    # convert from numpy arrays to pandas dataframes and save output
    out_a_df = pd.DataFrame(out_a)
    out_b_df = pd.DataFrame(out_b)

    out_a_df.to_csv(args.out_a, index=False)
    out_b_df.to_csv(args.out_b, index=False)

    adj_a_df = pd.DataFrame(adj_a)
    adj_b_df = pd.DataFrame(adj_b)

    adj_a_df.to_csv(args.adj_a, index=False)
    adj_b_df.to_csv(args.adj_b, index=False)

    sample_sizes_df = pd.DataFrame(sample_sizes, columns=["a", "b"])
    sample_sizes_df.to_csv(args.sample_sizes, index=False)



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mutations",
        type=str,
        help="Path to mutation data in CSV format.",
    )
    p.add_argument(
        "--config",
        type=str,
        help="Path to config file in JSON format.",
    )
    p.add_argument(
        "--out_a",
        help="Path in which to store the results of the spectral scan for set a.",
    )
    p.add_argument(
        "--out_b",
        help="Path in which to store the results of the spectral scan for set b.",
    )
    p.add_argument(
        "--adj_a",
        help="Path in which to store the adjusted results of the spectral scan for set a.",
    )
    p.add_argument(
        "--adj_b",
        help="Path in which to store the adjusted results of the spectral scan for set b.",
    )
    p.add_argument(
        "--sample_sizes",
        help="Path in which to store the sample sizes for each group.",
    )
    p.add_argument(
        "-k",
        type=int,
        default=1,
        help="k-mer context used to classify mutations. Default is 1.",
    )
    p.add_argument(
        "-permutations",
        type=int,
        default=1_000,
        help="Number of permutations to perform when calculating significance thresholds. Default is 1,000.",
    )
    p.add_argument(
        "-distance_method",
        default="cosine",
        type=str,
        help="""Method to use for calculating distance between aggregate spectra. Options are 'cosine' and 'chisquare', default is 'chisquare'.""",
    )
    p.add_argument(
        "-threads",
        default=1,
        type=int,
        help="""Number of threads to use during permutation testing step. Default is 1.""",
    )
    p.add_argument(
        "-progress",
        action="store_true",
        help="""Whether to output the progress of the permutation testing step (i.e., the number of completed permutations).""",
    )
    p.add_argument(
        "-callable_kmers",
        default=None,
        type=str,
        help="""Path to CSV file containing numbers of callable base pairs in each sample, stratified by nucleotide.""",
    )
    p.add_argument(
        "-stratify_column",
        default=None,
        type=str,
        help="""If specified, use this column to perform a stratified permutation test by only permuting BXDs within groups defined by the column to account for population structure.""",
    )
    p.add_argument(
        "-adj_region",
        default=None,
        type=str,
        help="""If specified, a chromosomal region (chr:start-end) that we should adjust for in our AMSD scans. Start and end coordinates should be specified in Mb. Default is None""",
    )
    args = p.parse_args()

    main(args)
