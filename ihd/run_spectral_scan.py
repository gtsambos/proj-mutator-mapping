import pandas as pd
import numpy as np
import argparse
import json
import sys
from utils import (
    calculate_covariate_by_marker,
    compute_spectra,
    compute_manual_chisquare,
    compute_manual_cosine_distance,
    get_covariate_matrix,
    get_sample_sizes,
    perform_spectral_scan,
    perform_wide_permutation_test,
)
from schema import MutationSchema
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


def adjust_row(row): # ensure C>A is always positive
    if row[3] < 0:
        return row * -1
    else:
        return row


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

    # Create a list showing each sample's number of inbreeding generations
    generations = []
    for s in samples:
        col_vals = mutations['sample'] == s
        col_vals = col_vals.values
        sample_exists = any(col_vals)
        if sample_exists:
            row_index = mutations[mutations['sample'] == s].index[0]
            row = mutations.iloc[row_index]
            generations.append(int(row["n_generations"]))
        else:
            print(f"Sample {s} not found in count file.")
            sys.exit()

    # print(generations)

    # Make a scatterplot with the number of generations on the x-axis and the sum of each row of spectra on the y-axis
    # import matplotlib.pyplot as plt
    # plt.scatter(generations, np.sum(spectra, axis=1), s=1)
    # fit a line to the data
    # m, b = np.polyfit(generations, np.sum(spectra, axis=1), 1)
    # plt.plot(generations, m*np.array(generations) + b)
    # plt.xlabel("Number of generations")
    # plt.ylabel("total mutation count")
    # plt.title("Total mutation count vs number of generations")
    # plt.show()

    # diff = [(m*x + b)/y for x, y in zip(generations, np.sum(spectra, axis=1))]
    # plt.scatter(generations, diff, s=1)
    # # add a line at y=1
    # plt.axhline(y=1, color='gray', linestyle='--')
    # plt.xlabel("Number of generations")
    # plt.ylabel("ratio between actual and expected mutation count")
    # plt.show()


    # count the correlation between the sum of each row of spectra and the number of generations
    # print("Before adjustment")
    # print(np.corrcoef(np.sum(spectra, axis=1), generations))
    # print(np.sum(spectra, axis=1).shape)

    # Adjust spectra by the number of inbreeding generations.
    # This is because the number of generations is likely to be correlated with the number of mutations.
    
    
    for i in range(len(spectra)):
        spectra[i] = spectra[i] / generations[i]

    # print("After adjustment")
    # print(np.corrcoef(np.sum(spectra, axis=1), generations))
    # print(np.sum(spectra, axis=1).shape)


    # Define strata
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

    # If args.shuffle is True, shuffle the columns of the genotype matrix
    # (note: it's easier to do this on the rows, using np.random.shuffle)
    if args.shuffle:
        geno_asint_filtered_matrix = geno_asint_filtered_matrix.T
        np.random.shuffle(geno_asint_filtered_matrix)
        geno_asint_filtered_matrix = geno_asint_filtered_matrix.T

        # save the shuffled matrix to disk.
        assert args.shuffle_file is not None
        np.savez(args.shuffle_file, geno_asint_filtered_matrix)

    # genotype_similarity = compute_genotype_similarity(geno_asint_filtered_matrix)
    genotype_similarity = np.ones(geno_asint_filtered_matrix.shape[0])
    # I don't understand why Tom has used all ones here?

    # Also print out these -- don't know what they are doing
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


    distance_method = compute_manual_cosine_distance
    if args.distance_method == "chisquare":
        distance_method = compute_manual_chisquare

    # compute overall mutation spectra between mice with B and D alleles
    # at each locus

    out_a, out_b = perform_spectral_scan(
        spectra,
        geno_asint_filtered_matrix,
        distance_method=distance_method,
    )

    # out_a = pd.DataFrame(out_a)
    # out_b = pd.DataFrame(out_b)

    # # turn these into frequencies
    # out_a['sum'] = out_a.sum(axis=1)
    # out_a = out_a.div(out_a['sum'], axis=0)
    # # out_a2 = out_a.div(out_a['sum'], axis=0)

    # out_b['sum'] = out_b.sum(axis=1)
    # out_b = out_b.div(out_b['sum'], axis=0)
    # # out_b2 = out_b.div(out_b['sum'], axis=0)

    # calculate the difference between the two
    # spectral_diff = out_a - out_b
    # spectral_diff.drop(columns=['sum'], inplace=True)

    # # Adjust differences so that the first (A>C) mutation frequency is always positive.
    # # This is because whether a signature is associated with the Bs or Ds is arbitrary.
    # # print("applying adjustment")
    # spectral_diff = spectral_diff.apply(adjust_row, axis=1)


    # get total generation times for each group
    sample_sizes = get_sample_sizes(geno_asint_filtered_matrix, generations)

    # adjust overall mutation spectra by total generation times
    # print(out_a.shape, "adj_a shape")   
    # print(sample_sizes.shape, "sample_sizes shape")

    adj_a = np.zeros(out_a.shape)
    adj_b = np.zeros(out_b.shape)
    for i in range(out_a.shape[0]):
        # print(out_a[i].shape, "out_a[i] shape")
        # print(sample_sizes[i, 0], "sample_sizes[i, 0] shape")
        adj_a[i] = out_a[i] / sample_sizes[i, 0]
        adj_b[i] = out_b[i] / sample_sizes[i, 1]

    # # adjust the columns to have mean 0 and variance 1
    # all_vals = np.concatenate((adj_a, adj_b), axis=0)
    # all_means = np.mean(all_vals, axis=0)
    # all_stds = np.std(all_vals, axis=0)
    # adj_a = (adj_a - all_means) / all_stds
    # adj_b = (adj_b - all_means) / all_stds

    # calculate spectral differences between groups
    spectral_diff = adj_a - adj_b

    # # Normalise these differences so that each row sums to 1. (Use absolute values)
    # spectral_diff_abs = np.abs(spectral_diff)
    # spectral_diff_norm = spectral_diff / np.sum(spectral_diff_abs, axis=1)[:, np.newaxis]

    # Permutation tests
    if args.shuffle is False:

        null_distances = perform_wide_permutation_test(
            spectra,
            geno_asint_filtered_matrix,
            genotype_similarity,
            covariate_ratios,
            strata,
            distance_method=distance_method,
            n_permutations=args.permutations,
            progress=args.progress,
            adjust_statistics=False,
        )

        # for the moment, save this output to disk (but automate it later) with numpy.save
        np.save("data/spectral-differences/bxd.spd.permutations.npy", null_distances)
        

        # Saving output

        # convert from numpy arrays to pandas dataframes and save output
        out_a_df = pd.DataFrame(adj_a)
        out_b_df = pd.DataFrame(adj_b)

        out_a_df.to_csv("data/spectral-differences/bxd.spd.a.csv", index=False)
        out_b_df.to_csv("data/spectral-differences/bxd.spd.b.csv", index=False)

        # spectral_diff_df = pd.DataFrame(spectral_diff)
        # spectral_diff_df.to_csv(args.norm, index=False)

    spectral_diff_df = pd.DataFrame(spectral_diff)
    spectral_diff_df.to_csv(args.norm, index=False)




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
        "--count_file",
        type=str,
        help="Path to file containing sample metadata (e.g., number of inbreeding generations).",
    )
    # p.add_argument(
    #     "--diff",
    #     help="Path in which to store the spectral differences between groups.",
    # )
    p.add_argument(
        "--norm",
        help="Path in which to store the normalised spectral differences between groups.",
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
    p.add_argument(
        "-shuffle",
        default=False,
        action="store_true",
        help="""Whether to shuffle the columns of the genotype matrix before performing the spectral scan. Default is False.""",
    )
    p.add_argument(
        '-shuffle_file',
        type=str,
        default=None,
        help="""Path to output for shuffled genotypes. Required if shuffle is True.""",
    )
    args = p.parse_args()

    main(args)
