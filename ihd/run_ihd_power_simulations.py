import numpy as np
import pandas as pd
import numba
from typing import List, Tuple, Callable
from utils import (
    perform_ihd_scan,
    perform_permutation_test,
    compute_genotype_similarity,
    compute_manual_chisquare,
    compute_manual_cosine_distance,
)
import argparse

@numba.njit
def simulate_genotypes(
    n_markers: int,
    n_haplotypes: int,
    exp_af: float = 0.5,
) -> np.ndarray:
    """Simulate genotypes on haplotypes. 

    Args:
        n_markers (int): Number of sites at which to simulate genotypes.
        
        n_haplotypes (int): Number of haplotypes to simulate.

        exp_af (float, optional): Expected frequency of ALT allele at each site. Defaults to 0.5.

    Returns:
        np.ndarray: Matrix of size (g, n), where g is the number of markers \
            and n is the number of haplotypes.
    """
    genotype_matrix = np.zeros((n_markers, n_haplotypes))

    # simulate genotypes at a specific number of markers.
    # for each haplotype, take a random draw from a uniform
    # distribution at each marker. if that draw is <= the
    # expected marker allele frequency, assign that marker's
    # genotype to be 1. otherwise, keep it at 0.
    for haplotype_idx in np.arange(n_haplotypes):
        genotype_probs = np.random.random(n_markers)
        # convert probs to genotypes
        alt_alleles = np.where(genotype_probs <= exp_af)[0]
        genotype_matrix[alt_alleles, haplotype_idx] = 2
    # get sums of alt alleles at each marker
    alt_counts = np.sum(genotype_matrix, axis=1)
    # if there are any indices where no samples have alt alleles,
    # add a handful artificially
    zero_ac_sites = np.where(alt_counts == 0)[0]

    if zero_ac_sites.shape[0] == 0:
        return genotype_matrix
    else:
        for idx in zero_ac_sites:
            # pick a random set of haplotypes
            random_hap_idxs = np.random.choice(
                np.arange(n_haplotypes),
                size=int(n_haplotypes * exp_af),
                replace=False,
            )
            for hap_idx in random_hap_idxs:
                genotype_matrix[idx, hap_idx] = 2
        return genotype_matrix


@numba.njit
def poisson_from_arr(lambdas: np.ndarray) -> np.ndarray:
    """Take Poisson draws from an array of lambda values
     that can be multi-dimensional. Numba doesn't allow
     lambdas to be arrays in np.random.poisson, so we have
     to wrap it up and njit it.

    Args:
        lambdas (np.ndarray): Any-dimensional array of lambdas.

    Returns:
        np.ndarray: Any-dimensional array of counts, of the same \
            shape as the input lambda array.
    """
    output_arr = np.zeros(lambdas.shape)
    for i in np.arange(lambdas.shape[0]):
        for j in np.arange(lambdas.shape[1]):
            sample = np.random.poisson(lambdas[i, j])
            output_arr[i, j] = sample
    return output_arr

#@numba.njit
def run_simulation_trials(
    base_lambdas: np.ndarray,
    mutation_type_idx: int = 0,
    effect_size: float = 1.1,
    n_permutations: int = 100,
    n_mutations: int = 50,
    n_haplotypes: int = 100,
    n_markers: int = 1000,
    number_of_trials: int = 100,
    exp_af: float = 0.5,
    distance_method: Callable = compute_manual_chisquare,
    vary_mutation_counts: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a selected number of simulation trials.

    Args:
        base_lambdas (np.ndarray): 1D numpy array of lambda values corresponding to the \
            expected number of each k-mer mutation type we observe on a haplotype.

        mutation_type_idx (int, optional): Index of the mutation type to augment by the specified \
            effect size. Defaults to 0.

        effect_size (float, optional): Effect size of the mutator allele. Defaults to 1.1.

        n_permutations (int, optional): Number of permutations to use when establishing distance \
            thresholds for significance. Defaults to 100.

        n_mutations (int, optional): Total number of mutations to simulate on each haplotype. Defaults to 50.

        n_haplotypes (int, optional): Number of haplotypes to simulate. Defaults to 100.

        n_markers (int, optional): Number of sites at which to simulate genotypes. Defaults to 1000.

        number_of_trials (int, optional): Number of trials to run for every combination of \
            simulation parameters. Defaults to 100.

        exp_af (float, optional): The fraction of haplotypes at the \
            simulated mutator locus that actually possess the mutator allele and augmented mutation \
            spectra. Defaults to 0.5.

        distance_method (Callable, optional): Distance metric to use in AMSD scan. Defaults to cosine.

        vary_mutation_counts (bool, optional): Whether to allow mutation counts on haplotypes to vary by \
            a factor of 20. Defaults to False.

    Returns:
        pvals (np.ndarray): 1D numpy array of p-values from `number_of_trials` trials.

        mutation_spectra_in_trials (np.ndarray): 3D numpy array of size (T, H, M), where T \
            is the number of trials, H is the number of haplotypes, and M is the number of \
            k-mer mutation types. Contains simulated mutation spectra for every haplotype.

        genotype_matrices_in_trials (np.ndarray): 3D numpy array of size (T, H, G), where T \
            is the number of trials, H is the number of haplotypes, and G is the number of \
            genotyped sites. Contains simulated genotypes at each marker in each trial.
        
        focal_markers (np.ndarray): 1D numpy array containing the marker at which we simulate \
            the mutator allele in every trial.
    """

    pvals = np.ones(number_of_trials)
    mutation_spectra_in_trials = np.zeros((number_of_trials, n_haplotypes, base_lambdas.shape[0]))
    genotype_matrices_in_trials = np.zeros((number_of_trials, n_markers, n_haplotypes))
    focal_markers = np.zeros(number_of_trials)

    if vary_mutation_counts:
        mutation_counts = np.linspace(n_mutations, 20 * n_mutations, n_haplotypes).reshape(-1, 1)
    else:
        mutation_counts = np.repeat(n_mutations, n_haplotypes).reshape(-1, 1)

    for trial_n in numba.prange(number_of_trials):

        genotype_matrix = simulate_genotypes(n_markers, n_haplotypes, exp_af = 0.5)

        # create a matrix of lambda values
        lambdas = np.broadcast_to(
            base_lambdas * mutation_counts,
            (
                n_haplotypes,
                base_lambdas.shape[0],
            ),
        ).copy()

        # pick a marker at which we'll artificially put an
        # "association" between genotypes and mutation spectra.
        focal_marker = np.random.randint(0, n_markers)
        focal_markers[trial_n] = focal_marker
        # at that marker, get the haplotype indices with alt alleles
        alt_haplotypes = np.where(genotype_matrix[focal_marker] == 2)[0]

        # figure out how many of the haplotypes at the "mutator locus"
        # actually carry the mutator allele. we may want the allele frequency
        # of the mutator allele to be less than 0.5
        n_true_alt_haplotypes = int(n_haplotypes * exp_af)
        alt_haplotypes = np.random.choice(np.arange(n_haplotypes), n_true_alt_haplotypes, replace=False)

        # TEST: if the mutator is present at lower frequency, bump
        # the genotypes to reflect it
        genotype_matrix[focal_marker, :] = 0
        genotype_matrix[focal_marker, alt_haplotypes] = 2
        genotype_matrices_in_trials[trial_n] = genotype_matrix

        # ensure that at least one haplotype at this site has the
        # alternate allele. otherwise, return a p-value of 1.
        if alt_haplotypes.shape[0] < 1:
            #print ("NO ALT HAPLOTYPES", n_true_alt_haplotypes, f_with_mutator, alt_haplotypes, genotype_matrix[focal_marker], np.sum(genotype_matrix, axis=1))
            pvals[trial_n] = 1.
            continue
        # augment the lambda on the alt haplotypes by an effect size
        for ai in alt_haplotypes:
            lambdas[ai, mutation_type_idx] *= effect_size
        # simulate mutation spectra using new lambdas
        mutation_spectra = poisson_from_arr(lambdas)
        mutation_spectra_in_trials[trial_n] = mutation_spectra

        genotype_similarity = compute_genotype_similarity(genotype_matrix)
        covariate_ratios = np.ones(genotype_similarity.shape[0]).reshape(-1, 1)

        # run an IHD scan
        focal_dists = perform_ihd_scan(
            mutation_spectra,
            genotype_matrix,
            genotype_similarity,
            covariate_ratios,
            adjust_statistics=False,
            distance_method=distance_method,
        )

        # and get null
        null_distances = perform_permutation_test(
            mutation_spectra,
            genotype_matrix,
            genotype_similarity,
            covariate_ratios,
            strata=np.ones(mutation_spectra.shape[0]),
            n_permutations=n_permutations,
            distance_method=distance_method,
            adjust_statistics=False,
        )
        thresh = np.percentile(null_distances, q=95)
        pvals[trial_n] = 1 - (focal_dists[focal_marker] > thresh)

    return pvals, mutation_spectra_in_trials, genotype_matrices_in_trials, focal_markers


def main(args):

    base_mutations = ["C>T", "CpG>TpG", "C>A", "C>G", "A>T", "A>C", "A>G"]
    base_lambdas = np.array([0.29, 0.17, 0.12, 0.075, 0.1, 0.075, 0.17])

    nucs = ["A", "T", "C", "G"]

    distance_method = compute_manual_cosine_distance
    if args.distance_method == "cosine": distance_method = compute_manual_cosine_distance
    else: distance_method = compute_manual_chisquare

    kmer_size = len(args.mutation_type.split('_')[0])

    if kmer_size == 1:
        mutations, lambdas = base_mutations.copy(), base_lambdas.copy()
    else:
        # if we want to do the 3-mer spectrum, we'll just parcel out
        # mutation probabilities equally to every 3-mer associated with
        # a particular "base" mutation type
        mutations, lambdas = [], []
        for m, l in zip(base_mutations, base_lambdas):
            if m == "CpG>TpG": continue
            per_k_l = l / 16
            if m == "C>T":
                per_k_l = 0.46 / 16
            orig, new = m.split('>')
            for fp in nucs:
                for tp in nucs:
                    kmer = f"{fp}{orig}{tp}>{fp}{new}{tp}"
                    mutations.append(kmer)
                    lambdas.append(per_k_l)
        lambdas = np.array(lambdas)


    mut2idx = dict(zip(mutations, range(len(lambdas))))

    res_df = []

    mutation_type_idx = mut2idx[args.mutation_type.replace("_", ">")]
    
    trial_pvals, trial_mutation_spectra, trial_genotype_matrices, focal_markers = run_simulation_trials(
        lambdas,
        mutation_type_idx=mutation_type_idx,
        effect_size=args.effect_size / 100.,
        n_permutations=args.n_permutations,
        n_mutations=args.n_mutations,
        n_haplotypes=args.n_haplotypes,
        n_markers=args.n_markers,
        number_of_trials=args.n_trials,
        exp_af=args.exp_af / 100.,
        distance_method=distance_method,
        vary_mutation_counts=False,
    )

    # if desired, output the genotype matrix to a CSV
    if args.raw_geno is not None:
        genotype_matrix_dfs = []
        for trial_n in range(args.n_trials):
            columns = [f"S{i}" for i in range(args.n_haplotypes)]
            genotype_matrix_df = pd.DataFrame(trial_genotype_matrices[trial_n], columns = columns)
            genotype_matrix_df.replace({0: "B", 2: "D"}, inplace=True)
            genotype_matrix_df["marker"] = [f"rs{i}" for i in range(args.n_markers)]
            genotype_matrix_df["trial"] = trial_n
            # swap column order
            new_columns = ["marker", "trial"]
            new_columns.extend(columns)
            genotype_matrix_dfs.append(genotype_matrix_df[new_columns])
        pd.concat(genotype_matrix_dfs).to_csv(args.raw_geno, index=False)

    # if desired, output the simulated mutation spectra
    if args.raw_spectra is not None:
        raw_spectra_df = []
        for trial_n in range(args.n_trials):
            trial_spectra_df = pd.DataFrame(
                trial_mutation_spectra[trial_n, :, :],
                columns=mutations,
            )
            trial_spectra_df["trial"] = trial_n
            trial_spectra_df["focal_marker"] = focal_markers[trial_n]
            trial_spectra_df["sample"] = [f"S{i}" for i in range(args.n_haplotypes)]
            raw_spectra_df.append(trial_spectra_df)
        raw_spectra_df = pd.concat(raw_spectra_df)
        raw_spectra_df.to_csv(args.raw_spectra, index=False)

    for i, pval in enumerate(trial_pvals):
        res_df.append({
            "n_markers": args.n_markers,
            "n_haplotypes": args.n_haplotypes,
            "n_permutations": args.n_permutations,
            "n_mutations": args.n_mutations,
            "effect_size": args.effect_size,
            "mutation_type": args.mutation_type,
            "exp_af": args.exp_af,
            "distance_method": args.distance_method,
            "trial": i,
            "pval": pval,
        })

    res_df = pd.DataFrame(res_df)
    res_df.to_csv(args.results, index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        help=
        """Path to file containing summary results for each simulation trial.""",
    )
    p.add_argument(
        "-n_markers",
        type=int,
        default=1_000,
        help=
        """Number of markers at which to simulate genotypes. Default is 1,000.""",
    )
    p.add_argument(
        "-n_haplotypes",
        type=int,
        default=100,
        help=
        """Number of haplotypes on which to simulate genotypes/mutation spectra. Default is 100.""",
    )
    p.add_argument(
        "-n_permutations",
        type=int,
        default=1_000,
        help=
        """Number of permutations to use for significance testing in each simulation. Default is 1,000.""",
    )
    p.add_argument(
        "-n_mutations",
        type=int,
        default=100,
        help=
        """Number of mutations to simulate on each haplotype. Default is 100.""",
    )
    p.add_argument(
        "-effect_size",
        type=int,
        default=110,
        help=
        """Effect size of simulated mutator allele, expressed as a percentage of the normal mutation rate. Default is 110.""",
    )
    p.add_argument(
        "-mutation_type",
        type=str,
        default="C_T",
        help=
        """Mutation type affected by simulated mutator allele. Default is C_T.""",
    )
    p.add_argument(
        "-exp_af",
        type=float,
        default=50,
        help=
        """Expected allele frequency at the mutator locus, expressed as a percentage. Default is 50.""",
    )
    p.add_argument(
        "-distance_method",
        default="chisquare",
    )
    p.add_argument(
        "-raw_geno",
        default=None,
        help=
        """Path to file containing the raw genotype data from each simulation.""",
    )
    p.add_argument(
        "-raw_spectra",
        default=None,
        help=
        """Path to file containing the raw mutation spectrum data from each simulation.""",
    )
    p.add_argument(
        "-n_trials",
        type=int,
        default=1,
        help="""Number of times to simulate data.""",
    )
    args = p.parse_args()
    main(args)