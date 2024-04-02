import pandas as pd
import numpy as np
from typing import Callable, Tuple, List
import numba


@numba.njit
def compute_manual_chisquare(a: np.ndarray, b: np.ndarray) -> np.float64:
    """Compute a chi-square test of independence between two
    groups of observations. Since numba doesn't cooperate with the
    `scipy.stats` implementation, I've written a "manual" version
    of the calculation so that it's jit-able.

    Args:
        a (np.ndarray): 1D numpy array of (N, ) observations.
        b (np.ndarray): 1D numpy array of (N, ) obseravtions.

    Returns:
        chisquare_stat (np.float64): The Chi-square statistic comparing \
            observed vs. expected values of each observation.
    """
    observed = np.vstack((a, b))
    row_sums = compute_nansum(observed, row=True)
    col_sums = compute_nansum(observed, row=False)
    total = np.nansum(observed)

    expected = np.zeros(observed.shape, dtype=np.float64)
    for row_i in np.arange(row_sums.shape[0]):
        for col_i in np.arange(col_sums.shape[0]):
            exp = (row_sums[row_i] * col_sums[col_i]) / total
            expected[row_i, col_i] = exp
    chi_stat = np.sum(np.square(observed - expected) / expected)
    return chi_stat


@numba.njit
def compute_nansum(a: np.ndarray, row: bool = True) -> np.ndarray:
    """Compute the sum of a 2D numpy array
    on a per-row basis, ignoring nans. Since `numba` does not
    support kwargs in the `np.nansum` function, it's
    necessary to piece this out into its own function
    so that it can be decorated with `numba.njit`.

    Args:
        a (np.ndarray): A 2D numpy array of size (N, M).

        row (bool, optional): Whether to calculate means by row or column. Defaults to True.


    Returns:
        rowsums (np.ndarray): A 1D numpy array of size (N, ) containing \
            sums of every row in the input.
    """
    idx = 0 if row else 1
    empty_a = np.zeros(a.shape[idx])
    for i in range(a.shape[idx]):
        empty_a[i] = np.nansum(a[i]) if row else np.nansum(a[:, i])
    return empty_a


@numba.njit
def compute_allele_frequency(genotype_matrix: np.ndarray) -> np.ndarray:
    """Given a genotype matrix of size (G, N) where G is the number of 
    genotyped sites and N is the number of samples, compute the allele 
    frequency at every marker.

    Args:
        genotype_matrix (np.ndarray): A 2D numpy array of genotypes at \
            every genotyped marker, of size (G, N), where G is the number \
            of genotyped sites and N is the number of samples.

    Returns:
        allele_frequencies (np.ndarray): A 1D numpy array of size (G, ) containing allele frequencies \
        at every genotyped site in the collection of N samples.
    """
    # figure out the allele number
    allele_number = genotype_matrix.shape[1] * 2
    # sum the genotypes at each locus
    allele_counts = compute_nansum(genotype_matrix, row=True)
    # get allele frequencies
    return allele_counts / allele_number


@numba.njit
def compute_manual_cosine_distance(a: np.ndarray, b: np.ndarray) -> np.float64:
    """Compute the cosine distance between
    two 1D numpy arrays. Although methods to compute
    cosine similarity and distance exist in `scipy` and
    `sklearn`, they are not `numba.njit`'able.

    Args:
        a (np.ndarray): A 1D numpy array of size (N, ).
        b (np.ndarray): A 1D numpy array of size (N, ).

    Returns:
        cosine_distance (np.float64): Cosine distance between a and b.
    """
    dot = a.dot(b)
    a_sumsq, b_sumsq = np.sum(np.square(a)), np.sum(np.square(b))
    a_norm, b_norm = np.sqrt(a_sumsq), np.sqrt(b_sumsq)
    cossim = dot / (a_norm * b_norm)
    return 1 - cossim


@numba.njit
def compute_haplotype_distance(
    a_haps: np.ndarray,
    b_haps: np.ndarray,
    distance_method: Callable = compute_manual_chisquare,
) -> np.float64:
    """Compute the distance between the aggregate
    mutation spectrum of two collections of haplotype mutation data.
    The input arrays should both be 2D numpy arrays of size (N, M), 
    with rows and columns corresponding to samples and mutation types, 
    respectively. This method will first aggregate the mutation spectra across 
    samples to create two new 1D arrays, each of size (M, ). Then, it 
    will compute the distance between those two 1D arrays using the provided 
    distance method.

    Args:
        a_haps (np.ndarray): 2D array of size (N, M) containing the mutation \
            spectrum of each sample, where N is the number of samples and M is \
            the number of mutation types.

        b_haps (np.ndarray): 2D array of size (N, M) containing the mutation \
            spectrum of each sample, where N is the number of samples and M \
            is the number of mutation types.

        distance_method (Callable, optional): Callable method to compute the \
            distance between aggregate mutation spectra. Must accept two 1D numpy \
            arrays and return a single floating point value. Defaults to \
            `compute_manual_chisquare`.

    Returns:
        distance (np.float64): Distance between the aggregate \
            mutation spectra of the two haplotypes.
    """
    # first, sum the spectrum arrays such that we add up the
    # total number of C>T, C>A, etc. across all samples.
    a_hap_sums = np.sum(a_haps, axis=0)
    b_hap_sums = np.sum(b_haps, axis=0)

    if np.sum(a_hap_sums) == 0 or np.sum(b_hap_sums) == 0:
        dist = 0
    else:
        dist = distance_method(a_hap_sums, b_hap_sums)
    return dist
    


@numba.njit
def shuffle_spectra(
    spectra: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Randomly shuffle the rows of a 2D numpy array of 
    mutation spectrum data of size (N, M), where N is the number
    of samples and M is the number of mutation types. Shuffled array
    is returned such that sample mutation spectra no longer correspond to the
    appropriate sample indices into the rows of the array. Samples are also
    shuffled within the specified `groups`. In other words, we only shuffle
    the indices of samples that correspond to a unique value in the `groups` 
    array.

    Args:
        spectra (np.ndarray): 2D numpy array of mutation spectrum data of \
        shape (N, M), where N is the number of samples and M is the number \
        of mutation types.

        groups (np.ndarray): 1D numpy array of group labels (e.g., epochs) \
            that each sample belongs to, of shape (N, ) where N is the number \
            of samples. Mutation spectra will only be shuffled *within* \
            their assigned group.

    Returns:
        shuffled_spectra (np.ndarray): The input array, but with shuffled rows.
    """
    shuffled_spectra = np.zeros(spectra.shape)

    uniq_groups = np.unique(groups)
    for g in uniq_groups:
        # get corresponding indices of samples in the current group
        g_i_true = np.where(groups == g)[0]
        g_i_shuffled = np.where(groups == g)[0]
        # shuffle just the group indices so that sample labels
        # in that group no longer correspond to the correct
        # mutation spectra arrays
        np.random.shuffle(g_i_shuffled)
        shuffled_spectra[g_i_true, :] = spectra[g_i_shuffled, :]
    return shuffled_spectra


@numba.njit
def compute_genotype_similarity(genotype_matrix: np.ndarray) -> np.ndarray:
    """Compute the genetic similarity between groups of haplotypes
    at every genotyped marker. At each marker, divide the haplotypes into
    two groups based on the parental allele each haplotype inherited. In 
    each group, calculate allele frequencies at every marker along the genome.
    Then, calculate the Pearson correlation coefficient between the two groups'
    allele frequency vectors.

    Args:
        genotype_matrix (np.ndarray): A 2D numpy array of genotypes at \
            every genotyped marker, of size (G, N), where G is the number \
            of genotyped sites and N is the number of samples.

    Returns:
        genotype_similarity (np.ndarray): A 1D numpy array of size (G, ), where G is the number \
        of genotyped sites.
    """
    # store genotype similarities at every marker
    genotype_sims: np.ndarray = np.zeros(genotype_matrix.shape[0], dtype=np.float64)
    # loop over every site in the genotype matrix
    for ni in np.arange(genotype_matrix.shape[0]):
        a_hap_idxs = np.where(genotype_matrix[ni] == 0)[0]
        b_hap_idxs = np.where(genotype_matrix[ni] == 2)[0]
        # compute allele frequencies in each haplotype group
        a_afs, b_afs = (
            compute_allele_frequency(genotype_matrix[:, a_hap_idxs]),
            compute_allele_frequency(genotype_matrix[:, b_hap_idxs]),
        )
        # compute Pearson correlation between allele frequencies
        af_corr = np.corrcoef(a_afs, b_afs)[0][1]
        # NOTE: not sure how good of an idea this is, even if it's rare
        if np.isnan(af_corr):
            af_corr = 0
        genotype_sims[ni] = af_corr

    return genotype_sims


def adjust_spectra_for_nuccomp(
    a_spectra: np.ndarray,
    b_spectra: np.ndarray,
    a_denom: np.ndarray,
    b_denom: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale the counts in the input mutation spectra
    such that they are adjusted in terms of nucleotide context.
    For a given mutation type (e.g., C>T), if the `a_spectra` 
    contains more C>[A/T/G] mutations, it may simply be because
    the samples represented in `a_spectra` had more cytosines 
    that were accessible to variant calling. If `a_denom` contains
    more C nucleotides than `b_denom`, we adjust the counts of the 
    C>[A/T/G] mutations in `a_spectra` by a scaling factor 
    (`b_denom` / `a_denom`). And vice versa.

    Args:
        a_spectra (np.ndarray): 1D numpy array containing the aggregate mutation \
            spectrum in group "A."

        b_spectra (np.ndarray): 1D numpy array containing the aggregate mutation \
            spectrum in group "B."

        a_denom (np.ndarray): 1D numpy array containing the aggregate number of \
            callable base pairs in group "A." At each index i in this array, the \
            total count of accessible nucleotides should correspond to the "reference" \
            nucleotide of the corresponding mutation type in `a_spectra` at the same index. \
            For example, if the first two entries of `a_spectra` corresponds to the aggregate count \
            of C>T and C>A mutations, the first two entries of `a_denom` should both contain \
            the aggregate counts of accessible C nucleotides in the group.

        b_denom (np.ndarray): 1D numpy array containing the aggregate number of \
            callable base pairs in group "B." At each index i in this array, the \
            total count of accessible nucleotides should correspond to the "reference" \
            nucleotide of the corresponding mutation type in `b_spectra` at the same index.
            For example, if the first two entries of `b_spectra` corresponds to the aggregate count \
            of C>T and C>A mutations, the first two entries of `b_denom` should both contain \
            the aggregate counts of accessible C nucleotides in the group.

    Returns:
        adj_a_spectra (np.ndarray): The first of the input mutation spectra, adjusted for nucleotide context.
        adj_b_spectra (np.ndarray): The second of the input mutation spectra, adjusted for nucleotide context.
    """
    # store the adjusted mutation spectra in two new arrays
    new_a_spectra, new_b_spectra = (
        np.zeros(a_spectra.shape),
        np.zeros(b_spectra.shape),
    )

    # loop over the indices of the input mutation spectra
    for nuc_i in np.arange(a_spectra.shape[0]):
        # get the aggregate count of the mutation type in either group
        a_mut_count, b_mut_count = a_spectra[nuc_i], b_spectra[nuc_i]
        # get the aggregate count of accessible nucleotides in either group
        a_nuc_count, b_nuc_count = a_denom[nuc_i], b_denom[nuc_i]

        # if the count of nucleotides in group A is larger than in group B,
        # adjust group A down by the scaling factor
        if a_nuc_count > b_nuc_count:
            adj_factor = b_nuc_count / a_nuc_count
            a_mut_count *= adj_factor
        # and vice versa if group B > group A
        elif b_nuc_count > a_nuc_count:
            adj_factor = a_nuc_count / b_nuc_count
            b_mut_count *= adj_factor

        new_a_spectra[nuc_i] = a_mut_count
        new_b_spectra[nuc_i] = b_mut_count

    return new_a_spectra, new_b_spectra


@numba.njit
def perform_ihd_scan(
    spectra: np.ndarray,
    genotype_matrix: np.ndarray,
    genotype_similarity: np.ndarray,
    covariate_ratios: np.ndarray,
    distance_method: Callable = compute_manual_chisquare,
    adjust_statistics: bool = True,
) -> np.ndarray:
    """Iterate over every genotyped marker in the `genotype_matrix`, 
    divide the haplotypes into two groups based on sample genotypes at the 
    marker, and compute the distance between the aggregate mutation spectra
    of each group. Return a list of cosine distances of length (G, ), where
    G is the number of genotyped sites.

    Args:
        spectra (np.ndarray): A 2D numpy array of mutation spectra in \
            all genotyped samples, of size (N, M) where N is the number of \
            samples and M is the number of mutation types.

        genotype_matrix (np.ndarray): A 2D numpy array of genotypes at \
            every genotyped marker, of size (G, N), where G is the number \
            of genotyped sites and N is the number of samples.

        genotype_similarity (np.ndarray): A 1D numpy array of size (G, ), where \
            G is the number of genotyped sites. Contains the correlation coefficient \
            between genome-wide allele frequencies of haplotypes with A vs. B genotypes \
            at every site.

        covariate_ratios (np.ndarray): A 1D numpy array of size (G, ), where \
            G is the number of genotyped sites. Contains the ratio of covariate \
            values between haplotypes with A vs. B genotypes at every site.

        distance_method (Callable, optional): Callable method to compute the \
            distance between aggregate mutation spectra. Must accept two 1D numpy \
            arrays and return a single floating point value. Defaults to \
            `compute_manual_chisquare`.

        adjust_statistics (bool, optional): Whether to compute adjusted statistics \
            at each marker by regressing genotype similarity against statistics. \
            Defaults to True.

    Returns:
        distances (List[np.float64]): List of inter-haplotype \
        distances at every marker.
    """

    # store distances at each marker
    focal_dist: np.ndarray = np.zeros(genotype_matrix.shape[0], dtype=np.float64)
    a_muts: np.ndarray = np.zeros((genotype_matrix.shape[0], spectra.shape[1]), dtype=np.float64)
    b_muts: np.ndarray = np.zeros((genotype_matrix.shape[0], spectra.shape[1]), dtype=np.float64)
    # loop over every site in the genotype matrix
    for ni in np.arange(genotype_matrix.shape[0]):
        a_hap_idxs = np.where(genotype_matrix[ni] == 0)[0]
        b_hap_idxs = np.where(genotype_matrix[ni] == 2)[0]

        a_spectra, b_spectra = (
            spectra[a_hap_idxs],
            spectra[b_hap_idxs],
        )

        cur_dist = compute_haplotype_distance(
            a_spectra,
            b_spectra,
            distance_method=distance_method,
        )
        a_hap_sums, b_hap_sums = compute_spectral_sets(a_spectra, b_spectra)
        focal_dist[ni] = cur_dist
        a_muts[ni] = a_hap_sums
        b_muts[ni] = b_hap_sums
        
    if adjust_statistics:
        covariate_matrix_full = np.hstack(
            (
                genotype_similarity.reshape(-1, 1),
                covariate_ratios,
            )
        )
        return compute_residuals(covariate_matrix_full, focal_dist)
    else:
        return focal_dist

@numba.njit
def perform_spectral_scan(
    spectra: np.ndarray,
    genotype_matrix: np.ndarray,
    distance_method: Callable = compute_manual_chisquare,
) -> np.ndarray:
    """Iterate over every genotyped marker in the `genotype_matrix`, 
    divide the haplotypes into two groups based on sample genotypes at the 
    marker, and compute the distance between the aggregate mutation spectra
    of each group. Return a list of cosine distances of length (G, ), where
    G is the number of genotyped sites.

    Args:
        spectra (np.ndarray): A 2D numpy array of mutation spectra in \
            all genotyped samples, of size (N, M) where N is the number of \
            samples and M is the number of mutation types.

        genotype_matrix (np.ndarray): A 2D numpy array of genotypes at \
            every genotyped marker, of size (G, N), where G is the number \
            of genotyped sites and N is the number of samples.

        genotype_similarity (np.ndarray): A 1D numpy array of size (G, ), where \
            G is the number of genotyped sites. Contains the correlation coefficient \
            between genome-wide allele frequencies of haplotypes with A vs. B genotypes \
            at every site.

        covariate_ratios (np.ndarray): A 1D numpy array of size (G, ), where \
            G is the number of genotyped sites. Contains the ratio of covariate \
            values between haplotypes with A vs. B genotypes at every site.

        distance_method (Callable, optional): Callable method to compute the \
            distance between aggregate mutation spectra. Must accept two 1D numpy \
            arrays and return a single floating point value. Defaults to \
            `compute_manual_chisquare`.

        adjust_statistics (bool, optional): Whether to compute adjusted statistics \
            at each marker by regressing genotype similarity against statistics. \
            Defaults to True.

    Returns:
        distances (List[np.float64]): List of inter-haplotype \
        distances at every marker.
    """

    # store distances at each marker
    # focal_dist: np.ndarray = np.zeros(genotype_matrix.shape[0], dtype=np.float64)
    a_muts: np.ndarray = np.zeros((genotype_matrix.shape[0], spectra.shape[1]), dtype=np.float64)
    b_muts: np.ndarray = np.zeros((genotype_matrix.shape[0], spectra.shape[1]), dtype=np.float64)
    # loop over every site in the genotype matrix
    for ni in np.arange(genotype_matrix.shape[0]):
        a_hap_idxs = np.where(genotype_matrix[ni] == 0)[0]
        b_hap_idxs = np.where(genotype_matrix[ni] == 2)[0]

        a_spectra, b_spectra = (
            spectra[a_hap_idxs],
            spectra[b_hap_idxs],
        )

        cur_dist = compute_haplotype_distance(
            a_spectra,
            b_spectra,
            distance_method=distance_method,
        )
        a_hap_sums, b_hap_sums = compute_spectral_sets(a_spectra, b_spectra)
    #     focal_dist[ni] = cur_dist
        a_muts[ni] = a_hap_sums
        b_muts[ni] = b_hap_sums

    return a_muts, b_muts
        

@numba.njit
def get_sample_sizes(
    genotype_matrix: np.ndarray,
    generations_list: List[int],
) -> np.ndarray:
    """Iterate over every genotyped marker in the `genotype_matrix`, 
    divide the haplotypes into two groups based on sample genotypes at the 
    marker, and compute the number of samples in each group at each marker.

    Args:
        genotype_matrix (np.ndarray): A 2D numpy array of genotypes at \
            every genotyped marker, of size (G, N), where G is the number \
            of genotyped sites and N is the number of samples.

    Returns:
        sample_sizes (np.ndarray): Sample sizes in the A and B sets at each locus.
    """
    sample_sizes: np.ndarray = np.zeros((genotype_matrix.shape[0], 2), dtype=np.float64)

    # loop over every site in the genotype matrix
    for ni in np.arange(genotype_matrix.shape[0]):
        a_hap_idxs = np.where(genotype_matrix[ni] == 0)[0]
        b_hap_idxs = np.where(genotype_matrix[ni] == 2)[0]

        # Get the number of inbreeding generations for each sample in each group.
        a_gens = [generations_list[i] for i in a_hap_idxs]
        b_gens = [generations_list[i] for i in b_hap_idxs]

        # store the total number of generations in each group
        sample_sizes[ni] = np.array([sum(a_gens), sum(b_gens)])

    return sample_sizes

@numba.njit
def compute_spectral_sets(
    a_haps: np.ndarray,
    b_haps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the spectral sets of two collections of haplotype
    mutation data. The input arrays should both be 2D numpy arrays of size (N, M), 
    with rows and columns corresponding to samples and mutation types, 
    respectively.

    Args:
        a_haps (np.ndarray): 2D array of size (N, M) containing the mutation \
            spectrum of each sample, where N is the number of samples and M is \
            the number of mutation types.

        b_haps (np.ndarray): 2D array of size (N, M) containing the mutation \
            spectrum of each sample, where N is the number of samples and M \
            is the number of mutation types.

    Returns:
        a_hap_sums (np.ndarray): The aggregate mutation spectrum of the first \
            collection of haplotypes.

        b_hap_sums (np.ndarray): The aggregate mutation spectrum of the second \
            collection of haplotypes.
    """
    # first, sum the spectrum arrays such that we add up the
    # total number of C>T, C>A, etc. across all samples.
    a_hap_sums = np.sum(a_haps, axis=0)
    b_hap_sums = np.sum(b_haps, axis=0)

    return a_hap_sums, b_hap_sums

def get_covariate_matrix(
    pheno: pd.DataFrame,
    samples: List[str],
    covariate_cols: List[str] = ["n_generations"],
) -> np.ndarray:
    """Generate a matrix of covariate values for each sample. Returns
    a 2D numpy array of size (C, N), where C is the number of covariates
    specified in the `covariate_cols` argument, and N is the number of samples.

    Args:
        pheno (pd.DataFrame): Pandas dataframe containing metadata about each sample. \
            This dataframe should have a "sample" column and should contain each of the columns \
            specified in `covariate_cols`.

        samples (List[str]): List of samples to subset the dataframe.

        covariate_cols (List[str], optional): List of column names in `pheno` that should \
            be used as covariates. Defaults to ["n_generations"].

    Returns:
        np.ndarray: A 2D numpy array of size (C, N), where C is the number of covariates \
            and N is the number of samples.
    """

    cols = ["sample"] + covariate_cols
    # subset pheno information to relevant samples
    pheno_sub = (
        pheno[pheno["sample"].isin(samples)].drop_duplicates(cols).set_index("sample")
    )
    covariate_matrix = pheno_sub.loc[samples][covariate_cols].values.T
    return covariate_matrix


@numba.njit
def calculate_covariate_by_marker(
    covariate_matrix: np.ndarray,
    genotype_matrix: np.ndarray,
) -> np.ndarray:
    """For each covariate in the `covariate_matrix`, compute the ratio of
        covariate values in samples with A or B genotypes at each marker in the
        `genotype_matrix`.

    Args:
        covariate_matrix (np.ndarray): A 2D numpy array of size (C, N), where C is \
            the number of covariates and N is the number of samples.

        genotype_matrix (np.ndarray): A 2D numpy array of genotypes at \
            every genotyped marker, of size (G, N), where G is the number \
            of genotyped sites and N is the number of samples.

    Returns:
        np.ndarray: A 2D numpy array of size (G, C), where G is the number of \
            genotyped sites and C is the number of covariates.
    """
    # store the ratio of covariate values between D and B haplotypes
    # at every marker along the genome for each covariate
    covariate_ratios: np.ndarray = np.zeros(
        (genotype_matrix.shape[0], covariate_matrix.shape[0]),
        dtype=np.float64,
    )
    # loop over every site in the genotype matrix
    for ni in np.arange(genotype_matrix.shape[0]):
        a_hap_idxs = np.where(genotype_matrix[ni] == 0)[0]
        b_hap_idxs = np.where(genotype_matrix[ni] == 2)[0]
        # loop over every covariate in the covariate matrix
        for ci in np.arange(covariate_matrix.shape[0]):
            # subset covariate matrix to the covariate of interest
            covariate_matrix_sub = covariate_matrix[ci]
            # compute ratio of covariate values between the two groups
            a_sum = np.sum(covariate_matrix_sub[a_hap_idxs])
            b_sum = np.sum(covariate_matrix_sub[b_hap_idxs])
            ratio = a_sum / b_sum
        covariate_ratios[ni, ci] = ratio

    return covariate_ratios


@numba.njit
def compute_residuals(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Use ordinary least-squares (OLS) to fit a linear model
    predicting `y` as a function of `X`. Then, compute the residuals
    between the predicted y-values and the true y-values.

    Args:
        X (np.ndarray): 2D numpy array of size (M, N), where M is the number of \
            observations and N is the number of independent variables.

        y (np.ndarray): 1D numpy array of size (M, ), containing the dependent variables \
            for all M observations.

    Returns:
        resids (np.ndarray): 1D numpy array of size (M, ) containing residuals.
    """
    ones = np.ones(X.shape[0]).reshape(-1, 1)
    X = np.hstack((X, ones))
    coefs = np.linalg.lstsq(X, y)[0]
    y_ = np.sum(X[:, :-1] * coefs[:-1], axis=1) + coefs[-1]
    resids = y - y_
    return resids


@numba.njit(parallel=True)
def perform_permutation_test(
    spectra: np.ndarray,
    genotype_matrix: np.ndarray,
    genotype_similarity: np.ndarray,
    covariate_ratios: np.ndarray,
    strata: np.ndarray,
    distance_method: Callable = compute_manual_chisquare,
    n_permutations: int = 1_000,
    progress: bool = False,
    adjust_statistics: bool = True,
) -> np.ndarray:
    """Conduct a permutation test to assess the significance of 
    any observed IHD peaks. In each of the `n_permutations` trials, 
    do the following: 1. create a shuffled version of the input mutation `spectra`, so that
    sample names/indices no longer correspond to the appropriate mutation
    spectrum. 2. run an IHD scan by computing the distance between
    the aggregate mutation spectrum of samples with either genotype
    at every marker in the `genotype_matrix`. 3. store the maximum distance encountered at any marker.
    Then, return a list of the maximum distances encountered in each of
    the trials. Alternatively, if `comparison_wide` is True, return a matrix of
    size (P, G), where P is the number of permutations and G is the number of
    genotyped markers, in which we store the distance value encountered at
    every marker in every permutation trial.

    Args:
        spectra (np.ndarray): A 2D numpy array of mutation spectra in all \
            genotyped samples, of size (N, M) where N is the number of samples \
            and M is the number of mutation types.

        genotype_matrix (np.ndarray): A 2D numpy array of genotypes at every \
            genotyped marker, of size (G, N), where G is the number of genotyped \
            sites and N is the number of samples.

        genotype_similarity (np.ndarray): A 1D numpy array of correlation coefficients \
            of size (G, ), where G is the number of genotyped sites. At each element of \
            the array, we store the correlation coefficient between genome-wide D allele \
            frequencies calculated in samples with either allele at the corresponding site G_i.

        covariate_ratios (np.ndarray): A 1D numpy array of size (G, ), where \
            G is the number of genotyped sites. Contains the ratio of covariate \
            values between haplotypes with A vs. B genotypes at every site.
        
        strata (np.ndarray): A 1D numpy array of "group labels" of size (N, ), where \
            N is the number of samples. If samples are assigned different group labels, their \
            spectra will be permuted *within* those groups during the permutation testing step.

        distance_method (Callable, optional): Callable method to compute the \
            distance between aggregate mutation spectra. Must accept two 1D numpy \
            arrays and return a single floating point value. Defaults to \
            `compute_manual_chisquare`.

        n_permutations (int, optional): Number of permutations to perform \
            (i.e., number of times to shuffle the spectra and compute IHDs at \
            each marker). Defaults to 1_000.
        
        progress (bool, optional): Whether to output a count of how many permutations \
            have completed. Defaults to False.
        
        adjust_statistics (bool, optional): Whether to compute adjusted statistics \
            at each marker by regressing genotype similarity against statistics. \
            Defaults to True.
              

    Returns:
        null_distances (np.ndarray): 2D numpy array of size (P, G) \
            where P is the number of permutations and G is either 1 \
            (if we're computing a genome-wide distance threshold) or \
            the number of genotyped markers (if we're computing thresholds \
            at each individual marker).
    """

    # store max distance encountered in each permutation
    null_distances: np.ndarray = np.zeros(n_permutations)

    for pi in numba.prange(n_permutations):
        if pi > 0 and pi % 100 == 0 and progress:
            print(pi)
        # shuffle the mutation spectra by row
        shuffled_spectra = shuffle_spectra(spectra, strata)

        # perform the IHD scan
        perm_distances = perform_ihd_scan(
            shuffled_spectra,
            genotype_matrix,
            genotype_similarity,
            covariate_ratios,
            distance_method=distance_method,
            adjust_statistics=adjust_statistics,
        )

        null_distances[pi] = np.max(perm_distances)

    return null_distances

@numba.njit(parallel=True)
def perform_wide_permutation_test(
    spectra: np.ndarray,
    genotype_matrix: np.ndarray,
    genotype_similarity: np.ndarray,
    covariate_ratios: np.ndarray,
    strata: np.ndarray,
    distance_method: Callable = compute_manual_chisquare,
    n_permutations: int = 1_000,
    progress: bool = False,
    adjust_statistics: bool = True,
) -> np.ndarray:
    """Conduct a permutation test to assess the significance of 
    any observed IHD peaks. In each of the `n_permutations` trials, 
    do the following: 1. create a shuffled version of the input mutation `spectra`, so that
    sample names/indices no longer correspond to the appropriate mutation
    spectrum. 2. run an IHD scan by computing the distance between
    the aggregate mutation spectrum of samples with either genotype
    at every marker in the `genotype_matrix`. 3. store the maximum distance encountered at any marker.
    Then, return a matrix of size (P, G), where P is the number of permutations and G is the number of
    genotyped markers, in which we store the distance value encountered at
    every marker in every permutation trial.

    Args:
        spectra (np.ndarray): A 2D numpy array of mutation spectra in all \
            genotyped samples, of size (N, M) where N is the number of samples \
            and M is the number of mutation types.

        genotype_matrix (np.ndarray): A 2D numpy array of genotypes at every \
            genotyped marker, of size (G, N), where G is the number of genotyped \
            sites and N is the number of samples.

        genotype_similarity (np.ndarray): A 1D numpy array of correlation coefficients \
            of size (G, ), where G is the number of genotyped sites. At each element of \
            the array, we store the correlation coefficient between genome-wide D allele \
            frequencies calculated in samples with either allele at the corresponding site G_i.

        covariate_ratios (np.ndarray): A 1D numpy array of size (G, ), where \
            G is the number of genotyped sites. Contains the ratio of covariate \
            values between haplotypes with A vs. B genotypes at every site.
        
        strata (np.ndarray): A 1D numpy array of "group labels" of size (N, ), where \
            N is the number of samples. If samples are assigned different group labels, their \
            spectra will be permuted *within* those groups during the permutation testing step.

        distance_method (Callable, optional): Callable method to compute the \
            distance between aggregate mutation spectra. Must accept two 1D numpy \
            arrays and return a single floating point value. Defaults to \
            `compute_manual_chisquare`.

        n_permutations (int, optional): Number of permutations to perform \
            (i.e., number of times to shuffle the spectra and compute IHDs at \
            each marker). Defaults to 1_000.
        
        progress (bool, optional): Whether to output a count of how many permutations \
            have completed. Defaults to False.
        
        adjust_statistics (bool, optional): Whether to compute adjusted statistics \
            at each marker by regressing genotype similarity against statistics. \
            Defaults to True.
              

    Returns:
        null_distances (np.ndarray): 2D numpy array of size (P, G) \
            where P is the number of permutations and G is either 1 \
            (if we're computing a genome-wide distance threshold) or \
            the number of genotyped markers (if we're computing thresholds \
            at each individual marker).
    """

    # store max distance encountered in each permutation
    null_distances: np.ndarray = np.zeros((n_permutations, genotype_matrix.shape[0]))

    for pi in numba.prange(n_permutations):
        if pi > 0 and pi % 100 == 0 and progress:
            print(pi)
        # shuffle the mutation spectra by row
        shuffled_spectra = shuffle_spectra(spectra, strata)

        # perform the IHD scan
        perm_distances = perform_ihd_scan(
            shuffled_spectra,
            genotype_matrix,
            genotype_similarity,
            covariate_ratios,
            distance_method=distance_method,
            adjust_statistics=adjust_statistics,
        )

        null_distances[pi] = perm_distances

    return null_distances


@numba.njit(parallel=True)
def calculate_confint(
    spectra: np.ndarray,
    genotype_matrix: np.ndarray,
    covariate_matrix: np.ndarray,
    distance_method: Callable = compute_manual_chisquare,
    n_permutations: int = 10_000,
    progress: bool = True,
    adjust_statistics: bool = True,
    conf_int: float = 90.0,
) -> Tuple[int, int]:
    """Calculate a confidence interval around the maximum observed 
    distance peak by performing bootstrap resampling.  In each of the 
    `n_permutations` trials do the following: 1. resample the input mutation `spectra`
    with replacement. 2. run an IHD scan by computing the distance between
    the aggregate mutation spectrum of samples with either genotype
    at every marker in the `genotype_matrix`. 3. record the maximum distance encountered 
    at any marker, as well as the marker index at which that distance was observed.
    We'll consider that marker to be the likely "peak."
    Then, we return the bounds of the markers that contain 95% of all observed 
    peaks.

    Args:
        spectra (np.ndarray): A 2D numpy array of mutation spectra in all \
            genotyped samples, of size (N, M) where N is the number of samples \
            and M is the number of mutation types.

        genotype_matrix (np.ndarray): A 2D numpy array of genotypes at every \
            genotyped marker, of size (G, N), where G is the number of genotyped \
            sites and N is the number of samples.

        covariate_matrix (np.ndarray): A 2D numpy array of size (C, N), where C is \
            the number of covariates and N is the number of samples.
        
        distance_method (Callable, optional): Callable method to compute the \
            distance between aggregate mutation spectra. Must accept two 1D numpy \
            arrays and return a single floating point value. Defaults to \
            `compute_manual_chisquare`.

        n_permutations (int, optional): Number of permutations to perform \
            (i.e., number of times to resample the spectra with replacement. \
            Defaults to 1_000.
        
        progress (bool, optional): Whether to output a count of how many permutations \
            have completed. Defaults to False.
        
        adjust_statistics (bool, optional): Whether to compute adjusted statistics \
            at each marker by regressing genotype similarity against statistics. \
            Defaults to True.

        conf_int (float, optional): The confidence interval to compute around the marker \
            with the highest distance value. Defaults to 95.
              

    Returns:
        conf_ints (Tuple): Tuple of two integers, corresponding to the lower \
            and upper bound markers defining the specified confidence interval.
    """

    # store index of marker at max peak encountered in each permutation
    peak_markers: np.ndarray = np.zeros(n_permutations)

    for pi in numba.prange(n_permutations):
        if pi > 0 and pi % 1000 == 0 and progress:
            print(pi)
        # resample the mutation spectra by bootstrapping
        resampled_idxs = np.random.randint(
            0,
            high=spectra.shape[0],
            size=spectra.shape[0],
        )

        resampled_spectra = spectra[resampled_idxs, :]
        # resample the corresponding genotype data using the indices
        resampled_genotype_matrix = genotype_matrix[:, resampled_idxs]
        # recalculate genotype similarities using the resampled genotype matrix. NOTE: this is slow!
        resampled_genotype_similarity = compute_genotype_similarity(
            resampled_genotype_matrix
        )
        # resample the covariate matrix to include the bootstrap resampled samples
        resampled_covariate_matrix = covariate_matrix[:, resampled_idxs]
        resampled_covariate_ratios = calculate_covariate_by_marker(
            resampled_covariate_matrix,
            resampled_genotype_matrix,
        )

        # perform the IHD scan
        focal_dists = perform_ihd_scan(
            resampled_spectra,
            resampled_genotype_matrix,
            resampled_genotype_similarity,
            resampled_covariate_ratios,
            distance_method=distance_method,
            adjust_statistics=adjust_statistics,
        )

        peak_marker_i = np.argmax(focal_dists)
        peak_markers[pi] = peak_marker_i

    pctile_lo = (100 - conf_int) / 2.0
    pctile_hi = conf_int + pctile_lo

    return (
        int(np.percentile(peak_markers, q=pctile_lo)),
        int(np.percentile(peak_markers, q=pctile_hi)),
        peak_markers,
    )


def find_central_mut(kmer: str, cpg: bool = True) -> str:
    orig, new = kmer.split(">")
    fp, tp = orig[0], orig[2]
    central_orig = orig[1]
    central_new = new[1]
    if central_orig == "C" and tp == "G" and central_new == "T":
        if cpg:
            return f"{central_orig}p{tp}>{central_new}p{tp}"
        else:
            return f"{central_orig}>{central_new}"
    else:
        return ">".join([central_orig, central_new])


def compute_spectra(
    mutations: pd.DataFrame,
    k: int = 1,
    cpg: bool = True,
):
    """Compute the mutation spectrum of every sample in 
    the input pd.DataFrame of mutation data. The input
    dataframe should either contain a single entry for every mutation observed
    in every sample, or the aggregate count of each mutation type
    observed in each sample. The input dataframe must have at least three columns:
    'Strain' (denoting the sample), 'kmer' (denoting the 3-mer context of the 
    mutation -- e.g, CCT>CAT), and 'count' (denoting the number of times
    a mutation of type 'kmer' was observed in 'sample'). If the dataframe contains
    a single entry for every mutation observed in every sample, then the 'count'
    column should contain a value of 1 in every row.

    Args:
        mutations (pd.DataFrame): Pandas dataframe containing information \
            about the mutations observed in each sample. The dataframe must have \
            at least three columns: 'sample' (denoting the sample), 'kmer' (denoting \
            the 3-mer context of the mutation), and 'count' (denoting the number \
            of times a mutation of type 'kmer' was observed in 'sample').

        k (int, optional): k-mer size of mutations (e.g., k=3 means that we \
            will treat mutations as NXN NYN, where Ns represent the nucleotide \
            contexts on either side of the mutation, whereas k=1 means we will \
            treat them as X Y). Defaults to 1.
        
        cpg (bool, optional): whether to separate CpG>TpG mutations into their \
            own mutation type, distinct from C>T. Defaults to True.

    Returns:
        samples (List[str]): A list of samples in the dataframe (which are \
            indexed in the same order as the rows of the 2D `spectra` array).
        mutation_types (List[str]): A list of the unique mutation types \
            observed in the dataframe.
        spectra (np.ndarray): A 2D numpy array of mutation spectra in \
            all genotyped samples, of size (N, M) where N is the number \
            of samples and M is the number of mutation types.
    """

    # compute 3-mer spectra
    hap_spectra_agg = (
        mutations.groupby(["sample", "kmer"]).agg({"count": sum}).reset_index()
    )
    # if 1-mer spectra are desired, compute that
    if k == 1:
        # add base mutation type
        hap_spectra_agg["base_mut"] = hap_spectra_agg["kmer"].apply(
            lambda k: find_central_mut(k, cpg=cpg)
        )
        hap_spectra_agg = (
            hap_spectra_agg.groupby(["sample", "base_mut"])
            .agg({"count": sum})
            .reset_index()
        )
    # get spectra as per-haplotype vectors of mutation counts
    mut_col = "base_mut" if k == 1 else "kmer"
    spectra = (
        hap_spectra_agg.pivot(index="sample", columns=mut_col)
        .reset_index()
        .fillna(value=0)
    )

    print("using mutation columns:\n")
    # print the column names of the spectra dataframe
    print(spectra.columns)

    samples, mutations, spectra = (
        spectra["sample"].to_list(),
        [el[1] for el in spectra.columns[1:]],
        spectra.values[:, 1:],
    )

    return samples, mutations, spectra.astype(np.float64)
