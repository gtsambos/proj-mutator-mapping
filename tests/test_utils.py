import numpy as np
from ihd.utils import (
    compute_haplotype_distance,
    perform_ihd_scan,
    perform_permutation_test,
    compute_spectra,
    shuffle_spectra,
    compute_nansum,
    compute_allele_frequency,
    compute_genotype_similarity,
    compute_residuals,
    compute_manual_chisquare,
    find_central_mut,
    adjust_spectra_for_nuccomp,
)
import pytest
import scipy.stats as ss


@pytest.mark.parametrize("arr,groups", [(
    np.random.randint(0, 100, size=(100, 6)),
    np.ones(100),
)])
def test_shuffle_spectra(
    arr,
    groups,
):
    shuffled_spectra = shuffle_spectra(arr, groups)
    assert np.array_equal(arr, shuffled_spectra) is False


@pytest.mark.parametrize("arr,groups", [(
    np.random.randint(0, 100, size=(100, 6)),
    np.concatenate([
        np.zeros(98),
        np.array([1]),
        np.array([2]),
    ]),
)])
def test_shuffle_spectra_strat(arr, groups):
    shuffled_spectra = shuffle_spectra(arr, groups)

    # ensure that all but the final two rows of the shuffled array are different
    assert np.array_equal(arr, shuffled_spectra) is False

    # ensure that the final two rows of the shuffled array are identical
    # to the input array, since they correspond to two unique groups that
    # shouldn't be mixed and matched with the others
    assert np.array_equal(arr[-1, :], shuffled_spectra[-1, :]) is True
    assert np.array_equal(arr[-2, :], shuffled_spectra[-2, :]) is True



def test_compute_haplotype_distance(
    wt_haplotype_array,
    mut_haplotype_array,
):

    exp = 4.940695

    assert np.isclose(
        compute_haplotype_distance(
            wt_haplotype_array,
            mut_haplotype_array,
        ),
        exp,
    )


@pytest.mark.parametrize("a,b", [
    (np.array([11, 23, 5, 6]), np.array([6, 8, 9, 12])),
])
def test_compute_manual_chisquare(a, b):
    exp_stat, _, _, _ = ss.chi2_contingency(np.vstack((a, b)))
    assert np.isclose(compute_manual_chisquare(a, b), exp_stat)


@pytest.mark.xfail
def test_compute_spectra(bad_mutation_dataframe):
    _, _, _ = compute_spectra(bad_mutation_dataframe, k=1)


def test_compute_nansum(genotype_array):
    rowsums = compute_nansum(genotype_array, row=True)
    assert np.array_equal(np.array([4., 3., 2.]), rowsums,)


def test_compute_nansum_nans(genotype_array_nans):
    rowsums = compute_nansum(genotype_array_nans, row=True)
    assert np.array_equal(np.array([4., 2., 0.]), rowsums,)


def test_compute_allele_frequency_nans(genotype_array_nans):
    af = compute_allele_frequency(genotype_array_nans)
    assert np.array_equal(np.array([0.5, 0.25, 0.]), af,)


def test_compute_genotype_similarity(genotype_array):
    genotype_sims = compute_genotype_similarity(genotype_array)
    exp = np.array([
        np.corrcoef([1., 0.25, 0.5], [0., 0.5, 0.]),
        np.corrcoef([0., 1., 0.], [0.5, 0., 0.5]),
        np.corrcoef([1., 0., 1.], [1/3, 0.5, 0.]),
    ])


def test_compute_spectra(good_mutation_dataframe):
    samples, mutations, spectra = compute_spectra(good_mutation_dataframe, k=1, cpg=True)
    assert samples == ["A", "B", "C"]
    assert mutations == ["A>C", "A>G", "A>T", "C>A", "C>G"]
    assert np.array_equal(
        spectra,
        np.array([
            [1, 1, 0, 5, 3],
            [3, 1, 3, 1, 2],
            [4, 4, 0, 2, 0],
        ]),
    )


@pytest.mark.parametrize("X,y", [(np.arange(5).astype(np.float64), np.arange(5).astype(np.float64)),])
def test_compute_residuals(X, y):
    assert np.allclose(compute_residuals(X, y), np.zeros(5))


def test_perform_ihd_scan(spectra_array, genotype_array, genotype_similarity):
    focal_dists = perform_ihd_scan(
        spectra_array,
        genotype_array,
        genotype_similarity,
    )

    exp_dists = np.array([
        -0.96818239,
        1.13693891,
        -0.16875652,
    ])
    assert np.allclose(np.array(focal_dists),
                       exp_dists)


@pytest.mark.parametrize("n_permutations,comparison_wide", [
    (100, False),
    (1000, False),
    (100, True),
])
def test_perform_ihd_permutation_test(
    spectra_array,
    genotype_array,
    genotype_similarity,
    n_permutations,
    comparison_wide,
):
    perm_res = perform_permutation_test(
        spectra_array,
        genotype_array,
        genotype_similarity,
        np.ones(4),
        n_permutations=n_permutations,
        comparison_wide=comparison_wide,
    )
    if comparison_wide:
        assert perm_res.shape[0] == n_permutations and perm_res.shape[1] == genotype_array.shape[0]
    else:
        assert perm_res.shape[0] == n_permutations


@pytest.mark.parametrize("kmer,exp", [
    ("CCT>CAT", "C>A"),
    ("CCG>CTG", "CpG>TpG"),
    ("ATN>NCG", "T>C"),
    pytest.param("A>G", "A>G", marks=pytest.mark.xfail(reason='input kmer is not a 3-mer')),
    pytest.param("ACG>ATG", "C>T", marks=pytest.mark.xfail(reason='should treat as CpG')),
])
def test_find_central_mut(kmer, exp):
    assert find_central_mut(kmer) == exp


def test_adjust_spectra_for_nuccomp(
    wt_haplotype_array,
    mut_haplotype_array,
    callable_kmer_arr_mut,
    callable_kmer_arr_wt,
):
    wt_adj, mut_adj = adjust_spectra_for_nuccomp(
        np.sum(wt_haplotype_array, axis=0),
        np.sum(mut_haplotype_array, axis=0),
        np.sum(callable_kmer_arr_wt, axis=0),
        np.sum(callable_kmer_arr_mut, axis=0),
    )

    assert np.array_equal(wt_adj, np.array([2, 3, 3, 9, 6, 3]))
    assert np.array_equal(mut_adj, np.array([6, 6, 15, 6, 8, 6]))
