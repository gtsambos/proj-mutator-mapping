import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

plt.rc("font", size=12)

def main(args):
    df = pd.read_csv(args.results)
    df['Power'] = df['pval'].apply(lambda p: p <= 0.05)

    df = df[(df["mutation_type"].isin(["C_G", "C_A", "C_T"]))
            & (df["distance_method"] == "cosine")]

    df["mutation_type"] = df["mutation_type"].apply(lambda m: m.replace("_", r"$\to$"))
    df["effect_size"] = df["effect_size"] / 100.

    replace_dict = {
        "mutation_type": "Mutation type",
        "n_haplotypes": "# haplotypes",
        "effect_size": "Mutator effect size",
        "n_mutations": "# mutations",
        "n_markers": "# markers",
        "tag_strength": "Linkage",
        "exp_af": "Allele frequency",
    }
    df.rename(columns=replace_dict, inplace=True)

    g = sns.FacetGrid(data=df, row="Mutation type", col="# haplotypes", aspect=1.5)
    g.map(sns.lineplot, "Mutator effect size", "Power", "# mutations", palette="colorblind", ci=95, n_boot=1_000)
    g.add_legend(title = "# of mutations\nper haplotype")
    g.tight_layout()

    g.savefig(args.out, dpi=300)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", help="""results of power simulation""")
    p.add_argument("--out", help="""name of output plot""")
    args = p.parse_args()
    main(args)
