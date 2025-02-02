rule plot_wild_afs:
    input:
        vcf = "data/vcf/wild.chr6.108_116Mb.vcf.gz",
        py_script = "scripts/plot_wild_afs.py"
    output: "figs/wild_afs.{ext}"
    shell:
        """
        python {input.py_script} --vcf {input.vcf} --out {output}
        """