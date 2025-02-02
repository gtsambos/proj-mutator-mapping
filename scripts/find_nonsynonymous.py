from cyvcf2 import VCF
import numpy as np

PROJDIR = "/Users/tomsasani/quinlanlab/proj-mutator-mapping"

vcf = VCF(f"{PROJDIR}/data/vcf/bxd.regions.snpeff.vcf.gz", gts012=True)
smp2idx = dict(zip(vcf.samples, range(len(vcf.samples))))

DBA = "sample_4512-JFI-0334_DBA_2J_three_lanes_phased_possorted_bam"
C57 = "sample_4512-JFI-0333_C57BL_6J_two_lanes_phased_possorted_bam"

outfh = open(f"{PROJDIR}/data/vcf/ns.vcf", "w")
print (vcf.raw_header.rstrip(), file=outfh)

for v in vcf:
    if v.POS < 95_000_000: continue
    if v.POS > 114_100_000: break
    if len(v.ALT) > 1: continue
    annotations = v.INFO.get("ANN").split(';')
    rel_ann = [a for a in annotations if a.split('|')[7] == "protein_coding"]
    if not any([a.split('|')[2] in ("MODERATE", "HIGH") for a in rel_ann]):
      continue

    gts = v.gt_types
    gqs = v.gt_quals
    ad, rd = v.gt_alt_depths, v.gt_ref_depths
    td = ad + rd
    ab = ad / td

    good_idxs = np.where((gts != 3) & (gqs >= 10) & (td >= 5))[0]
    if good_idxs.shape[0] == 0: continue

    founder_idxs = np.array([smp2idx[DBA], smp2idx[C57]])
    if np.intersect1d(good_idxs, founder_idxs).shape[0] != 2: continue

    ac_in_founders = []
    for fi in founder_idxs:
        if gts[fi] == 0:
            ac_in_founders.append(0)
        elif gts[fi] == 1:
            if ab[fi] >= 0.8: ac_in_founders.append(2)
            else: ac_in_founders.append(1)
        elif gts[fi] == 2:
            ac_in_founders.append(2)
        else: continue

    if sum(ac_in_founders) != 2: continue
    if any([gt == 1 for gt in ac_in_founders]): continue

    ac = np.sum(gts[good_idxs])
    an = good_idxs.shape[0] * 2
    for a in rel_ann:
        alt, cons, impact, gene = a.split('|')[:4]
        change = a.split('|')[10]
        tx = a.split('|')[6]
        print('\t'.join(
            list(
                map(str, [
                    v.CHROM,
                    v.start,
                    v.end,
                    v.REF,
                    v.ALT[0],
                    gts[smp2idx[C57]],
                    gts[smp2idx[DBA]],
                    ac,
                    an,
                    cons,
                    impact,
                    gene,
                    change,
                    tx,
                ]))))
