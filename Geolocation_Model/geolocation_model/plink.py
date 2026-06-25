from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def log(message: str) -> None:
    print(message, flush=True)


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        fail(f"Missing {label}: {path}")


def resolve_executable(explicit: str | None, env_name: str, names: Sequence[str]) -> str:
    if explicit:
        path = Path(explicit)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        found = shutil.which(explicit)
        if found:
            return found
        fail(f"Executable not found or not executable for --{env_name.lower()}: {explicit}")
    env_value = os.environ.get(env_name)
    if env_value:
        path = Path(env_value)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        found = shutil.which(env_value)
        if found:
            return found
        fail(f"{env_name} is set but not executable/found: {env_value}")
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    fail(f"Could not find executable. Tried env {env_name} and names: {', '.join(names)}")


def run_cmd(cmd: Sequence[object], *, cwd: Path | None = None, log_path: Path | None = None) -> None:
    log("+ " + " ".join(str(x) for x in cmd))
    started = time.time()
    result = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - started
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout, encoding="utf-8")
    if result.stdout:
        print(result.stdout[-6000:], flush=True)
    if result.returncode != 0:
        suffix = f" See {log_path}" if log_path else ""
        fail(f"Command failed with exit code {result.returncode} after {elapsed:.1f}s.{suffix}")
    log(f"Command completed in {elapsed:.1f}s")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def read_ind(path: Path) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw in enumerate(handle, start=1):
            parts = raw.split()
            if not parts:
                continue
            if len(parts) < 3:
                fail(f"Malformed .ind line {line_no} in {path}")
            rows.append((parts[0], parts[1], parts[2]))
    if not rows:
        fail(f"No samples found in .ind: {path}")
    return rows


def write_subset_snp(src_snp: Path, dst_snp: Path, max_snps: int) -> int:
    n = 0
    dst_snp.parent.mkdir(parents=True, exist_ok=True)
    with src_snp.open("r", encoding="utf-8", errors="ignore") as src, dst_snp.open("w", encoding="utf-8") as dst:
        for raw in src:
            if not raw.strip():
                continue
            if max_snps > 0 and n >= max_snps:
                break
            dst.write(raw)
            n += 1
    if n == 0:
        fail(f"SNP subset is empty from {src_snp}")
    return n


def convert_eigenstrat(
    *,
    convertf_bin: str,
    geno: Path,
    snp: Path,
    ind: Path,
    out_prefix: Path,
    output_format: str,
    packedped: bool = False,
    newsnpname: Path | None = None,
) -> Tuple[Path, Path, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if packedped:
        geno_out = out_prefix.with_suffix(".bed")
        snp_out = out_prefix.with_suffix(".bim")
        ind_out = out_prefix.with_suffix(".fam")
    else:
        geno_out = out_prefix.with_suffix(".geno")
        snp_out = out_prefix.with_suffix(".snp")
        ind_out = out_prefix.with_suffix(".ind")
    par = out_prefix.with_suffix(f".{output_format.lower()}.par")
    lines = [
        f"genotypename: {geno}",
        f"snpname: {snp}",
        f"indivname: {ind}",
        f"outputformat: {output_format}",
        f"genooutfilename: {geno_out}",
        f"snpoutfilename: {snp_out}",
        f"indoutfilename: {ind_out}",
        f"genotypeoutname: {geno_out}",
        f"snpoutname: {snp_out}",
        f"indivoutname: {ind_out}",
        "familynames: NO",
        "outputgroup: YES",
        "checksizemode: NO",
    ]
    if newsnpname is not None:
        lines.append(f"newsnpname: {newsnpname}")
    par.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_cmd([convertf_bin, "-p", par], log_path=out_prefix.with_suffix(".convertf.log"))
    for produced in (geno_out, snp_out, ind_out):
        require_file(produced, f"convertf {output_format} output")
    return geno_out, snp_out, ind_out


def prepare_plink_bed(
    *,
    convertf_bin: str,
    plink_bin: str,
    geno: Path,
    snp: Path,
    ind: Path,
    work_dir: Path,
    max_samples: int,
    max_snps: int,
) -> Tuple[Path, List[str], Dict[str, object]]:
    original_n_samples = count_lines(ind)
    original_n_snps = count_lines(snp)
    all_sample_ids = [sid for sid, _, _ in read_ind(ind)]
    sample_ids = list(all_sample_ids)
    source_geno, source_snp, source_ind = geno, snp, ind
    conversion_inputs: Dict[str, object] = {
        "geno": str(geno),
        "snp": str(snp),
        "ind": str(ind),
        "original_n_samples": original_n_samples,
        "original_n_snps": original_n_snps,
    }

    if max_snps > 0:
        subset_dir = work_dir / "subset_inputs"
        subset_snp = subset_dir / "subset.snp"
        subset_snps = write_subset_snp(snp, subset_snp, max_snps)
        subset_prefix = subset_dir / "subset_packed"
        log(f"Creating SNP smoke subset: samples={len(sample_ids)} snps={subset_snps}")
        source_geno, source_snp, source_ind = convert_eigenstrat(
            convertf_bin=convertf_bin,
            geno=geno,
            snp=snp,
            ind=ind,
            out_prefix=subset_prefix,
            output_format="PACKEDANCESTRYMAP",
            newsnpname=subset_snp,
        )
        conversion_inputs.update({"smoke_geno": str(source_geno), "smoke_snp": str(source_snp), "smoke_ind": str(source_ind)})

    plink_prefix = work_dir / "plink_input" / "dataset"
    bed, bim, fam = convert_eigenstrat(
        convertf_bin=convertf_bin,
        geno=source_geno,
        snp=source_snp,
        ind=source_ind,
        out_prefix=plink_prefix,
        output_format="PACKEDPED",
        packedped=True,
    )
    if max_samples > 0:
        sample_ids = all_sample_ids[:max_samples]
        keep_path = work_dir / "plink_input" / "sample_keep.txt"
        fam_pairs: Dict[str, Tuple[str, str]] = {}
        with fam.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                parts = raw.split()
                if len(parts) >= 2:
                    fam_pairs[parts[1]] = (parts[0], parts[1])
        missing_keep = [sid for sid in sample_ids if sid not in fam_pairs]
        if missing_keep:
            fail(f"PLINK FAM missing keep samples. Examples: {', '.join(missing_keep[:10])}")
        with keep_path.open("w", encoding="utf-8") as handle:
            for sid in sample_ids:
                fid, iid = fam_pairs[sid]
                handle.write(f"{fid}\t{iid}\n")
        kept_prefix = work_dir / "plink_input" / "dataset_kept"
        run_cmd([plink_bin, "--bfile", plink_prefix, "--allow-no-sex", "--keep", keep_path, "--make-bed", "--out", kept_prefix], log_path=kept_prefix.with_suffix(".plink_keep.log"))
        plink_prefix = kept_prefix
        bed, bim, fam = kept_prefix.with_suffix(".bed"), kept_prefix.with_suffix(".bim"), kept_prefix.with_suffix(".fam")
        for produced in (bed, bim, fam):
            require_file(produced, "PLINK kept output")

    conversion_inputs.update({"bed": str(bed), "bim": str(bim), "fam": str(fam), "n_samples_used": len(sample_ids), "n_snps_used": count_lines(source_snp)})
    return plink_prefix, sample_ids, conversion_inputs


def run_plink_ld_pca(
    *,
    plink_bin: str,
    bfile_prefix: Path,
    out_dir: Path,
    window: int,
    step: int,
    r2: float,
    pca_dims: int,
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prune_prefix = out_dir / "plink_prune"
    run_cmd([plink_bin, "--bfile", bfile_prefix, "--allow-no-sex", "--indep-pairwise", window, step, r2, "--out", prune_prefix], log_path=out_dir / "plink_prune.log")
    prune_in = prune_prefix.with_suffix(".prune.in")
    require_file(prune_in, "PLINK prune.in")
    if count_lines(prune_in) == 0:
        fail(f"PLINK prune list is empty: {prune_in}")
    pca_prefix = out_dir / "plink_pca"
    run_cmd([plink_bin, "--bfile", bfile_prefix, "--allow-no-sex", "--extract", prune_in, "--pca", pca_dims, "--out", pca_prefix], log_path=out_dir / "plink_pca.log")
    eigenvec = pca_prefix.with_suffix(".eigenvec")
    eigenval = pca_prefix.with_suffix(".eigenval")
    require_file(eigenvec, "PLINK PCA eigenvec")
    require_file(eigenval, "PLINK PCA eigenval")
    return prune_in, eigenvec, eigenval


def load_plink_eigenvec(eigenvec: Path, sample_ids: Sequence[str], pca_dims: int) -> np.ndarray:
    rows: Dict[str, List[float]] = {}
    with eigenvec.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            parts = raw.strip().split()
            if not parts or parts[0].upper() in {"FID", "#FID"}:
                continue
            if len(parts) < 2 + pca_dims:
                fail(f"PCA row has {len(parts) - 2} PCs, expected {pca_dims}: {raw[:120]}")
            rows[parts[1]] = [float(x) for x in parts[2 : 2 + pca_dims]]
    missing = [sid for sid in sample_ids if sid not in rows]
    if missing:
        fail(f"PCA eigenvec missing {len(missing)} sample IDs. Examples: {', '.join(missing[:10])}")
    arr = np.asarray([rows[sid] for sid in sample_ids], dtype=np.float32)
    if arr.shape != (len(sample_ids), pca_dims):
        fail(f"Unexpected PCA feature shape: {arr.shape}, expected {(len(sample_ids), pca_dims)}")
    if not np.isfinite(arr).all():
        fail("PCA feature matrix contains non-finite values")
    return arr
