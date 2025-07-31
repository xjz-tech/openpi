#!/bin/bash
#SBATCH --job-name=build_openpi
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=build_openpi-%j.out

module load singularity
export SINGULARITY_CACHEDIR=/scratch/$USER/sif_cache
export SINGULARITY_TMPDIR=/scratch/$USER/tmp
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

cd /nfs/turbo/coe-vkamat/openpi
singularity build --fakeroot openpi.sif openpi.def
