#!/bin/bash

args=''
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="$args \"${i//\"/\\\"}\""
done
echo $args
ls
if [ "$args" == "" ]; then args="/bin/bash"; fi

tmp=/tmp/$USER/$$
if [[ "$SLURM_TMPDIR" != "" ]]; then
    tmp="$SLURM_TMPDIR/miopen/$$"
fi
mkdir -p $tmp

if [[ "$(hostname -s)" =~ ^g[r,v,a,h] ]]; then nv="--nv"; fi

# if [[ "$(hostname -s)" =~ ^g[m] ]]; then nv="--rocm"; fi

singularity \
    exec $nv \
    --bind $tmp:$HOME/.config/miopen \
    --overlay /scratch/bf996/singularity_containers/openclip_env_cuda_n.ext3:ro \
    --overlay /scratch/projects/hegdelab/bf996/datasets/arboretum_rare_combined.sqf:ro \
    --overlay /scratch/projects/hegdelab/bf996/datasets/arboretum_test_set.sqf:ro \
    /scratch/work/public/singularity/rocm5.4.2-ubuntu22.04.2.sif \
    /bin/bash -c "
    source /ext3/env.sh;
    $args 
"