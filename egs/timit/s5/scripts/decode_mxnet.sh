#!/bin/bash

# Copyright 2012-2013 Karel Vesely, Daniel Povey
# 	    2015 Yu Zhang
# Apache 2.0

# Begin configuration section.
nnet= # Optionally pre-select network to use for getting state-likelihoods
feature_transform= # Optionally pre-select feature transform (in front of nnet)
model= # Optionally pre-select transition model

stage=0 # stage=1 skips lattice generation
nj=4
cmd=run.pl
max_active=7000 # maximum of active tokens
min_active=200 #minimum of active tokens
max_mem=50000000 # limit the fst-size to 50MB (larger fsts are minimized)
beam=13.0 # GMM:13.0
latbeam=8.0 # GMM:6.0
acwt=0.10 # GMM:0.0833, note: only really affects pruning (scoring is on lattices).
scoring_opts="--min-lmwt 1 --max-lmwt 10"
skip_scoring=false
use_gpu_id=-1 # disable gpu
#parallel_opts="-pe smp 2" # use 2 CPUs (1 DNN-forward, 1 decoder)
parallel_opts= # use 2 CPUs (1 DNN-forward, 1 decoder)
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

graphdir=$1
decode_dir=$2
data=$3
expdir=`dirname $decode_dir`; # The model directory is one level up from decoding directory.
split_dir=$decode_dir/split$nj;

mkdir -p $decode_dir/log
[[ -d $split_dir && $decode_dir/feats.scp -ot $split_dir ]] || split_data.sh $decode_dir $nj || exit 1;
echo $nj > $decode_dir/num_jobs

if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
  if [ -z $iter ]; then model=$expdir/final.mdl;
  else model=$expdir/$iter.mdl; fi
fi

for f in $model $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "decode_mxnet.sh: no such file $f" && exit 1;
done


# check that files exist
for f in $split_dir/1/feats.scp $model $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# Create the feature stream:
feats="scp:$split_dir/JOB/feats.scp"


# Run the decoding in the queue
if [ $stage -le 0 ]; then
  $cmd $parallel_opts JOB=1:$nj $decode_dir/log/decode.JOB.log \
    copy-feats $feats ark:- \| \
    latgen-faster-mapped --min-active=$min_active --max-active=$max_active --max-mem=$max_mem --beam=$beam --lattice-beam=$latbeam \
    --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    $model $graphdir/HCLG.fst ark:- "ark:|gzip -c > $decode_dir/lat.JOB.gz" || exit 1;
fi

echo "start scoring"

# Run the scoring
if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $decode_dir || exit 1;
fi

exit 0;
