#!/bin/bash

# This script trains and evaluate LSTM models. There is no
# discriminative training yet.
# In this recipe, MXNet directly read Kaldi features and labels,
# which makes the whole pipline much simpler.

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will
                 #return non-zero return code
set -u           #Fail on an undefined variable

global_config_file=scripts/global.cfg
if [ ! -f  $global_config_file ]; then
    echo "Global config file doesn't exist"
    exit 1
else
    source $global_config_file
fi

kaldi_dataset_root=$kaldi_root/egs/$dataset/s5

cmd=run.pl
# root folder,
expdir=exp
scripts_path=../../../

export KALDI_ROOT=$kaldi_root

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

prefix=$dataset

if [[ ! -d utils ]]; then
    ln -s ${KALDI_ROOT}/egs/${prefix}/s5/utils utils
    ln -s ${KALDI_ROOT}/egs/${prefix}/s5/local local
    ln -s ${KALDI_ROOT}/egs/${prefix}/s5/conf conf
fi


##################################################
# Kaldi generated folder
##################################################

# alignment folder
ali_src=$kaldi_dataset_root/$ali_dir

# decoding graph
graph_src=$kaldi_dataset_root/$graph_dir

# features
train_src=$kaldi_dataset_root/$train_dir
dev_src=$kaldi_dataset_root/$dev_dir

# config file
config=default.cfg
# optional settings,
njdec=$nj
scoring="--min-lmwt 5 --max-lmwt 6"

cp scripts/template.cfg $config
sed -i "s:EPOCH_NUM:${num_epoch}:g" $config
sed -i "s:DATASET:${dataset}:g" $config
sed -i "s:DATA_PREFIX:${kaldi_dataset_root}:g" $config
sed -i "s:OUTPUT_DIM:${ydim}:g" $config

# The device number to run the training
# change to AUTO to select the card automatically
deviceNumber=gpu0
ydim=$((`cat ${graph_src}/num_pdfs`+1))
echo "ydim: " $ydim

# decoding method
method=simple
modelName=
# model
num_epoch=$num_epoch
acwt=0.1
#smbr training variables
num_utts_per_iter=40
smooth_factor=0.1
use_one_sil=true

stage=$stage
. utils/parse_options.sh || exit 1;


###############################################
# Training
###############################################

mkdir -p $expdir
dir=$expdir/data-for-mxnet

# prepare listing data
if [ $stage -le 0 ] ; then
    mkdir -p $dir
    mkdir -p $dir/log
    mkdir -p $dir/rawpost

    # for compressed ali
    num=`cat $ali_src/num_jobs`
    $cmd JOB=1:$num $dir/log/gen_post.JOB.log \
        ali-to-pdf $ali_src/final.mdl "ark:gunzip -c $ali_src/ali.JOB.gz |" \
            ark:- \| ali-to-post ark:- ark,scp:$dir/rawpost/post.JOB.ark,$dir/rawpost/post.JOB.scp || exit 1;
    #num=`cat $ali_src/num_jobs`
    #$cmd JOB=1:$num $dir/log/gen_post.JOB.log \
    #    ali-to-pdf $ali_src/final.mdl ark:$ali_src/ali.JOB.ark \
    #        ark:- \| ali-to-post ark:- ark,scp:$dir/rawpost/post.JOB.ark,$dir/rawpost/post.JOB.scp || exit 1;


    for n in $(seq $num); do
        cat $dir/rawpost/post.${n}.scp || exit 1;
    done > $dir/post.scp
fi

if [ $stage -le 1 ] ; then
    # split the data : 90% train and 10% held-out
    utils/subset_data_dir_tr_cv.sh $train_src ${train_src}_tr90 ${train_src}_cv10

    # generate dataset list
    echo NO_FEATURE_TRANSFORM scp:${train_src}_tr90/feats.scp > $dir/train.feats
    echo scp:$dir/post.scp >> $dir/train.feats

    echo NO_FEATURE_TRANSFORM scp:${train_src}_cv10/feats.scp > $dir/dev.feats
    echo scp:$dir/post.scp >> $dir/dev.feats

    echo NO_FEATURE_TRANSFORM scp:${dev_src}/feats.scp > $dir/test.feats
fi

# generate label counts
if [ $stage -le 2 ] ; then
    $cmd JOB=1:1 $dir/log/gen_label_mean.JOB.log \
        python $scripts_path/make_stats.py --configfile $config --data_train $dir/train.feats \| copy-feats ark:- ark:$dir/label_mean.ark
    echo NO_FEATURE_TRANSFORM ark:$dir/label_mean.ark > $dir/label_mean.feats
fi


# training, note that weight decay is for the whole batch (0.00001 * 20 (minibatch) * 40 (batch_size))
if [ $stage -le 3 ] ; then
    python $scripts_path/train_lstm_proj.py --configfile $config --data_train $dir/train.feats --data_dev $dir/dev.feats --train_prefix $PWD/$expdir/$prefix --train_num_epoch $num_epoch --train_optimizer speechSGD --train_learning_rate 1 --train_context $deviceNumber --train_weight_decay 0.008 --train_show_every 1000
fi

# decoding
if [ $stage -le 4 ] ; then
  cp $ali_src/final.mdl $expdir
  mxnet_string="OMP_NUM_THREADS=1 python $scripts_path/decode_mxnet.py --config $config --data_test $dir/test.feats --data_label_mean $dir/label_mean.feats --train_method $method --train_prefix $PWD/$expdir/$prefix --train_num_epoch $num_epoch --train_context $deviceNumber --train_batch_size 1"
  $scripts_path/decode_mxnet.sh --nj $njdec --cmd $cmd --acwt $acwt --scoring-opts "$scoring" \
    $graph_src $dev_src $expdir/decode_${prefix}_$(basename $dev_src) "$mxnet_string"

fi
