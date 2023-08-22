# create temporary model checkpoint directory and create symlinks
CHECKPOINT_TO_PROCESS=checkpoint_1_25
RANK_PATHS=`find /ramyapra/fairseq/checkpoint_test/ -name $CHECKPOINT_TO_PROCESS-rank-*.pt`
TEMP_FOLDER=`mktemp -d`
pushd $TEMP_FOLDER
for m in $RANK_PATHS;
do
    filename=`echo $m | rev | cut -d '/' -f1 | rev | sed 's/-shard[0-9]*//g'` # extract only filename from full path
    ln -s $m ./$filename 
done;
SHARED_PATH=`find /ramyapra/fairseq/checkpoint_test/ -name $CHECKPOINT_TO_PROCESS-shared-shard0.pt`
filename=`echo $SHARED_PATH | rev | cut -d '/' -f1 | rev | sed 's/-shard[0-9]*//g'`
ln -s $SHARED_PATH ./$filename
popd
