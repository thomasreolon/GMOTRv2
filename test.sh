args=''
num_nodes=1
for var in "$@"
do
    # get additional arguments
    if [ -a $var ]
    then
        tmp=$(<"$var")
        args="${args} ${tmp}"
    fi

    # if number set it as number of GPUs
    if [[ $var =~ ^[0-9]+$ ]]
    then
        num_nodes=$var
    fi
done

## always use these options
tmp=$(<"configs/_general.args")
args="${args} ${tmp}"
tmp=$(<"configs/_paths.args")
args="${args} ${tmp}"

# run code
if [ $num_nodes -gt "1" ]
then
    python3 -m torch.distributed.launch --nproc_per_node=$num_nodes --use_env test.py $args |& tee -a out.log &
else
    CUDA_LAUNCH_BLOCKING=1 python3 test.py ${args} &
fi

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
read -p "Press [Enter] key to exit..."