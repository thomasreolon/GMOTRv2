trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
args=''

## always use these options
tmp=$(<"configs/_general.args")
args="${args} ${tmp}"
tmp=$(<"configs/_paths.args")
args="${args} ${tmp}"

for FILE in outputs/*
do
    if [[ $FILE == *.pth ]]
    then
        echo $FILE
        python3 test.py ${args} --resume $FILE --prob_detect 0.55 --debug
    fi
done
