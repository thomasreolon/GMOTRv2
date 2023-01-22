trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
args=''

## always use these options
tmp=$(<"configs/_general.args")
args="${args} ${tmp}"
tmp=$(<"configs/_paths.args")
args="${args} ${tmp}"

python3 test.py ${args} --resume /home/intern/Desktop/GMOTRv2/outputs/motr/baseline.pth --prob_detect 0.33 --debug
# for FILE in outputs/*
# do
#     if [[ $FILE == *.pth ]]
#     then
#         echo $FILE
#         python3 test.py ${args} --resume $FILE --prob_detect 0.32 --debug
#     fi
# done
