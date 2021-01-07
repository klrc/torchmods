#!/bin/sh

######################################
#                                    #
#           Magic by sh              #
#                                    #
######################################

# check if there is new tasks
source activate dl
while ! [[ -z `ls -A ../input` ]]
do
    cd ../input
    for file in ./*
    do  
        # clean virtual env
        rm ../vm/*
        # decompress task & chroot
        full_file_name=${file##*/}
        file_name=${full_file_name%.tar.bz2}
        tar -jxf $file -C ../vm/
        rm $file
        # run task
        cd ../vm
        echo True >../server/awake
        nohup python -u main.py >history 2>&1 & pid=$!
        echo $pid >../server/pid
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] running task ${file_name}, pid=${pid}"
        wait $pid
        # compress task outputs
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] task ${file_name} finished, compress outputs"
        echo False >../server/awake
        tar -jcf ../output/${file_name}.tar.bz2 .
        # recover dir
        cd ../input
    done
    # recover dir
    cd ../server
done
