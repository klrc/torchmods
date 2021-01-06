#!/bin/sh

######################################
#                                    #
#           Magic by sh              #
#                                    #
######################################

# check if there is new tasks
source activate dl
while ! [ -z `ls -A ../input` ]
do
    cd ../input
    for file in ./*
    do  
        # clean virtual env
        rm ../vm/*
        # decompress task & chroot
        full_file_name=${file##*/}
        file_name=${full_file_name%.tar.bz2}
        tar -jxvf $file -C ../vm/
        rm $file
        cd ../vm
        mv ${file_name}/* .
        rm ${file_name} -r
        # run task
        echo True >../server/awake
        nohup python -u main.py >history 2>&1 & pid=$!
        echo $pid >../server/pid
        wait $pid
        # compress task outputs
        echo False >../server/awake
        tar -jcvf ../output/${file_name}.tar.bz2 .
        # recover dir
        cd ../input
    done
    # recover dir
    cd ../server
done
