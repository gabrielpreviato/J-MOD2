import os
import sys

import subprocess32 as subprocess

file = open("training_status.txt", "a")

#                                 #
#    TREINAMENTOS DE 60 EPOCHS    #
#                                 #
#try:
#    retcode = subprocess.call("python2.7 train.py" + "--data_set_dir /home/previato/J-MOD2/data/dataset/ --data_train_dirs data1 --data_test_dirs data1 --is_train True --dataset Soccer --is_deploy False --exp_name pre-trained-60 --num_epochs 60 --gpu_memory_fraction 0.75 --resume_training True --weights_path ../weights/jmod2.hdf5", shell=True)
#    print >> file, "pre-trained-60:"
#
#    if retcode < 0:
#        print >>file, "Child was terminated by signal", -retcode
#    else:
#        print >>file, "Child returned", retcode,
#except OSError as e:
#    print >>file, "Execution failed:", e

# try:
#     retcode = subprocess.call(
#         "python2.7 train.py" + " --data_set_dir /home/previato/J-MOD2/data/dataset/ --data_train_dirs data1 --data_test_dirs data1 --is_train True --dataset Soccer --is_deploy False --exp_name non-trained-60 --num_epochs 60 --gpu_memory_fraction 0.75",
#         shell=True)
#     print >> file, "non-trained-60:"
#
#     if retcode < 0:
#         print >>file, "Child was terminated by signal", -retcode
#     else:
#         print >>file, "Child returned", retcode,
# except OSError as e:
#     print >>file, "Execution failed:", e

#                                  #
#    TREINAMENTOS DE 120 EPOCHS    #
#                                  #
try:
    retcode = subprocess.call(
        "python2.7 train.py" + " --data_set_dir /home/previato/dataset/ --data_train_dirs data2 --data_test_dirs data2 --is_train True --dataset Soccer --is_deploy False --exp_name obs-pre-trained-120 --num_epochs 120 --gpu_memory_fraction 0.9 --resume_training True --weights_path ../weights/jmod2.hdf5",
        shell=True)
    print >> file, "pre-trained-120:"

    if retcode < 0:
        print >>file, "Child was terminated by signal", -retcode
    else:
        print >>file, "Child returned", retcode,
except OSError as e:
    print >>file, "Execution failed:", e

try:
    retcode = subprocess.call(
        "python2.7 train.py" + " --data_set_dir /home/previato/dataset/ --data_train_dirs data2 --data_test_dirs data2 --is_train True --dataset Soccer --is_deploy False --exp_name obs-non-trained-120 --num_epochs 120 --gpu_memory_fraction 0.9",
        shell=True)
    print >> file, "non-trained-120:"

    if retcode < 0:
        print >>file, "Child was terminated by signal", -retcode
    else:
        print >>file, "Child returned", retcode,
except OSError as e:
    print >>file, "Execution failed:", e


file.close()
