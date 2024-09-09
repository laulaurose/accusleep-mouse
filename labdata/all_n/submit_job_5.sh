#!/bin/bash
#BSUB -q gpua100
#BSUB -J all_lab_se
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 48:00
##BSUB -u laurarose@sund.ku.dk 
#BSUB -N 
#BSUB -o jobfiles/Output_%J.out 
#BSUB -e jobfiles/Output_%J.err 

#ml load matlab/R2022a
#matlab nodesktop < train_accusleep.m
/appl/matlab/9150/bin/matlab -nodisplay -batch test_lab5 -logfile myjoboutput.txt

