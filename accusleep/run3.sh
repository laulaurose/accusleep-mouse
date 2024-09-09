
#!/bin/bash
#BSUB -q gpuv100
#BSUB -J accusleep
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -W 24:00
##BSUB -u laurarose@sund.ku.dk 
#BSUB -N 
#BSUB -o jobfiles/Output_%J.out 
#BSUB -e jobfiles/Output_%J.err 

#ml load matlab/R2022a
#matlab nodesktop < main_accusleep.m
/appl/matlab/9150/bin/matlab -nodisplay -batch orig_run3  -logfile myjoboutput.txt
#/appl/matlab/9150/bin/matlab -nodisplay -batch test_lab2 -logfile myjoboutput.txt
#/appl/matlab/9150/bin/matlab -nodisplay -batch test_lab3 -logfile myjoboutput.txt
#/appl/matlab/9150/bin/matlab -nodisplay -batch test_lab4 -logfile myjoboutput.txt
#/appl/matlab/9150/bin/matlab -nodisplay -batch test_lab5 -logfile myjoboutput.txt

