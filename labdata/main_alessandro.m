addpath(genpath("/zhome/dd/4/109414/Validationstudy/accusleep_v2/"))


%imbalanced model training 
clc; 
clear all; 
load fileList_test_Alessandro 
SR = 128;
epochLen = 4;
epochs   = 9;
imageLocation = '/work3/laurose/accusleep/data/labAlessandro/';
[net,info] = AccuSleep_create_images(fileList, SR, epochLen, epochs,imageLocation); 

