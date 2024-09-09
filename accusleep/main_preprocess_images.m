
addpath(genpath("/zhome/dd/4/109414/Validationstudy/accusleep_v4/AccuSleep/"))
imageLocation  ='/work3/laurose/accusleep/data/prep/';
imds           = imageDatastore(imageLocation, 'IncludeSubfolders',true,'LabelSource','foldernames');
CheckpointPath = '/Users/qgf169/Documents/MATLAB/AccuSleep/dlnetwork_accusleep/';  
load fileList.mat
SR            = 512; 
epochLen      = 4; 
epochs        = 9; 
minBoutLen    = 5; 
weights       = [.1 .35 .55];
preprocess_images(fileList, SR, epochLen, epochs, imageLocation,weights)
