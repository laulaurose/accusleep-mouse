addpath(genpath("/zhome/dd/4/109414/Validationstudy/accusleep_v4/"))
imageLocation  ='/work3/laurose/accusleep/data/prep/training_images_2024-03-01_18-07-24/';
imds           = imageDatastore(imageLocation, 'IncludeSubfolders',true,'LabelSource','foldernames');
CheckpointPath = '/zhome/dd/4/109414/Validationstudy/accusleep_v4/accusleep/models/balanced/';  

modeltype_    = "accusleep"; 
isWCE_        = 1;
fold          = '1'
w_vec         = [1];
SR            = 512; 
epochLen      = 4; 
epochs        = 9; 
minBoutLen    = 5; 

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,"randomize");
train_dlnetwork(imdsTrain,imdsValidation,CheckpointPath,modeltype_,isWCE_,fold,w_vec); 
