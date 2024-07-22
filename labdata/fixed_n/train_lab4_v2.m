clc; 
clear all; 
subsetSize = 107760*1/4;
addpath(genpath("/zhome/dd/4/109414/Validationstudy/accusleep_v4/"))


imageLocation1 = '/work3/laurose/accusleep/data/labAlessandro/training_images_2024-03-01_12-58-15/';
imageLocation2 = '/work3/laurose/accusleep/data/labAntoine/training_images_2024-03-01_13-47-42/';
imageLocation3 = '/work3/laurose/accusleep/data/labKornum/training_images_2024-03-01_14-01-59/';
imageLocation4 = '/work3/laurose/accusleep/data/labMaiken/training_images_2024-03-01_15-15-46/';
imageLocation5 = '/work3/laurose/accusleep/data/labSebastian/training_images_2024-03-01_15-21-54/';
% 1) Load data for each imageLocation ! 
imds1 = imageDatastore(imageLocation1, 'IncludeSubfolders',true,'LabelSource','foldernames');
imds2 = imageDatastore(imageLocation2, 'IncludeSubfolders',true,'LabelSource','foldernames');
imds3 = imageDatastore(imageLocation3, 'IncludeSubfolders',true,'LabelSource','foldernames');
imds4 = imageDatastore(imageLocation5, 'IncludeSubfolders',true,'LabelSource','foldernames');

disp("done loading imds")
allFiles1 = imds1.Files;
allFiles2 = imds2.Files;
allFiles3 = imds3.Files;
allFiles4 = imds4.Files;

% Randomly sampling;
randomIndices1 = randperm(length(allFiles1), subsetSize);
randomIndices2 = randperm(length(allFiles2), subsetSize);
randomIndices3 = randperm(length(allFiles3), subsetSize);
randomIndices4 = randperm(length(allFiles4), subsetSize);

% Split into training and validation set 
train_1 = allFiles1(randomIndices1(1:floor(subsetSize*0.8)));
train_2 = allFiles2(randomIndices2(1:floor(subsetSize*0.8)));
train_3 = allFiles3(randomIndices3(1:floor(subsetSize*0.8)));
train_4 = allFiles4(randomIndices4(1:floor(subsetSize*0.8)));

val_1   = allFiles1(randomIndices1(floor(subsetSize*0.8)+1:end));
val_2   = allFiles2(randomIndices2(floor(subsetSize*0.8)+1:end));
val_3   = allFiles3(randomIndices3(floor(subsetSize*0.8)+1:end));
val_4   = allFiles4(randomIndices4(floor(subsetSize*0.8)+1:end));

imdsTrain1      = imageDatastore(train_1, 'Labels', imds1.Labels(randomIndices1(1:floor(subsetSize*0.8))));
imdsTrain2      = imageDatastore(train_2, 'Labels', imds2.Labels(randomIndices2(1:floor(subsetSize*0.8))));
imdsTrain3      = imageDatastore(train_3, 'Labels', imds3.Labels(randomIndices3(1:floor(subsetSize*0.8))));
imdsTrain4      = imageDatastore(train_4, 'Labels', imds4.Labels(randomIndices4(1:floor(subsetSize*0.8))));

imdsVal1        = imageDatastore(val_1, 'Labels', imds1.Labels(randomIndices1(floor(subsetSize*0.8)+1:end)));
imdsVal2        = imageDatastore(val_2, 'Labels', imds2.Labels(randomIndices2(floor(subsetSize*0.8)+1:end)));
imdsVal3        = imageDatastore(val_3, 'Labels', imds3.Labels(randomIndices3(floor(subsetSize*0.8)+1:end)));
imdsVal4        = imageDatastore(val_4, 'Labels', imds4.Labels(randomIndices4(floor(subsetSize*0.8)+1:end)));


disp("done splitting it into train and val")

allTrainFiles = [imdsTrain1.Files; imdsTrain2.Files; imdsTrain3.Files; imdsTrain4.Files];
allTrainLabels = [imdsTrain1.Labels; imdsTrain2.Labels; imdsTrain3.Labels; imdsTrain4.Labels];

allValFiles = [imdsVal1.Files; imdsVal2.Files; imdsVal3.Files; imdsVal4.Files];
allValLabels = [imdsVal1.Labels; imdsVal2.Labels; imdsVal3.Labels; imdsVal4.Labels];

imdsTrain = imageDatastore(allTrainFiles, 'Labels', allTrainLabels);
imdsValidation = imageDatastore(allValFiles, 'Labels', allValLabels);

disp("done combining imds from each lab into train and val")

%%%%%%%%%----------------------------------------%%%%%%%%%----------------------------------------%%%%%%%%%----------------------------------------

%CheckpointPath = '/zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/accusleep/models/balanced/';  
CheckpointPath  = '/zhome/dd/4/109414/Validationstudy/accusleep_v4/labdata/models/fixed_n/balanced/lab4/';

modeltype_     = "lab"; 
isWCE_         = 1;
fold           = '4'
w_vec =	[1,2,3,5];
train_dlnetwork(imdsTrain,imdsValidation,CheckpointPath,modeltype_,isWCE_,fold,w_vec); 
