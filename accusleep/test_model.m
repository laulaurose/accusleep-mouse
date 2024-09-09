
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

outdir = "/zhome/dd/4/109414/Validationstudy/accusleep_v4/accusleep/models/balanced/";
load fileList_test.mat
load /zhome/dd/4/109414/Validationstudy/accusleep_v4/accusleep/models/balanced/best_model_epoch_9.mat

nFiles        = size(fileList,1);
all_pred      = [];
all_labels    = [];
all_labs      = [];

for i = 1:nFiles
    
    data   = struct;
    disp(fileList{i,1})
    data.a = load(fileList{i,1});
    data.b = load(fileList{i,2});
    data.c = load(fileList{i,3});

    fieldNamesA = fieldnames(data.a);
    fieldNamesC = fieldnames(data.c);
    
    EEG    = data.a.(fieldNamesA{1});
    EMG    = data.b.EMG; 
    labels = data.c.(fieldNamesC{1}); 

    calibrationData = createCalibrationData(EEG, EMG, labels, SR, epochLen);
    
    pred = AccuSleep_classify(EEG, EMG, net, SR, epochLen, calibrationData, minBoutLen);
    
    %% til at lave confuision matri 
    all_labels = [all_labels labels];
    all_pred   = [all_pred   pred];
    all_labs   = [all_labs   fileList{i,1}];
end 

save(strcat(outdir,'labels.mat'),'all_labels')
save(strcat(outdir,'preds.mat'),'all_pred')
save(strcat(outdir,'labs.mat'),'all_labs')

