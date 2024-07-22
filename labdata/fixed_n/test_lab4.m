addpath(genpath("/zhome/dd/4/109414/Validationstudy/accusleep_v4/"))
outdir = '/zhome/dd/4/109414/Validationstudy/accusleep_v4/labdata/models/fixed_n/balanced/lab4/';
load /zhome/dd/4/109414/Validationstudy/accusleep_v4/labdata/fileList_test_Maiken.mat
load ~/Validationstudy/accusleep_v4/labdata/models/fixed_n/balanced/lab4/best_model_epoch_9.mat
% load model !
nFiles        = size(fileList,1);
all_pred      = [];
all_labels    = [];
all_labs      = [];

SR            = 128; 
epochLen      = 4; 
epochs        = 9; 
minBoutLen    = 5; 

allMetrics    = zeros(nFiles, 5, 3); % For n subjects, 5 metrics, 3 classes

for i = 1:nFiles
    
    data   = struct;
    data.a = load(fileList{i,1});
    data.b = load(fileList{i,2});
    data.c = load(fileList{i,3});
    
    fieldNamesA = fieldnames(data.a);
    fieldNamesC = fieldnames(data.c);
    
    EEG    = data.a.(fieldNamesA{1});
    EMG    = data.b.EMG; 
    labels = data.c.(fieldNamesC{1}); 
    
    if all([sum(labels==1)>=3, sum(labels==2)>=3, sum(labels==3)>=3])
        calibrationData = createCalibrationData(EEG, EMG, labels, SR, epochLen);
        pred = AccuSleep_classify(EEG, EMG, net, SR, epochLen, calibrationData, minBoutLen);
        
        if size(pred,1)>1 % if pred is a column vector => row 
           pred = pred'; 
        else 
        end 
        
        all_labels = [all_labels labels];
        all_pred   = [all_pred   pred];
        all_labs   = [all_labs   fileList{i,1}];
    end 

end 
save(strcat(outdir,'labels.mat'),'all_labels')
save(strcat(outdir,'preds.mat'),'all_pred')
save(strcat(outdir,'labs.mat'),'all_labs')
