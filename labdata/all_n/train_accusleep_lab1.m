addpath(genpath("/zhome/dd/4/109414/Validationstudy/accusleep_v2/"))

clc; 
clear all; 

%imageLocation1 = '/work3/laurose/accusleep/data/labAlessandro/training_images_2024-01-25_16-02-37/';
imageLocation2 = '/work3/laurose/accusleep/data/labAntoine/training_images_2024-01-25_16-17-23/';
imageLocation3 = '/work3/laurose/accusleep/data/labKornum/training_images_2024-01-25_16-17-51/';
imageLocation4 = '/work3/laurose/accusleep/data/labMaiken/training_images_2024-01-25_16-02-33/';
imageLocation5 = '/work3/laurose/accusleep/data/l

[net,info] = AccuSleep_train_lab(imageLocation2,imageLocation3,imageLocation4,imageLocation5); 
save('/zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/labdata/models/fixed_n/balanced/lab1/net.mat', 'net');
outdir = '/zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/labdata/models/fixed_n/balanced/lab1/';
load fileList_test_Alessandro.mat

nFiles        = size(fileList,1);
all_pred      = [];
all_labels    = [];

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
        
        [recall,precision, f1score,accuracy,b_accuracy] = cal_metrics(labels,pred);
        
        % Store the metrics in the array
        met_recall(i,:)    = recall;
        met_precision(i,:) = precision;
        met_f1score(i,:)   = f1score;
        met_accuracy(i,:)  = accuracy;
        met_baccuracy(i,:) = b_accuracy;
    
    
        % til at lave confuision matri 
        all_labels = [all_labels labels];
        all_pred   = [all_pred   pred];
    end 

end 

% Convert to table for display
met_recall(i+1,:) = mean(met_recall); 
met_recall(i+2,:) = std(met_recall)/sqrt(nFiles); 

met_precision(i+1,:) = mean(met_precision); 
met_precision(i+2,:) = std(met_precision)/sqrt(nFiles); 

met_f1score(i+1,:) = mean(met_f1score); 
met_f1score(i+2,:) = std(met_f1score)/sqrt(nFiles); 

met_accuracy(i+1,:) = mean(met_accuracy); 
met_accuracy(i+2,:) = std(met_accuracy)/sqrt(nFiles); 

met_baccuracy(i+1,:) = mean(met_baccuracy); 
met_baccuracy(i+2,:) = std(met_baccuracy)/sqrt(nFiles); 

% Concatenate to one table W | N | R 
all_metrics = [met_recall(:,2),met_precision(:,2),met_f1score(:,2),met_accuracy(:,2),met_baccuracy(:,2),...
               met_recall(:,3),met_precision(:,3),met_f1score(:,3),met_accuracy(:,3),met_baccuracy(:,3),...
               met_recall(:,1),met_precision(:,1),met_f1score(:,1),met_accuracy(:,1),met_baccuracy(:,1)]; 

% Create initial row names as a cell array
%rows = arrayfun(@(x) ['Row' num2str(x)], 1:nFiles, 'UniformOutput', false);

% Append additional strings to the cell array
%rows = [rows, {'mean'}, {'sem'}];


%T = array2table(all_metrics, 'VariableNames', {'W_recall', 'W_precision', 'W_f1score','W_accuracy','W_baccuracy',...
%                                               'N_recall', 'N_precision', 'N_f1score','N_accuracy','N_baccuracy',...
%                                               'R_recall', 'R_precision', 'R_f1score','R_accuracy','R_baccuracy'},...
%                                               'RowNames', rows);
T = array2table(all_metrics, 'VariableNames', {'W_recall', 'W_precision', 'W_f1score','W_accuracy','W_baccuracy',...
                                               'N_recall', 'N_precision', 'N_f1score','N_accuracy','N_baccuracy',...
                                               'R_recall', 'R_precision', 'R_f1score','R_accuracy','R_baccuracy'});


writetable(T, strcat(outdir,'mytable.csv'));
save(strcat(outdir,'predictions_f1.mat'), 'all_pred')
save(strcat(outdir,'labels.mat'),'all_labels')



function [recall,precision, f1score,accuracy,b_accuracy] = cal_metrics(labels,pred)
    % 1st argument are true labels 2nd argument are predictions 
    assert(length(pred)==length(labels))

    for i = 1:3 % loop across class
        TP = sum((labels==i)&(pred==i)); 
        FP = sum((labels~=i)&(pred==i)); % når pred siger positiv men det var en anden klasse  
        FN = sum((labels==i)&(pred~=i)); % når vi har en positiv vi ikke har predicteret som positiv 
        TN = sum((labels~=i)&(pred~=i));

        recall(i)      = TP/(TP+FN);
        precision(i)   = TP/(TP+FP);
        specificity(i) = TN/(TN+FP);
        if (precision(i)>0) & (recall(i)>0) 
            f1score(i) = 2*(precision(i)*recall(i))/(precision(i)+recall(i)); 
        else 
            f1score(i) = 0; 
        end 
        accuracy(i)    = (TP+TN)/(TP+FP+FN+TN); 
        b_accuracy(i)  = (recall(i)+specificity(i))/2;

    end 
    

end

