clc; 
clear all;
addpath(genpath("/zhome/dd/4/109414/Validationstudy/accusleep_v2/"))
load /zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/accusleep/models/balanced_resampling/net_f1.mat
disp(net)

exps     = ["Alessandro","Antoine", "Kornum", "Maiken", "Sebastian"];

outf     = "/zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/accusleep/models/balanced_resampling/";
testsets = {['/zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/labdata/fileList_test_Alessandro.mat'],...
            ['/zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/labdata/fileList_test_Antoine.mat'],...
            ['/zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/labdata/fileList_test_Kornum.mat'],...
            ['/zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/labdata/fileList_test_Maiken.mat'],...
            ['/zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/labdata/fileList_test_Sebastian.mat']}; 

SR         = 128;
epochLen   = 4;
epochs     = 9;
minBoutLen = 5; 

all_pred   = [];
all_labels  = [];

for j = 1:5 % loop across LOLO b
    load(testsets{j})
    disp("testing starting")
    nFiles        = size(fileList,1);
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
    met_recall(i+1,:)  = mean(met_recall); 
    met_recall(i+2,:)  = std(met_recall)/sqrt(nFiles); 
    
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
    clear met_recall met_precision met_f1score met_accuracy met_baccuracy
    % Create initial row names as a cell array
   
    
    T = array2table(all_metrics, 'VariableNames', {'W_recall', 'W_precision', 'W_f1score','W_accuracy','W_baccuracy',...
                                                   'N_recall', 'N_precision', 'N_f1score','N_accuracy','N_baccuracy',...
                                                   'R_recall', 'R_precision', 'R_f1score','R_accuracy','R_baccuracy'});

    writetable(T, strcat(outf,exps(j),'mytable.csv'));
    save(strcat(outf,exps(j),'predictions_f1.mat'), 'all_pred')
    save(strcat(outf,exps(j),'labels.mat'),'all_labels')

end 




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

