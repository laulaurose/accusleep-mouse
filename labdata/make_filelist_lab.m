%% For testing 
clear all; 
clc; 

%addpath(genpath("/Volumes/T9/DATA_v2/data_v2/"))
%I       = "/Volumes/T9/DATA_v2/data_v2/"; 
I       = "/work3/laurose/accusleep/data_v2/";
I = "/work3/laurose/accusleep/labdata/raw/"
outf    = ["fileList_test_Alessandro","fileList_test_Antoine","fileList_test_Kornum","fileList_test_Maiken","fileList_test_Sebastian"] ;
exp_all = {["Alessandro-cleaned-data/","Antoine-cleaned-data/","Kornum-cleaned-data_v2/","Maiken-cleaned-data/", "Sebastian-cleaned-data/"]};
idx_all = {["W*","MB*","*-*","*_*","(*)*"]};

exp      = exp_all{1}; 
idx      = idx_all{1};

%fileList = cell(0, 3); % Initialize fileList as an empty cell array
    
for k = 1:length(exp) % loops experiments 
    disp(exp(k))
    subf     = dir(strcat(I,exp(k),idx(k))); 
    temp2    = cell(length(subf),3);
    %fileList = cell(0, 3); % Initialize fileList as an empty cell array        
    for j = 1:length(subf)
        disp(subf(j).name)
        disp(strcat(fullfile(subf(j).folder,subf(j).name),"/accusleep/EEG*.mat"))
        eeg_path = dir(strcat(fullfile(subf(j).folder,subf(j).name),"/accusleep/EEG*.mat"));
            
        if length(eeg_path)>1
           randomInteger = randi([1, length(eeg_path)]);
           temp2{j,1} = strcat(fullfile(subf(j).folder,subf(j).name), "/accusleep/EEG",num2str(randomInteger),".mat");
           temp2{j,2} = strcat(fullfile(subf(j).folder,subf(j).name), "/accusleep/EMG.mat");
           temp2{j,3} = strcat(fullfile(subf(j).folder,subf(j).name), "/accusleep/labels_accusleep.mat");
    
        else
           temp2{j,1} = strcat(fullfile(subf(j).folder,subf(j).name), "/accusleep/EEG1.mat");
           temp2{j,2} = strcat(fullfile(subf(j).folder,subf(j).name), "/accusleep/EMG.mat");
           temp2{j,3} = strcat(fullfile(subf(j).folder,subf(j).name), "/accusleep/labels_accusleep.mat");
        end  

     end
     %fileList = [fileList; temp2];
     fileList = temp2;
     save(outf(k),'fileList'); 
end 
        
    %save(outf(1),'fileList');
