clear all; 
clc; 

%addpath(genpath("/work3/laurose/accusleep/data/")) % path to data  
addpath(genpath("/zhome/dd/4/109414/Validationstudy/accusleep_v2/")) % path to scripts 
I = dir("/work3/laurose/accusleep/data/Mouse*");
disp("starting")

fileList = cell(20,3);


for k = 1:length(I)
    
    subf = dir(strcat(I(k).folder,"/",I(k).name,"/Da*"));

    for j = 4:5
        tempcell{j-3,1} = strcat(subf(1).folder,"/",subf(j).name,"/EEG.mat");
        tempcell{j-3,2} = strcat(subf(1).folder,"/",subf(j).name,"/EMG.mat");
        tempcell{j-3,3} = strcat(subf(1).folder,"/",subf(j).name,"/labels4.mat");
    end


    % Calculate the indices to insert tempcell into fileList
    idx_start = (k - 1) * size(tempcell, 1) + 1;
    idx_end = k * size(tempcell, 1);
    
    % Insert tempcell into fileList
    fileList(idx_start:idx_end, :) = tempcell;

end
disp("saving")

save('fileList_test.mat', 'fileList');

