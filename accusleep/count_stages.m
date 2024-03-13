clc; 
clear all;

%exps     = ["Alessandro","Antoine", "Kornum", "Maiken", "Sebastian"];

%outf     = "/zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/accusleep/labdata/";


%testsets = {['/zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/labdata/fileList_test_Alessandro.mat'],...
%            ['/zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/labdata/fileList_test_Antoine.mat'],...
%            ['/zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/labdata/fileList_test_Kornum.mat'],...
%            ['/zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/labdata/fileList_test_Maiken.mat'],...
%            ['/zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/labdata/fileList_test_Sebastian.mat']}; 



load fileList.mat 
count_w = 0; 
count_n = 0; 
count_r = 0; 

for i = 1:length(fileList) % loops across mouse 
    ss  = load(fileList{i,3}); 
   %disp(labels)
    labels = ss.labels; 
    count_r = count_r + length(find(labels==1)); % rem 
    count_w = count_w + length(find(labels==2)); % wake 
    count_n = count_n + length(find(labels==3)); % nrem 
      %disp("hours")
       % disp((count_r+count_w+count_n)/128/3600)
        %disp("--------------------------------")
end 
    
counts(1,:) = [count_w,count_n,count_r];


save("/zhome/dd/4/109414/Validationstudy/accusleep_v2/AccuSleep/accusleep/counts.mat",'counts')
