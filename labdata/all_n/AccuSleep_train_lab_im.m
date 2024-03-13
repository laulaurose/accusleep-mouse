% Train the network

disp("imbalanced")
% 1) Load data for each imageLocation ! 

imds1 = imageDatastore(imageLocation1, 'IncludeSubfolders',true,'LabelSource','foldernames');
imds2 = imageDatastore(imageLocation2, 'IncludeSubfolders',true,'LabelSource','foldernames');
imds3 = imageDatastore(imageLocation3, 'IncludeSubfolders',true,'LabelSource','foldernames');
imds4 = imageDatastore(imageLocation4, 'IncludeSubfolders',true,'LabelSource','foldernames');

disp("done loading imds")

allFiles1 = imds1.Files;
allFiles2 = imds2.Files;
allFiles3 = imds3.Files;
allFiles4 = imds4.Files;

% permute the data;
randomIndices1 = randperm(length(allFiles1));  
randomIndices2 = randperm(length(allFiles2));
randomIndices3 = randperm(length(allFiles3));
randomIndices4 = randperm(length(allFiles4));

subsetSize1    = length(randomIndices1); 
subsetSize2    = length(randomIndices2); 
subsetSize3    = length(randomIndices3); 
subsetSize4    = length(randomIndices4); 

% Split into training (80%) and validation (20%) set 
train_1 = allFiles1(randomIndices1(1:subsetSize1*0.8));
train_2 = allFiles2(randomIndices2(1:subsetSize2*0.8));
train_3 = allFiles3(randomIndices3(1:subsetSize3*0.8));
train_4 = allFiles4(randomIndices4(1:subsetSize4*0.8));

val_1   = allFiles1(randomIndices1(subsetSize1*0.8+1:end));
val_2   = allFiles2(randomIndices2(subsetSize2*0.8+1:end));
val_3   = allFiles3(randomIndices3(subsetSize3*0.8+1:end));
val_4   = allFiles4(randomIndices4(subsetSize4*0.8+1:end));

imdsTrain1      = imageDatastore(train_1, 'Labels', imds1.Labels(randomIndices1(1:subsetSize1*0.8)));
imdsTrain2      = imageDatastore(train_2, 'Labels', imds2.Labels(randomIndices2(1:subsetSize2*0.8)));
imdsTrain3      = imageDatastore(train_3, 'Labels', imds3.Labels(randomIndices3(1:subsetSize3*0.8)));
imdsTrain4      = imageDatastore(train_4, 'Labels', imds4.Labels(randomIndices4(1:subsetSize4*0.8)));

imdsVal1        = imageDatastore(val_1, 'Labels', imds1.Labels(randomIndices1(subsetSize1*0.8+1:end)));
imdsVal2        = imageDatastore(val_2, 'Labels', imds2.Labels(randomIndices2(subsetSize2*0.8+1:end)));
imdsVal3        = imageDatastore(val_3, 'Labels', imds3.Labels(randomIndices3(subsetSize3*0.8+1:end)));
imdsVal4        = imageDatastore(val_4, 'Labels', imds4.Labels(randomIndices4(subsetSize4*0.8+1:end)));


disp("done splitting it into train and val")

allTrainFiles = [imdsTrain1.Files; imdsTrain2.Files; imdsTrain3.Files; imdsTrain4.Files];
allTrainLabels = [imdsTrain1.Labels; imdsTrain2.Labels; imdsTrain3.Labels; imdsTrain4.Labels];

allValFiles = [imdsVal1.Files; imdsVal2.Files; imdsVal3.Files; imdsVal4.Files];
allValLabels = [imdsVal1.Labels; imdsVal2.Labels; imdsVal3.Labels; imdsVal4.Labels];

imdsTrain = imageDatastore(allTrainFiles, 'Labels', allTrainLabels);
imdsValidation = imageDatastore(allValFiles, 'Labels', allValLabels);

disp("done combining imds from each lab into train and val")


disp(length(imdsTrain.Files)+length(imdsValidation.Files))
% get image size
img = readimage(imds1,1);
imsize = size(img);

% get number of images per label
vf = 50; 

metric = fScoreMetric(AverageType="macro");

disp('print here 1')
% set training options
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.85, ...
    'LearnRateDropPeriod',1, ...
    'InitialLearnRate',0.015, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',vf, ...
    'ValidationPatience',15,...
    'Verbose',true, ...
    'ExecutionEnvironment','auto',...
    'MiniBatchSize',256,...
    'Plots','none', ...
    'Metrics',metric);

layers = [
    imageInputLayer([imsize(1) imsize(2) 1]);%,'AverageImage',ai)
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(3)
    softmaxLayer];

% train
disp('Training network')
[net, trainInfo] = trainnet(imdsTrain,layers,"crossentropy",options);

%disp(length(find(double(imdsTrain.Labels)==1)))
%disp(length(find(double(imdsTrain.Labels)==2)))
%disp(length(find(double(imdsTrain.Labels)==3)))
%N = sum([length(find(double(imdsTrain.Labels)==1)),length(find(double(imdsTrain.Labels)==2)),length(find(double(imdsTrain.Labels)==3))]);
%weights = dlarray([N/(3*length(find(double(imdsTrain.Labels)==1))), N/(3*length(find(double(imdsTrain.Labels)==2))), N/(3*length(find(double(imdsTrain.Labels)==3)))],'C');
%customLossFunction = @(Y, T) crossentropy(Y, T,weights);
%[net, trainInfo] = trainnet(imdsTrain,layers,customLossFunction,options);


