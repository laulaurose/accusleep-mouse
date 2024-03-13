function [net] = train_dlnetwork(imdsTrain,imdsValidation,CheckpointPath,modeltype_,isWCE_,fold,w_vec)
% LAURA ROSE 
% This function will train a deep learning model for sleep stage
% classifiation in mice. 
% imdsTrain and imdsValidatios are datastores 
% CheckpointPath is the pathway that should be used for storing the model 
% modeltype = whether its accusleep or labdata 
% isWCE whether its should a weighted crossentropy or regular crossentropy 

img            = readimage(imdsTrain,1);
imsize         = size(img);
disp(imsize)

% Loading and prepping dataset 
augimdsValidation = augmentedImageDatastore([imsize(1) imsize(2)],imdsValidation);


% Defining network 
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

net = dlnetwork(layers); 

% Setting Training parameters 
classes                 = categories(imdsTrain.Labels);
numEpochs               = 10;
miniBatchSize           = 256;
LearnRateDropFactor     = 0.8500; 
currentLearnRate        = 0.015;
momentum                = 0.9;

mbqValidation = minibatchqueue(augimdsValidation, ...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB", ...
    DispatchInBackground=0);

% validation labels
TValidation = onehotencode(imdsValidation.Labels,2);
TValidation = TValidation';
valfiles    = next_files_val(imdsValidation.Files,modeltype_);

% set iterations parameters 
velocity              = [];
numObservationsTrain  = numel(imdsTrain.Files);
numIterationsPerEpoch = floor(numObservationsTrain / miniBatchSize);
numIterations         = numEpochs * numIterationsPerEpoch;

if numIterations <= 3360
    disp("number of iterations: 3360")
    vf = 50;
else 
    vf = floor(numIterations/67); % 3360 / 50
    disp("validation frequency:")
    disp(num2str(vf))
end 


monitor = trainingProgressMonitor(Metrics  = ["TrainingLoss","ValidationLoss","TrainingFScore","ValidationFScore"], ...
                                  Visible  ="off", ...
                                  XLabel   = "Iteration", ...
                                  Status   = "Running",...
                                  Info     = ["LearningRate","Epoch","Iteration","ExecutionEnvironment"]);
                                    
executionEnvironment     = "auto";
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    updateInfo(monitor,ExecutionEnvironment="GPU");
else
    updateInfo(monitor,ExecutionEnvironment="CPU");
end

epoch                = 0;
iteration            = 0;
bestF1Score          = 0;
trainingLosses       = [];
trainingIterations   = [];
validationLosses     = [];
validationIterations = [];

[weights_mat,N_mat] = calc_weights(imdsTrain,modeltype_,w_vec,numObservationsTrain)  ; 
save(strcat(CheckpointPath,fold,'N_mat.mat'),'N_mat')
save(strcat(CheckpointPath,fold,'weights_mat.mat'),'weights_mat')

   
bestF1Score_all = [];
bestIteration   = [];
% Start training 
% Loop over epochs

while epoch < numEpochs && ~monitor.Stop  
    
    epoch = epoch + 1;
    
    %shuffle
    perm             = randperm(length(imdsTrain.Files));
    shuffledFiles    = imdsTrain.Files(perm);
    shuffledLabels   = imdsTrain.Labels(perm);
    files            = next_files(shuffledFiles,miniBatchSize,modeltype_);    

    % Create a new imageDatastore with the shuffled files and labels
    idsShuffled  = imageDatastore(shuffledFiles, 'Labels', shuffledLabels);
    augimdsTrain = augmentedImageDatastore([imsize(1) imsize(2)],idsShuffled);
    
    mbq = minibatchqueue(augimdsTrain, ...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB", ...
    DispatchInBackground=0);
     
    % Loop over mini-batches.
    disp("epoch")
    disp(epoch)
    for ii = 1:numIterationsPerEpoch 
        if ~hasdata(mbq) %[300:2000] % early stopping ! 
            break; 
        end

        iteration = iteration + 1;
        
        % Read mini-batch of data.
        [XX,T]  = next(mbq);
                
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelLoss function and update the network state.
        [loss,gradients,state] = dlfeval(@modelLoss,net,XX,T,files(:,ii),isWCE_,weights_mat);
        net.State = state;
        
        % Update the network parameters using the SGDM optimizer.
        [net,velocity] = sgdmupdate(net,gradients,velocity,currentLearnRate,momentum);
        
        Tdecode        = onehotdecode(T,classes,1);
        scores         = predict(net,XX);
        YY             = onehotdecode(scores,classes,1);
        TrainingFScore = calc_macro_f1score(Tdecode,YY);

        trainingLosses(end+1)     = TrainingFScore;
        trainingIterations(end+1) = iteration;

        % Update the training progress monitor.
        recordMetrics(monitor,iteration,TrainingLoss=loss,TrainingFScore=TrainingFScore);    

        updateInfo(monitor, LearningRate=currentLearnRate, Epoch=string(epoch) + " of " + string(numEpochs), ...
            Iteration=string(iteration) + " of " + string(numIterations));
        
        % Validation loop 
        if mod(iteration, vf) == 0|| (iteration == 1)|| ~hasdata(mbq) % change here to not only validate pr epoch

            [YTest,scoresValidation] = modelPredictions(net,mbqValidation,classes);
            lossValidation           = modelLossVal(net,scoresValidation,TValidation,valfiles,isWCE_,weights_mat);
            ValidationFScore         = calc_macro_f1score(imdsValidation.Labels,YTest);
            
            recordMetrics(monitor,iteration,ValidationLoss=lossValidation, ValidationFScore=ValidationFScore);
            
            verbose(epoch, iteration, TrainingFScore, loss, ValidationFScore, lossValidation, currentLearnRate);
            validationLosses(end+1)     = ValidationFScore;
            validationIterations(end+1) = iteration;
            
            % Check for early stopping
            if ValidationFScore > bestF1Score
                bestF1Score = ValidationFScore;
                bestModelFileName = fullfile(CheckpointPath, ['best_model_epoch_' num2str(epoch) '.mat']);
                save(bestModelFileName, 'net', 'epoch', 'bestF1Score','iteration');
                bestIteration   = [bestIteration, iteration];
                bestF1Score_all = [bestF1Score_all, bestF1Score];
            else
            end

        end

        
    end

    currentLearnRate = currentLearnRate * LearnRateDropFactor;
    %checkpointFileName = fullfile(CheckpointPath, ['checkpoint_epoch_' num2str(epoch) '.mat']);
    %save(checkpointFileName, 'net', 'iteration', 'epoch', 'bestF1Score');
end


fig=figure;
set(fig, 'Position', [100, 100, 800, 200]);
plot(trainingIterations, trainingLosses);
hold on;
plot(validationIterations, validationLosses, 'r-');
xlabel('Iteration');
ylabel('macro FScore');
if ~isempty(bestIteration)
    [~, inde] = max(bestF1Score_all); 
    xline(bestIteration(inde),'--k','LineWidth',2)
else 
end 
xline(bestIteration(inde),'--k')
xline(numIterationsPerEpoch*[1:10])
legend('Training FScore', 'Validation FScore', 'Location', 'Best');
grid on;
hold off;
saveas(fig, strcat(CheckpointPath,'TrainingAndValidationLossPlot.png'));


function p_f = file2lab(p,modeltype_)

   for k = 1:length(p)
       file = p(k); 
       aa = split(file,"/"); 
       a  = aa(end-2); 
       if modeltype_ == "accusleep" 
           if a == "training_images_2024-03-01_18-07-24" 
               p_f(k) = 1; 
           else
               p_f(k) = nan; 
           end 
       elseif modeltype_ == "lab" 
            if a == "training_images_2024-03-01_12-58-15"
               p_f(k) = 1; 
            elseif a == "training_images_2024-03-01_13-47-42"
               p_f(k) = 2; 
            elseif a == "training_images_2024-03-01_14-01-59"
               p_f(k) = 3; 
            elseif a == "training_images_2024-03-01_15-15-46"
               p_f(k) = 4; 
            elseif a == "training_images_2024-03-01_15-21-54"
               p_f(k) = 5;
            else 
               p_f(k) = nan; 
            end 
       else 
       end 
       assert(sum(isnan(p_f))==0)

   end 

end 


function p_f = next_files_val(shuffledFiles,modeltype_)
shuffledFiles  = string(shuffledFiles);
p_f = file2lab(shuffledFiles,modeltype_);
end


function files  = next_files(shuffledFiles,miniBatchSize,modeltype_)
shuffledFiles  = string(shuffledFiles);
cut_          = mod(length(shuffledFiles),256); 
shuffledFiles = shuffledFiles(1:length(shuffledFiles)-cut_); 

t   = [1:miniBatchSize:length(shuffledFiles),length(shuffledFiles)+1];

for j = 1:length(t)-1
    p = shuffledFiles(t(j):t(j+1)-1);
    % mappes til laps 
    p_f = file2lab(p,modeltype_);
    files(:,j) = p_f;
end 

end


function macroF1score = calc_macro_f1score(T,Y)
numClasses = 3; 
C = confusionmat(T, Y);

% Calculate precision, recall, and F1-score for each class
precision = zeros(numClasses, 1);
recall    = zeros(numClasses, 1);
f1score   = zeros(numClasses, 1);

for i = 1:numClasses
    precision(i) = C(i,i) / sum(C(:,i));
    recall(i)    = C(i,i) / sum(C(i,:));
    f1score(i)   = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

% Macro F1-score is the average of F1-scores for all classes
macroF1score = nanmean(f1score);

end


function [loss,gradients,state] = modelLoss(net,X,T,labs,isWCE,weights_mat)
[unique_vals, ~, idx] = unique(labs);
new_vals              = 1:length(unique_vals);
lab_transformed       = new_vals(idx);
 
if size(weights_mat,2)==3
   w_ = weights_mat'; 
else 
   w_ = weights_mat;
end 

[Y,state]      = forward(net,X);

if isWCE==1 
    w_ = dlarray(w_); 
    ce = dlarray(zeros(1, length(lab_transformed)));
    epsi = 10^-8;

    for i = 1:length(lab_transformed) 
        idxx    = lab_transformed(i); 
        weights = w_(:,idxx); 
        ce(i) = sum(-weights.*T(:,i).*log(Y(:,i)+epsi),'all'); 
    end
    	
    weightedLoss = sum(ce,'all')/256;
else 
    weightedLoss = crossentropy(Y+epsi,dlarray(extractdata(T),'CB'));
end


% regularization 
l2Regularization = 1.0000e-04;
x                = net.Learnables.Value{:};
l2loss           = l2Regularization * sum(x.^2,'all');
loss             = weightedLoss + l2loss; 

% Calculate gradients of loss with respect to learnable parameters.
gradients      = dlgradient(loss,net.Learnables);

end


function loss = modelLossVal(net,Y,T,labs,isWCE,weights_mat)

[unique_vals, ~, idx] = unique(labs);
new_vals              = 1:length(unique_vals);
lab_transformed       = new_vals(idx);

if size(weights_mat,2)==3
   w_ = weights_mat'; 
else 
   w_ = weights_mat;
end 
 
if isWCE==1 
    w_ = dlarray(w_); 
    ce = dlarray(zeros(1, length(lab_transformed)));
    epsi = 10^-8;

    for i = 1:length(lab_transformed) 
        idxx    = lab_transformed(i); 
        weights = w_(:,idxx); 
        ce(i) = sum(-weights.*T(:,i).*log(Y(:,i)+epsi),'all'); 
    end
    
    weightedLoss = sum(ce,'all')/length(lab_transformed);
else 
    weightedLoss = crossentropy(Y+epsi,dlarray(T,'CB'));
end

% regularization 
l2Regularization = 1.0000e-04;
x                = net.Learnables.Value{:};
l2loss           = l2Regularization * sum(x.^2,'all');
loss             = weightedLoss + l2loss; 

end 


function [predictions,scores] = modelPredictions(net,mbq,classes)

predictions = [];
scores = [];

% Reset mini-batch queue.
reset(mbq);

% Loop over mini-batches.
while hasdata(mbq)
    X = next(mbq);
    Y = predict(net,X);

    % Make prediction.
    scores = [scores Y];

    % Decode labels and append to output.
    Y = onehotdecode(Y,classes,1)';
    predictions = [predictions;Y];
end

end


function [X,T] = preprocessMiniBatch(dataX,dataT)

X = cat(4,dataX{1:end});

% Extract label data from cell and concatenate.
T = cat(2,dataT{1:end});

% One-hot encode labels.
T = onehotencode(T,1);
end


function verbose(epoch, iteration, TrainFScore, lossTrain, ValidationFScore, lossValidation, currentLearnRate)
    fprintf('Iteration: %d, Epoch: %d\n', iteration, epoch);
    fprintf('Training F1score: %.2f%%, Training Loss: %.4f\n', TrainFScore, lossTrain);
    fprintf('Validation F1score: %.2f%%, Validation Loss: %.4f\n', ValidationFScore, lossValidation);
    fprintf('Learning Rate: %.6f\n', currentLearnRate);
    fprintf('-------------------------------------------------------------\n');
end

function [weights_mat,N_mat] = calc_weights(imdsTrain,modeltype_,w_vec,numObservationsTrain)  
            % calc weights 
            s  = 3; 
            if modeltype_== "accusleep" 
                c = 1; 
            else 
                c = 4; 
            end 
            lablab = imdsTrain.Labels;
            f      = next_files_val(imdsTrain.Files,modeltype_); 
            for kk = 1:length(w_vec)
               lab_vec           = double(lablab(find(f==w_vec(kk))));
               N_mat(kk,:)       = [length(find(lab_vec==1)),length(find(lab_vec==2)),length(find(lab_vec==3))];
               weights_mat(kk,:) = [numObservationsTrain/(N_mat(kk,1)*c*s),...
                                    numObservationsTrain/(N_mat(kk,2)*c*s),...
                                    numObservationsTrain/(N_mat(kk,3)*c*s)];
            end 
            

    end 


end

