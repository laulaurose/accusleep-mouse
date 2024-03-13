function [net,info] = trainnet(varargin)
% trainnet   Train a neural network
%   trainedNet = trainnet(DATA, NET, LOSS, OPTIONS) trains and returns a
%   network trainedNet from training data in DATA.
%
%   trainedNet = trainnet(PREDICTORS, RESPONSES, NET, LOSS, OPTIONS) trains and returns a
%   network trainedNet from training PREDICTORS and RESPONSES.
%
%   The LOSS function handle syntax is
%      loss = fun(Y1,...,Yn,T1,...,Tm)
%   where Yi is the ith network output and Tj is the jth training target. The
%   outputs and targets are formatted dlarrays.
%
%   [trainedNet, INFO] = trainnet(__) additionally returns information on
%   training progress in INFO.
%
%   Input Arguments:
%      DATA       - training data
%                   datastore
%      PREDICTORS - predictors
%                   numeric array | cell array
%      RESPONSES  - responses
%                   numeric array | categorical array | cell array
%      NET        - network to train
%                   dlnetwork | Layer array
%      LOSS       - loss function
%                   "crossentropy" | "binary-crossentropy" | "mean-squared-error" |
%                   "mean-absolute-error" | "huber" | "mse" | "mae" |
%                   "l2loss" | "l1loss" | function handle
%      OPTIONS    - training options
%                   trainingOptions object
%
%   Output Arguments:
%      trainedNet - trained network
%                   dlnetwork
%      INFO       - training information
%                   TrainingInfo
%
%   Example:
%      predictors = rand(500,1);
%      responses = 10.*predictors - 1;
%      layers = [featureInputLayer(1)
%          fullyConnectedLayer(1)];
%      options = trainingOptions("sgdm",...
%          InitialLearnRate = 0.5,...
%          VerboseFrequency = 10);
%      [net,info] = trainnet(predictors,responses,layers,"mean-squared-error",options);
%
%   See also: dlnetwork, trainingOptions

%   Copyright 2022-2023 The MathWorks, Inc.

narginchk(4,5);
% Guess which kind of syntax the user was intending based on which input
% matches the layers input.
if iIsLayersArgument(varargin{2})  % try to match {data, layers, loss, options}
    narginchk(4,4);
    data = varargin{1};
elseif iIsLayersArgument(varargin{3}) % try to match {predictors, responses, layers, loss, options}
    narginchk(5,5);
    data = varargin(1:2);
else
    error(message('deep:train:InvalidNetwork'));
end
[net, loss, options] = deal(varargin{end-2:end});

try
    net = deep.internal.train.validateLayersAndConstructNetwork(net);
    loss = deep.internal.train.validateAndStandardizeLoss(loss);
    deep.internal.train.validateOptions(options);
    
    [mbq,networkInfo,ds] = deep.internal.train.createMiniBatchQueue(net,data,...
        MiniBatchSize = options.MiniBatchSize,...
        SequenceLength = options.SequenceLength, ...
        SequencePaddingValue = options.SequencePaddingValue, ...
        SequencePaddingDirection = options.SequencePaddingDirection,...
        InputDataFormats = options.InputDataFormats,...
        TargetDataFormats = options.TargetDataFormats,...
        PartialMiniBatch = "discard");

    [net,loss] = deep.internal.train.prepareForTraining(mbq,ds,net,loss,options);

    [networkInfo, outputDataFormats] = getOutputFormats(networkInfo, net, options.InputDataFormats);
    metricCollection = ...
        deep.internal.train.MetricFactory.createMetricCollection(...
        options.Metrics, string(net.OutputNames),...
        outputDataFormats, options.TargetDataFormats);

    deep.internal.train.validateUniqueMetricAndLossNames(metricCollection, loss);

    [net,info] = deep.internal.train.trainnet(mbq, net, loss, options, ...
        MetricCollection = metricCollection, ...
        NetworkInfo = networkInfo);
catch err
    nnet.internal.cnn.util.rethrowDLExceptions(err);
end
end

function tf = iIsLayersArgument(layers)
tf = isa(layers, 'nnet.cnn.layer.Layer') || isa(layers, 'dlnetwork');
end

