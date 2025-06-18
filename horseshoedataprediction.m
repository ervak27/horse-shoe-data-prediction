close all;
clear all;
p = readtable('HorseShoe (1).csv', 'ReadVariableNames', false);

% Display basic information about data
summary(p);
size(p);

dataInputs = p{:, 1:4}';
dataOutputs = p{:, 5}';

% Normalize inputs
normalizedinputs = normalize(dataInputs);

% Initialize results table to store all configurations and performances
resultsTable = table('Size', [0, 8], 'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double', 'string', 'string'}, ...
                     'VariableNames', {'TrainingFunction', 'HiddenL1', 'HiddenL2', 'Epochs', 'LearningRate', 'AveragePerf', 'PerformFcn', 'DivideFcn'});

% Parameters for cross-validation and network training
k = 5; % Number of folds
numofsamples = size(normalizedinputs, 2);
indices = crossvalind('Kfold', numofsamples, k); % Indices for k-fold cross-validation

% Define parameter ranges (reduced for faster testing)
trainingFncts = ["traingd", "trainlm"];   % Smaller set for faster testing
epochsList = [50, 100];                   % Fewer epochs options
learningRates = [0.01, 0.05];             % Fewer learning rates
performFcns = ["mse"];                    % Single performance function to simplify
divideFcns = ["dividerand"];              % Single divide function to simplify

% Grid search over each parameter combination
tic
for i = 1:length(trainingFncts)
    for j = 1:5
        hiddenUnitsL1 = 5 * j;
        for k = 1:5
            hiddenUnitsL2 = 5 * k;
            for e = 1:length(epochsList)
                for lr = 1:length(learningRates)
                    for pf = 1:length(performFcns)
                        for df = 1:length(divideFcns)
                            foldPerf = zeros(1, k); % Store performance for each fold

                            % Display progress
                            disp(['Training Function: ' trainingFncts(i) ', Hidden Layers: ' num2str(hiddenUnitsL1) ' & ' num2str(hiddenUnitsL2) ...
                                  ', Epochs: ' num2str(epochsList(e)) ', Learning Rate: ' num2str(learningRates(lr))]);

                            % Perform k-fold cross-validation
                            for fold = 1:k
                                % Split data into training and test sets for this fold
                                testindices = (indices == fold);
                                trainindices = ~testindices;
                                trainInputs = normalizedinputs(:, trainindices);
                                trainOutputs = dataOutputs(:, trainindices);
                                testInputs = normalizedinputs(:, testindices);
                                testOutputs = dataOutputs(:, testindices);

                                % Create and configure the neural network
                                net = fitnet([hiddenUnitsL1, hiddenUnitsL2], trainingFncts(i));
                                net.trainParam.showWindow = true;  % Show the training window
                                net.trainParam.epochs = epochsList(e);
                                net.trainParam.lr = learningRates(lr);
                                net.performFcn = performFcns(pf);
                                net.divideFcn = divideFcns(df);
                                net.divideParam.trainRatio = 0.7;
                                net.divideParam.valRatio = 0.15;
                                net.divideParam.testRatio = 0.15;

                                % Train the network
                                [net, tr] = train(net, trainInputs, trainOutputs);

                                % Test the network on the test subset
                                annOutputs = net(testInputs);
                                foldPerf(fold) = perform(net, annOutputs, testOutputs); % Store performance for this fold
                            end

                            % Calculate average performance across folds
                            averageperf = mean(foldPerf);

                            % Append configuration and performance results to resultsTable
                            resultsTable = [resultsTable; {trainingFncts(i), hiddenUnitsL1, hiddenUnitsL2, ...
                                                           epochsList(e), learningRates(lr), averageperf, ...
                                                           performFcns(pf), divideFcns(df)}];
                        end
                    end
                end
            end
        end
    end
end
toc

% Display results
disp(resultsTable);

% Sort results by average performance to find the best configuration
sortedresults = sortrows(resultsTable, 'AveragePerf', 'ascend');
bestConfig = sortedresults(1, :);
disp('Best Configuration: ');
disp(bestConfig);

% Use the best configuration to train the final model
bestTrainingFnct = bestConfig.TrainingFunction{1};
bestHiddenLayerSizes = [bestConfig.HiddenL1, bestConfig.HiddenL2];
bestEpochs = bestConfig.Epochs;
bestLearningRate = bestConfig.LearningRate;
bestPerformFcn = bestConfig.PerformFcn{1};
bestDivideFcn = bestConfig.DivideFcn{1};

% Initialize and train the final network on the full dataset
finalnn = fitnet(bestHiddenLayerSizes, bestTrainingFnct);
finalnn.trainParam.showWindow = true;        % Show the training window for the final model
finalnn.trainParam.epochs = bestEpochs;
finalnn.trainParam.lr = bestLearningRate;
finalnn.performFcn = bestPerformFcn;
finalnn.divideFcn = bestDivideFcn;
finalnn.divideParam.trainRatio = 0.7;
finalnn.divideParam.valRatio = 0.15;
finalnn.divideParam.testRatio = 0.15;

% Train the final model on full dataset
finalnn = train(finalnn, normalizedinputs, dataOutputs);

% Test the final model
finalOutput = finalnn(normalizedinputs);
finalPerf = perform(finalnn, finalOutput, dataOutputs);
disp('Performance of Final Model on Full Dataset: ');
disp(finalPerf);
