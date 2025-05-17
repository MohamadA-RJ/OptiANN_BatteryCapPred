% Define the optimization function
function best_hyperparameters = optimize_hyperparameters(Train_Input, Train_Output, k)
    % Define the objective function
    fun = @(x) myObjectiveFunction(x, Train_Input, Train_Output, k);

    % Define the variables to be optimized
    min_learning_rate = 0.0001;
    max_learning_rate = 0.1;
    min_mu_max = 1e-20;
    max_mu_max = 0.1;

    vars = [
        optimizableVariable('numHiddenUnits',[1, 40],'Type','integer'),
        optimizableVariable('numHiddenLayers',[1, 2],'Type','integer'),
        optimizableVariable('learningRate', [log(min_learning_rate), log(max_learning_rate)]),
        optimizableVariable('dropoutProb',[0, 0.1]),
        optimizableVariable('mu_max', [min_mu_max, max_mu_max])
    ];

    initial_layers = 1;
    initial_hidden_units = 1; % Initial guess for the number of hidden units
    initial_learning_rate = log(0.001); % Initial guess for the learning rate
    initial_dropout_prob = 0.1; % Initial guess for the dropout probability
    initial_mu_max = 0.01; % Initial guess for mu_max
    
    initial_points = table(initial_hidden_units, initial_layers, initial_learning_rate, initial_dropout_prob, initial_mu_max);

    maxIterations = 100; % Maximum number of iterations to avoid infinite loop
    iteration = 0;
    RmseThreshold = 0.05; % 5% RMSE threshold
    combinedLoss = Inf; % Initialize combined loss

    while combinedLoss > RmseThreshold && iteration < maxIterations
        iteration = iteration + 1;
        
        % Perform Bayesian optimization for hyperparameter tuning
        results = bayesopt(fun, vars, 'MaxObjectiveEvaluations', 100, 'Verbose', 1, 'InitialX', initial_points, 'AcquisitionFunctionName', 'expected-improvement-per-second-plus');
        best_hyperparameters = results.XAtMinObjective;

        numEvaluations = results.NumObjectiveEvaluations;
        numHiddenUnits = best_hyperparameters.numHiddenUnits;
        numHiddenLayers = best_hyperparameters.numHiddenLayers;
        learningRate = exp(best_hyperparameters.learningRate);
        dropoutProb = best_hyperparameters.dropoutProb;
        mu_max = best_hyperparameters.mu_max;

        % Calculate the combined loss using the best hyperparameters
        combinedLoss = myObjectiveFunction(best_hyperparameters, Train_Input, Train_Output, k);
        
        % Display the current iteration and combined loss
        disp(['Iteration: ', num2str(iteration), ' Combined Loss: ', num2str(combinedLoss)]);

        % Create a table
        optTable = table(numEvaluations, numHiddenUnits, numHiddenLayers, learningRate, dropoutProb, mu_max);

        % Display the table
        disp(optTable);
    end
    
    if combinedLoss <= RmseThreshold
        disp('Optimization successful: Combined loss is below 5% RMSE');
    else
        disp('Maximum iterations reached: Combined loss is still above 5% RMSE');
    end
end

% Define the objective function with k-fold cross-validation
function loss = myObjectiveFunction(x, Train_Input, Train_Output, k)
    % Convert hyperparameters to appropriate format
    numHiddenUnits = x.numHiddenUnits;
    numHiddenLayers = x.numHiddenLayers;
    learningRate = exp(x.learningRate); % Exponentiate the log-transformed variable
    dropoutProb = x.dropoutProb;
    mu_max = x.mu_max;
    
    % Define the number of samples
    numSamples = size(Train_Input, 1);
    
    % Initialize array to store performance metrics
    trainLosses = zeros(1, k);
    valLosses = zeros(1, k);
    
    % Create cross-validation indices
    cv = cvpartition(numSamples, 'KFold', k);
    
    % Perform cross-validation
    for i = 1:k
        % Get training and validation indices for current fold
        trainIndices = find(cv.training(i));
        valIndices = find(cv.test(i));
        
        % Extract training and validation data
        X_train = Train_Input(trainIndices, :);
        y_train = Train_Output(trainIndices);
        X_val = Train_Input(valIndices, :);
        y_val = Train_Output(valIndices);
        
        % Create the neural network Architecture
        netFNN = feedforwardnet(repmat(numHiddenUnits, 1, numHiddenLayers));

        % Set the network training parameters
        netFNN.trainParam.epochs = 1000;
        netFNN.trainParam.min_grad = 1e-30;
        netFNN.divideParam.trainRatio = 0.8;
        netFNN.divideParam.valRatio = 0.2;
        netFNN.divideParam.testRatio = 0;
        netFNN.trainParam.max_fail = 1e+100; % Set to infinity to disable early stopping based on validation checks
        netFNN.trainParam.lr = learningRate;  % Set the learning rate
        netFNN.divideFcn = 'divideind';
        netFNN.divideParam.trainInd = trainIndices;
        netFNN.divideParam.valInd = valIndices;
        netFNN.divideParam.testInd = [];
        
        % Set the network architecture
        netFNN.layers{1}.transferFcn = 'tansig';
        for j = 1:(numHiddenLayers-1)
            netFNN.layers{j+1}.transferFcn = 'tansig';
        end
        netFNN.layers{numHiddenLayers+1}.transferFcn = 'purelin';
        
        % Train the neural network
        [netFNN, tr] = train(netFNN, X_train', y_train', 'useparallel', 'no');
        
        % Evaluate the model performance on the training and validation sets
        trainLoss = sqrt(tr.best_perf);
        valLoss = sqrt(tr.best_vperf);
        
        % Store losses for current fold
        trainLosses(i) = trainLoss;
        valLosses(i) = valLoss;
    end
    
    % Calculate mean training and validation losses across all folds
    meanTrainLoss = mean(trainLosses);
    meanValLoss = mean(valLosses);
    
    % Combine training and validation losses (e.g., sum or average)
     combinedLoss = meanTrainLoss + meanValLoss;


    
    % Return the combined loss as the objective to minimize
    loss = combinedLoss;
end
