% This Code is used for reading the battery data structure made from the
% Data_Structuring Code Similar to the NASA "Battery Data Set", NASA Ames Prognostics Data Repository
% Then, the goal is to use the DeepLearningToolbox to train a ANN with optimized Hyperparameters via Bayesian Optimization  
%%
% If you use this code please reference it accodring to the assocaited paper: 
% M. A. Raja, W. Kim, W. Kim, S. H. Lim, and S. S. Kim, 
% “Computational micromechanics and machine learning-informed design of composite carbon fiber-based structural battery for multifunctional performance prediction,” 
% ACS Applied Materials &amp; Interfaces, vol. 17, no. 13, pp. 20125–20137, Feb. 2025. doi:10.1021/acsami.4c19073 

%%
% Some parts of this code were built upon previous works of 
% Choi, Y.; Ryu, S.; Park, K.; Kim, H. Machine Learning-Based Lithium-Ion Battery Capacity Estimation Exploiting Multi-Channel Charging Profiles. 
% IEEE Access 2019, 7, 75143– 75152,  DOI: 10.1109/ACCESS.2019.2920932

%%
clc;
clear;
close all;
%%
% Load data
load SPE40_0p1C.mat
%%

% This one is just to access certain cycles, and plot the V/I

InitialTime = SPE40_0p1C.cycles(1).data.test_time/60;
Initial_V = SPE40_0p1C.cycles(1).data.Voltage_measured;
Initial_I = SPE40_0p1C.cycles(1).data.Current_measured;

FinalTime = SPE40_0p1C.cycles(99).data.test_time/60;
Final_V = SPE40_0p1C.cycles(99).data.Voltage_measured;
Final_I = SPE40_0p1C.cycles(99).data.Current_measured;

% Graphing the V,I for the two cycles
figure(1)                      
subplot(211)
plot(InitialTime, Initial_V, 'linewidth', 2), hold on, plot(FinalTime, Final_V, 'r--','linewidth', 2)
hold off, legend('Initial (1st Cycle)', 'Selected Cycle (100th Cycle)'), ylabel('Voltage(V)')
ylim([0, 2.6]); % Set y-axis limits for voltage
grid on

subplot(212)
plot(InitialTime, Initial_I, 'linewidth', 2), hold on, plot(FinalTime, Final_I, 'r--', 'linewidth', 2)
hold off, ylabel('Current(A)'), 
ylim([-0.001,0.001]); % Set y-axis limits for current
grid on


%%
% Extracting the Discharge Capacity Data from the Batteries
Discharge_Cap = extract_discharge(SPE40_0p1C);
max_discharge_capacity = max(Discharge_Cap);

% Plotting the Capacities Degradation as a function of cycles
figure
plot(Discharge_Cap)
grid on
xlabel('Cycle'), ylabel('Capacity(Ah)')
legend('SPE40_0p1C')
title('Capacity Degradations in Cycle')

% Extracting and Organizing the charge input Data (V,I) 
% also selecting the data selection interval (i.e., 10 points per each cycle)
Charging_input = extract_charge_preprocessing(SPE40_0p1C);   

% Initial Rated Capacity for the Battery 
Initial_Cap = max_discharge_capacity;

% Min-max normalization
[xB5, yB5, ym5, yr5] = minmax_norm(Charging_input, Initial_Cap, Discharge_Cap);
%%
% ANN Train/Validation/Training Datasets
Train_Input = xB5; % The charge Input Matrix that has 10 samples for V,I and Normalized
Train_Output = yB5;
%% Block for: Bayesian optimization for hyperparameter tuning
% Call the optimization function to find the best hyperparameters
% k = 5; % Number of folds for cross-validation
% best_hyperparameters = optimize_hyperparameters(Train_Input, Train_Output, k);
% disp ('Optimization Done')
% Unpack the best hyperparameters
% numHiddenUnits = best_hyperparameters.numHiddenUnits;
% numHiddenLayers = best_hyperparameters.numHiddenLayers;
% learningRate = best_hyperparameters.learningRate;
% dropoutProb = best_hyperparameters.dropoutProb;
% mu_max = best_hyperparameters.mu_max;
%%
% Using the Optimized best_hyperparameters from the Bayesian Optimization
numHiddenUnits = 25;
numHiddenLayers = 2;
learningRate = 0.040784;
dropoutProb = 0.031828;
mu_max = 0.055636;
%%

% Create the neural network Architecture
netFNN = feedforwardnet(repmat(numHiddenUnits, 1, numHiddenLayers));
netFNN.trainParam.epochs = 1000; 
netFNN.trainParam.min_grad = 1e-30;
netFNN.trainParam.max_fail = 1e+100;
netFNN.trainParam.mu = 1e-20;
netFNN.trainParam.mu_max = mu_max;
netFNN.divideParam.trainRatio = 0.8;
netFNN.divideParam.valRatio = 0.1;
netFNN.divideParam.testRatio = 0.1;
netFNN.trainParam.lr = learningRate; 
netFNN.trainParam.dropoutFraction = dropoutProb;

% Selection of the data for the Training/Validation/Testing
numSamples = size(Train_Input, 1);     
numTrain = round(netFNN.divideParam.trainRatio * numSamples);
numVal = round(netFNN.divideParam.valRatio * numSamples);
numTest = round(netFNN.divideParam.testRatio * numSamples);
trainIndices = 1:numTrain;
valIndices = (numTrain + 1):(numTrain + numVal);
testIndices = (numTrain + numVal + 1):numSamples;
netFNN.divideFcn = 'divideind';
netFNN.divideParam.trainInd = trainIndices;
netFNN.divideParam.valInd = valIndices;
netFNN.divideParam.testInd = testIndices;

    % Set the network architecture
    netFNN.layers{1}.transferFcn = 'tansig';
    for i = 1:(numHiddenLayers-1)
        netFNN.layers{i+1}.transferFcn = 'tansig';
    end
    netFNN.layers{numHiddenLayers+1}.transferFcn = 'purelin';

% Train the neural network
[netFNN, tr] = train(netFNN, Train_Input', Train_Output', 'useparallel', 'no');

% Making Predictions using the Trained Models
Train_Predicted = netFNN(Train_Input(tr.trainInd, :)');
Val_Predicted = netFNN(Train_Input(tr.valInd, :)');
Test_Predicted = netFNN(Train_Input(tr.testInd, :)');

%%
% Denormalization for graphical output
realvalue = yB5 * yr5 + ym5;
realvalue = realvalue';
Train_Predicted = Train_Predicted * yr5 + ym5;
Val_Predicted = Val_Predicted * yr5 + ym5;
Test_Predicted = Test_Predicted * yr5 + ym5;
Assembled_Pred = [Train_Predicted Val_Predicted Test_Predicted];
Assembled_Pred = Assembled_Pred';

% Visualize the prediction result
figure, hold on, grid on,
plot(realvalue) 
plot(Assembled_Pred, '*--')
title('Capacity Estimation using V, I (SBE40_0p1C)')
xlabel('Cycle'), ylabel('Capacity(Ah)')
legend('Real Value', 'Predicted')
%%

% Predicition Performance Mean Absolute Percentage Error (MAPE)
Prediction_Cycles_True_data = realvalue(end-(numTest-1):end);
mape_predicited_cycles = sum(abs((Prediction_Cycles_True_data - Test_Predicted) ./ Prediction_Cycles_True_data)) * 100 / numel(Prediction_Cycles_True_data);

% Performance of Prediction 
disp(['MAPE for predicted cycles: ', num2str(mape_predicited_cycles)]);
% Performance of Training 
mse_training = tr.best_perf;
disp(['MSE used during training: ', num2str(mse_training)]);