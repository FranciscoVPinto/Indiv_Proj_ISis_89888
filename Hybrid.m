%% Load data and create train-test sets
Train = readtable('dataTrain.csv');
Test = readtable('dataTest.csv');

% Split features (X) and labels (Y)
X_train = table2array(Train(:, 1:end-1));  % All columns except 'STATUS'
Y_train = table2array(Train(:, end));     % 'STATUS' column as label
X_test = table2array(Test(:, 1:end-1));   % All columns except 'STATUS'
Y_test = table2array(Test(:, end));      % 'STATUS' column as label

%% Train initial Takagi-Sugeno model
opt = genfisOptions('FCMClustering', 'FISType', 'sugeno');
opt.NumClusters = 5;  % Example number of clusters (can adjust)
ts_model = genfis(X_train, Y_train, opt);

%% Function to convert predictions to binary classification
convert_to_binary = @(Y_pred) double(Y_pred >= 0.5);  % Ensure output is numeric

%% Function to calculate metrics
function [accuracy, recall, precision, f1, kappa] = calc_metrics(Y_true, Y_pred)
    % Ensure both Y_true and Y_pred are numeric
    Y_true = double(Y_true);  % Convert Y_true to numeric if it's not
    Y_pred = double(Y_pred);  % Convert Y_pred to numeric if it's not
    
    % Confusion Matrix
    confusion = confusionmat(Y_true, Y_pred);
    TP = confusion(2,2);  % True Positive
    TN = confusion(1,1);  % True Negative
    FP = confusion(1,2);  % False Positive
    FN = confusion(2,1);  % False Negative
    
    % Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN);
    
    % Recall (Sensitivity)
    recall = TP / (TP + FN);  % Recall = TP / (TP + FN)
    
    % Precision
    precision = TP / (TP + FP);  % Precision = TP / (TP + FP)
    
    % F1 Score
    if (precision + recall) > 0
        f1 = 2 * (precision * recall) / (precision + recall);
    else
        f1 = 0;  % If both precision and recall are zero
    end
    
    % Cohen's Kappa
    P_o = (TP + TN) / (TP + TN + FP + FN);  % Observed Accuracy
    P_e = ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / (TP + TN + FP + FN)^2;  % Expected Accuracy
    kappa = (P_o - P_e) / (1 - P_e);
end

%% Check initial performance on test set
Y_pred_initial = evalfis(ts_model, X_test);
Y_pred_initial = convert_to_binary(Y_pred_initial);  % Convert to binary class

% Calculate performance metrics
[accuracy_initial, recall_initial, precision_initial, f1_initial, kappa_initial] = calc_metrics(Y_test, Y_pred_initial);
fprintf('Initial Accuracy: %4.3f \n', accuracy_initial);
fprintf('Initial Recall: %4.3f \n', recall_initial);
fprintf('Initial Precision: %4.3f \n', precision_initial);
fprintf('Initial F1-Score: %4.3f \n', f1_initial);
fprintf('Initial Cohen''s Kappa: %4.3f \n', kappa_initial);

%% Plot Membership Functions for Initial Model
figure;
plotmf(ts_model, 'input', 1);  % Plot membership functions for the first input feature
title('Membership Functions for Feature 1 (Initial Model)');
xlabel('Feature 1 Value');
ylabel('Membership Degree');
saveas(gcf, 'mf_initial.png');  % Save the plot for LaTeX inclusion

%% Tune initial model using ANFIS
[in, out, rule] = getTunableSettings(ts_model);
anfis_model = tunefis(ts_model, [in; out], X_train, Y_train, tunefisOptions("Method", "anfis"));

%% Check ANFIS tuned model performance
Y_pred_final = evalfis(anfis_model, X_test);
Y_pred_final = convert_to_binary(Y_pred_final);  % Convert to binary class

% Calculate performance metrics
[accuracy_final, recall_final, precision_final, f1_final, kappa_final] = calc_metrics(Y_test, Y_pred_final);
fprintf('Final Accuracy: %4.3f \n', accuracy_final);
fprintf('Final Recall: %4.3f \n', recall_final);
fprintf('Final Precision: %4.3f \n', precision_final);
fprintf('Final F1-Score: %4.3f \n', f1_final);
fprintf('Final Cohen''s Kappa: %4.3f \n', kappa_final);

%% Plot Membership Functions for Tuned Model (ANFIS)
figure;
plotmf(anfis_model, 'input', 1);  % Plot membership functions for the first input feature after ANFIS tuning
title('Membership Functions for Feature 1 (After ANFIS Tuning)');
xlabel('Feature 1 Value');
ylabel('Membership Degree');
saveas(gcf, 'mf_tuned.png');  % Save the plot for LaTeX inclusion

%% Plot Performance Metrics
metrics = [accuracy_initial, recall_initial, precision_initial, f1_initial, kappa_initial; 
           accuracy_final, recall_final, precision_final, f1_final, kappa_final];

metric_names = {'Accuracy', 'Recall', 'Precision', 'F1-Score', 'Cohen''s Kappa'};

figure;
bar(metrics, 'grouped');
set(gca, 'xticklabel', {'Initial Model', 'Tuned Model (ANFIS)'});
ylabel('Metric Value');
title('Performance Metrics Comparison');
legend(metric_names, 'Location', 'best');
saveas(gcf, 'metrics_comparison.png');  % Save the performance metrics comparison plot

%% Conclusion of the model performance
fprintf('The hybrid fuzzy model has been trained and evaluated.\n');
fprintf('The membership functions and coefficients have been visualized and saved for further analysis.\n');
