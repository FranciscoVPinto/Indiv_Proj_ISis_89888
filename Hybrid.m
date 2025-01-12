clc
clear all

%% Load data and create train-test sets
Train = readtable('dataTrain.csv');
Test = readtable('dataTest.csv');

% Split features (X) and labels (Y)
X_train = table2array(Train(:, 1:end-1));  % All columns except 'STATUS'
Y_train = table2array(Train(:, end));     % 'STATUS' column as label
X_test = table2array(Test(:, 1:end-1));   % All columns except 'STATUS'
Y_test = table2array(Test(:, end));      % 'STATUS' column as label

%% Function to convert predictions to binary classification
convert_to_binary = @(Y_pred) double(Y_pred >= 0.5);  % Ensure output is numeric

%% Function to calculate metrics
function [accuracy, recall, precision, f1] = calc_metrics(Y_true, Y_pred)
    Y_true = double(Y_true);
    Y_pred = double(Y_pred);
    confusion = confusionmat(Y_true, Y_pred);
    TP = confusion(2,2);
    TN = confusion(1,1);
    FP = confusion(1,2);
    FN = confusion(2,1);
    
    accuracy = (TP + TN) / (TP + TN + FP + FN);
    recall = TP / (TP + FN);
    precision = TP / (TP + FP);
    f1 = 2 * (precision * recall) / (precision + recall);
end

%% Grid Search for Best Number of Clusters
best_num_clusters = 0;
best_accuracy = 0;
best_model = [];
best_metrics = [];

for num_clusters = 2:10  % Search from 2 to 10 clusters
    opt = genfisOptions('FCMClustering', 'FISType', 'sugeno');
    opt.NumClusters = num_clusters;
    ts_model = genfis(X_train, Y_train, opt);
    
    % Evaluate on the test set
    Y_pred = evalfis(ts_model, X_test);
    Y_pred_binary = convert_to_binary(Y_pred);
    
    % Calculate metrics
    [accuracy, recall, precision, f1] = calc_metrics(Y_test, Y_pred_binary);
    
    fprintf('Num Clusters: %d, Accuracy: %4.3f, Recall: %4.3f, Precision: %4.3f, F1-Score: %4.3f\n', num_clusters, accuracy, recall, precision, f1);
    
    % Check for best accuracy
    if accuracy > best_accuracy
        best_accuracy = accuracy;
        best_num_clusters = num_clusters;
        best_model = ts_model;
        best_metrics = [accuracy, recall, precision, f1];
    end
end

fprintf('Best Number of Clusters: %d with Accuracy: %4.3f\n', best_num_clusters, best_accuracy);

%% Plot Membership Functions for Best Model
figure;
plotmf(best_model, 'input', 1);
title(sprintf('Membership Functions for Feature 1 (Best Model with %d Clusters)', best_num_clusters));
xlabel('Feature 1 Value');
ylabel('Membership Degree');

%% Tune Best Model using ANFIS
[in, out, rule] = getTunableSettings(best_model);
anfis_model = tunefis(best_model, [in; out], X_train, Y_train, tunefisOptions("Method", "anfis"));

%% Evaluate Tuned Model
Y_pred_final = evalfis(anfis_model, X_test);
Y_pred_final = convert_to_binary(Y_pred_final);
[final_accuracy, final_recall, final_precision, final_f1] = calc_metrics(Y_test, Y_pred_final);

fprintf('Final Accuracy after ANFIS Tuning: %4.3f\n', final_accuracy);

%% Plot Performance Comparison
metrics_names = {'Accuracy', 'Recall', 'Precision', 'F1-Score'};
initial_metrics = best_metrics;
tuned_metrics = [final_accuracy, final_recall, final_precision, final_f1];

figure;
bar_handle = bar([initial_metrics; tuned_metrics]', 'grouped', 'BarWidth', 0.4);  
set(gca, 'xticklabel', metrics_names);
ylabel('Metric Value');
title('Performance Comparison Before and After ANFIS Tuning');
legend({'Best Initial Model', 'After ANFIS Tuning'}, 'Location', 'best');

% Add values on each bar with spacing
xtips1 = bar_handle(1).XEndPoints;
ytips1 = bar_handle(1).YEndPoints;
labels1 = string(round(bar_handle(1).YData, 3));
text(xtips1, ytips1 + 0.01, labels1, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

xtips2 = bar_handle(2).XEndPoints;
ytips2 = bar_handle(2).YEndPoints;
labels2 = string(round(bar_handle(2).YData, 3));
text(xtips2, ytips2 + 0.01, labels2, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
