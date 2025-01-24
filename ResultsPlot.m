% MATLAB Code: Bar Plot for Model Performance Comparison

% Define models
models = {'Logistic Regression', 'Random Forest', 'SVM', 'Fuzzy Logic (ANFIS)', 'DNN', 'MLP ([100, 50])'};

% Define performance metrics for each model
accuracy = [87.47, 96.37, 95.89, 87.2, 96.4, 97.3];
recall = [85.65, 96.13, 95.85, 85.0, 95.1, 97.3];
precision = [88.65, 96.51, 95.86, 89.0, 96.9, 97.3];
f1_score = [87.12, 96.32, 95.85, 86.9, 96.0, 97.3];
kappa = [74.92, 92.73, 91.79, 80.0, 92.1, 94.6];

% Combine data for grouped bar plot
performance_data = [accuracy; recall; precision; f1_score; kappa]';

% Create grouped bar plot
figure;
bar(performance_data, 'grouped');
set(gca, 'XTickLabel', models, 'XTickLabelRotation', 45);
ylabel('Percentage (%)');
title('Comparison of Model Evaluation Metrics');
legend({'Accuracy', 'Recall', 'Precision', 'F1-Score', 'Cohen''s Kappa'}, 'Location', 'northwest');
grid on;

% Adjust plot for better readability
ylim([70 100]);
xtickangle(30);
