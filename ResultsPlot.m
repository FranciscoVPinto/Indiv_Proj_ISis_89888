% MATLAB Code: Bar Plot for Model Performance Comparison

% Define models
models = {'Logistic Regression', 'Random Forest', 'SVM', 'Fuzzy Logic (Takagi-Sugeno)', 'DNN', 'MLP ([100, 50])', 'Hybrid (ANFIS)'};

% Define performance metrics for each model
accuracy = [87.47, 96.37, 95.89, 85.3, 96.4, 97.3, 87.3];
recall = [85.65, 96.13, 95.85, 83.2, 95.1, 97.3, 83.6];
precision = [88.65, 96.51, 95.86, 86.0, 96.9, 97.3, 90.0];
f1_score = [87.12, 96.32, 95.85, 84.6, 96.0, 97.3, 86.7];
kappa = [74.92, 92.73, 91.79, 78.5, 92.1, 94.6, 74.6];


% Combine data for grouped bar plot
performance_data = [accuracy; recall; precision; f1_score; kappa]';

% Create grouped bar plot
figure;
b = bar(performance_data, 'grouped');

% Customize bar colors
colors = lines(size(performance_data, 2));
for i = 1:size(performance_data, 2)
    b(i).FaceColor = colors(i, :);
end

% Set axis labels and title
set(gca, 'XTickLabel', models, 'XTickLabelRotation', 30, 'FontSize', 10);
ylabel('Percentage (%)', 'FontSize', 12);
xlabel('Models', 'FontSize', 12);
title('Comparison of Model Evaluation Metrics', 'FontSize', 14);

% Add legend
legend({'Accuracy', 'Recall', 'Precision', 'F1-Score', 'Cohen''s Kappa'}, ...
    'Location', 'northwest', 'FontSize', 10);

% Add grid and adjust y-axis
grid on;
ylim([70 100]);

