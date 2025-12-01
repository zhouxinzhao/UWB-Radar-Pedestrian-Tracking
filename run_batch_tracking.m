function run_batch_tracking(opts)
%RUN_BATCH_TRACKING Execute all CIR/VAR main scripts and store tracking results.
%
% Optional input:
%   opts.cirOptions - Struct passed to main_CIR* (trainFraction, trainTimeLimit,
%                    testTimeWindow, forceRetrain, rngSeed).
%   opts.varOptions - Struct passed to main_var* with the same fields.
%
% This script runs main_CIR1-4 and main_var1-4 sequentially, collects
% tracking outputs, writes detailed trajectories to CSV files, and prints
% RMSE/MAE/P50/P90 summaries for each run.

    if nargin < 1
        opts = struct;
    end
    if ~isfield(opts, 'cirOptions')
        opts.cirOptions = struct;
    end
    if ~isfield(opts, 'varOptions')
        opts.varOptions = struct;
    end

    cirRuns = {@() main_CIR1(opts.cirOptions), @() main_CIR2(opts.cirOptions), ...
        @() main_CIR3(opts.cirOptions), @() main_CIR4(opts.cirOptions)};
    cirLabels = ["main_CIR1", "main_CIR2", "main_CIR3", "main_CIR4"];
    [cirResults, cirMetrics] = collect_runs(cirRuns, cirLabels);
    writetable(cirResults, 'cir_tracking_results.csv');
    writetable(cirMetrics, 'cir_tracking_metrics.csv');
    display_metrics('CIR', cirMetrics);

    varRuns = {@() main_var1(opts.varOptions), @() main_var2(opts.varOptions), ...
        @() main_var3(opts.varOptions), @() main_var4(opts.varOptions)};
    varLabels = ["main_var1", "main_var2", "main_var3", "main_var4"];
    [varResults, varMetrics] = collect_runs(varRuns, varLabels);
    writetable(varResults, 'var_tracking_results.csv');
    writetable(varMetrics, 'var_tracking_metrics.csv');
    display_metrics('Variance', varMetrics);
end

function [allRows, metricsTable] = collect_runs(runFns, labels)
    allRows = table();
    metricsTable = table('Size', [0 5], ...
        'VariableTypes', {'string','double','double','double','double'}, ...
        'VariableNames', {'run','rmse','mae','p50','p90'});

    for idx = 1:numel(runFns)
        results = runFns{idx}();
        errors = results.error(:);
        estPos = results.estimated_position;
        gtPos = results.ground_truth;
        times = results.time(:);
        runLabel = repmat(labels(idx), numel(errors), 1);

        runRows = table(runLabel, times, estPos(:,1), estPos(:,2), gtPos(:,1), gtPos(:,2), errors, ...
            'VariableNames', {'run','time','est_x','est_y','gt_x','gt_y','error_m'});
        allRows = [allRows; runRows]; %#ok<AGROW>

        metrics = compute_metrics(errors);
        metricsTable = [metricsTable; {labels(idx), metrics.rmse, metrics.mae, metrics.p50, metrics.p90}]; %#ok<AGROW>
    end
end

function metrics = compute_metrics(errors)
    metrics.rmse = sqrt(mean(errors.^2, 'omitnan'));
    metrics.mae = mean(abs(errors), 'omitnan');
    metrics.p50 = prctile(errors, 50);
    metrics.p90 = prctile(errors, 90);
end

function display_metrics(groupName, metricsTable)
    fprintf('\n%s tracking metrics (units: meters)\n', groupName);
    fprintf('---------------------------------------\n');
    for r = 1:height(metricsTable)
        fprintf('%s -> RMSE: %.3f, MAE: %.3f, P50: %.3f, P90: %.3f\n', ...
            metricsTable.run(r), metricsTable.rmse(r), metricsTable.mae(r), metricsTable.p50(r), metricsTable.p90(r));
    end
end
