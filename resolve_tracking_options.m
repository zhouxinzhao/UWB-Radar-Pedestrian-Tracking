function opts = resolve_tracking_options(userOpts)
%RESOLVE_TRACKING_OPTIONS Apply defaults and validation for tracking runs.
%
% Supported fields (all optional):
%   trainFraction  - Portion of data used for training vs. validation (0-1).
%   trainTimeLimit - Upper bound on Dyn_re_tUWB timestamps to include in the
%                    training set. Defaults to 73.8 seconds, matching the
%                    original scripts.
%   forceRetrain   - If true, always retrain the CNN instead of loading a
%                    cached ConvNet_*.mat file.
%   rngSeed        - If provided, used to seed MATLAB's RNG prior to shuffles.

    defaults = struct( ...
        'trainFraction', 0.85, ...
        'trainTimeLimit', 73.8, ...
        'forceRetrain', false, ...
        'rngSeed', [], ...
        'testTimeWindow', []); %#ok<NASGU>

    if nargin < 1 || isempty(userOpts)
        opts = defaults;
    else
        opts = defaults;
        userFields = fieldnames(userOpts);
        for idx = 1:numel(userFields)
            opts.(userFields{idx}) = userOpts.(userFields{idx});
        end
    end

    if ~(opts.trainFraction > 0 && opts.trainFraction < 1)
        error('trainFraction must be in the open interval (0, 1).');
    end

    if ~(isscalar(opts.trainTimeLimit) && isfinite(opts.trainTimeLimit) && opts.trainTimeLimit > 0)
        error('trainTimeLimit must be a positive scalar.');
    end

    if ~isempty(opts.testTimeWindow)
        if ~(isnumeric(opts.testTimeWindow) && numel(opts.testTimeWindow) == 2 && ...
                all(isfinite(opts.testTimeWindow)) && opts.testTimeWindow(1) < opts.testTimeWindow(2))
            error('testTimeWindow must be a 1x2 vector [lower upper] with lower < upper.');
        end
    end
end
