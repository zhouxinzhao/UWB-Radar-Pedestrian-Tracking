function [trainIdx, testIdx] = split_train_test_by_quarter(totalCount, quarterIdx)
%SPLIT_TRAIN_TEST_BY_QUARTER Split indices into train/test by equal quarters.
%
%   [trainIdx, testIdx] = split_train_test_by_quarter(totalCount, quarterIdx)
%   divides the sequence 1:totalCount into four consecutive quarters and
%   returns the indices for the requested quarter (testIdx) and the
%   remaining indices (trainIdx). quarterIdx must be an integer from 1 to 4.

    if ~(isscalar(totalCount) && totalCount > 0 && floor(totalCount) == totalCount)
        error('totalCount must be a positive integer scalar.');
    end
    if ~(isscalar(quarterIdx) && any(quarterIdx == 1:4))
        error('quarterIdx must be 1, 2, 3, or 4.');
    end

    edges = round(linspace(1, totalCount + 1, 5));
    testIdx = edges(quarterIdx):(edges(quarterIdx + 1) - 1);
    testIdx = testIdx(testIdx <= totalCount);
    trainIdx = setdiff(1:totalCount, testIdx);

    % always return column vectors to avoid row/column indexing issues
    testIdx = testIdx(:);
    trainIdx = trainIdx(:);

    if isempty(testIdx)
        error('Test set is empty for totalCount=%d and quarter=%d.', totalCount, quarterIdx);
    end
    if isempty(trainIdx)
        error('Train set is empty for totalCount=%d and quarter=%d.', totalCount, quarterIdx);
    end
end
