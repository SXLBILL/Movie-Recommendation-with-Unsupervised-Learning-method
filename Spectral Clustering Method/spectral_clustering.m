function [mse, predicts, MSEs, clusterLabel, eigVec, eigVal] = spectral_clustering(user_mov_rate, W, customerMatch, movieMatch, testdata, Kclu)
    %% Input
    % user_movie_rate is a user_movie_rating matrix
    % W is adjaency matrix
    % customerMatch is a uniqe sequential index mapping to old customer id
    % movieMatch is a unique sequential index mapping to old movie id
    % testdata is data predicting
    % Kclu is the cluster numbers
    %% Output
    % mse is mean suqare error of predicted rating with ture rating
    % predicts is the predicted value
    % MSEs is collection of errors between predicted rating and true rating
    % clusterLabel is the cluster label each customer from training set belong to
    % eigVec is the eigen vector from Lv = lambda*Dv
    % eigVal is the eigen value from Lv = lambda*Dv

    UMR = user_mov_rate;
    %% spectral clustering choosing Kclu
    D = diag(sum(W));
    L = D - W;
    [eigVec, eigVal]=eigs(L,D,size(L,1));
    [eigVal, idx_up] = sort(diag(eigVal));
    eigVec = eigVec(:, idx_up);
    clusterLabel = kmeans(eigVec(:,1:Kclu), Kclu);
    %% calculate MSE
    MSEs = zeros(size(testdata,1),1);
    predicts = zeros(size(testdata,1),1);
    custom = 1:size(customerMatch,1);
    for i=1:length(MSEs)
        user = testdata(i,2);
        mov = testdata(i,1);
        % For prediction
        if(size(testdata,2)>2)
            rating = testdata(i,3);
        else
            rating = NaN;
        end
        % for example user2 is from 1 to 5905; mov2 is from 1 to 10000
        [user2, mov2] = transform(user, mov, customerMatch, movieMatch);
        % if in testdata, we don't have user or movie id then
        % get the average of the same group of user or the same movie
        if(length(mov2)==0)
            sml_custom = setdiff(custom(clusterLabel== cl_label), user2);
            predict = round(mean(mean(UMR(sml_custom, :), 'omitnan'),'omitnan'));
            if(isnan(predict))
                predict = round(mean(mean(UMR,'omitnan'),'omitnan'));
            end
        elseif(length(user2)==0)
            predict = round(mean(UMR(:,mov2),'omitnan'));
        else
            cl_label = clusterLabel(user2);
            % find similar cluster customers and remove self
            sml_custom = setdiff(custom(clusterLabel== cl_label), user2);
            predict = round(mean(UMR(sml_custom, mov2), 'omitnan'));
            if(isnan(predict))
                predict = round(mean(UMR(:,mov2),'omitnan'));
            end
        end
        predicts(i) = predict;
        MSEs(i) = (rating - predict)^2;
    end
    mse = mean(MSEs, 'omitnan');
end