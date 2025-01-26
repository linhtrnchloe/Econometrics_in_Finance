
clear; 
clc;

data = readtable('StockIndices.xlsx');

stock_codes = {'S_PCOMP', 'JAPDOWA', 'FTSE100', 'DAXINDX', 'FRCAC40'};

for i = 1:length(stock_codes)
    index_name = stock_codes{i}; 
    price = data.(index_name); 
    
    returns = diff(log(price)); 

    % --- Model 1: AR(1)-GARCH(1,1) ---
    model1 = arima(1, 0, 0);
    model1.Variance = garch(1, 1);
    estModel1 = estimate(model1, returns, 'Display', 'off');
    bic1 = summarize(estModel1).BIC;

    % --- Model 2: ARMA(1,1)-GARCH(1,1) ---
    model2 = arima(1, 0, 1);
    model2.Variance = garch(1, 1);
    estModel2 = estimate(model2, returns, 'Display', 'off');
    bic2 = summarize(estModel2).BIC;

    % --- Model 3: ARMA(1,1)-GARCH(2,2) ---
    model3 = arima(1, 0, 1);
    model3.Variance = garch(2, 2);
    estModel3 = estimate(model3, returns, 'Display', 'off');
    bic3 = summarize(estModel3).BIC;

    % --- Model 4: ARMA(1,1)-GARCH(1,1) with Student's t ---
    model4 = arima(1, 0, 1);
    model4.Variance = garch(1, 1);
    model4.Distribution = 't'; 
    estModel4 = estimate(model4, returns, 'Display', 'off');
    bic4 = summarize(estModel4).BIC;

    % --- Compare BICs ---
    fprintf('BIC Comparisons for %s:\n', index_name);
    fprintf('1. AR(1)-GARCH(1,1) vs. ARMA(1,1)-GARCH(1,1): %.2f vs. %.2f\n', bic1, bic2);
    fprintf('2. ARMA(1,1)-GARCH(1,1) vs. ARMA(1,1)-GARCH(2,2): %.2f vs. %.2f\n', bic2, bic3);
    fprintf('3. Gaussian vs. Student-t: %.2f vs. %.2f\n', bic2, bic4);

    % --- Interpret Results ---
    if bic2 < bic1
        disp('ARMA(1,1)-GARCH(1,1) is better than AR(1)-GARCH(1,1).');
    else
        disp('AR(1)-GARCH(1,1) is better than ARMA(1,1)-GARCH(1,1).');
    end

    if bic3 < bic2
        disp('ARMA(1,1)-GARCH(2,2) is better than ARMA(1,1)-GARCH(1,1).');
    else
        disp('ARMA(1,1)-GARCH(1,1) is better than ARMA(1,1)-GARCH(2,2).');
    end

    if bic4 < bic2
        disp('Student-t distribution is better than Gaussian distribution.');
    else
        disp('Gaussian distribution is better than Student-t distribution.');
    end
    
    disp(' ');
end


%%% Interpretation:
%% ARMA(1,1)-GARCH(1,1) is the best model for the S&PCOMP and FTSE100 indices, while AR(1)-GARCH(1,1) performs better for JAPDOWA.
%% ARMA(1,1)-GARCH(2,2) is the best model for FTSE100, DAXINDX, and FRCAC40.
%% The Student-t distribution generally performs better than the Gaussian distribution across all indices.