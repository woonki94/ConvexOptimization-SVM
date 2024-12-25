function projected = simplexProjectionD(z, d)

    % Reference:
    %   L. Adam, V. Mácha,
    %   "Projections onto the canonical simplex with additional linear inequalities" 
    %   URL: https://arxiv.org/pdf/1905.03488

    %Compute h7 as presented on theorem 3.8
    h7 = @(lambda) sum(min(max(z - lambda, 0), d)) - 1;

    lambda_d = min(z - d); %lowerbound
    lambda_0 = max(z - 0); %upperbound for lambda

    %search for the lambda that makes h7(lambda) =0
    lambda_opt = fzero(h7, [lambda_d, lambda_0]);

    %clip by 0 and d
    projected = min(max(z - lambda_opt, 0), d);
end