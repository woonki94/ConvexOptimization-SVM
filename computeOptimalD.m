function [opt_d] = computeOptimalD(train_data, test_data,d_values, method)

    fprintf("\nSearching for d with Least Classification Error...\n");
    A = train_data.A;
    B = train_data.B;
    X_test = test_data.X_test;
    true_labels = test_data.true_labels;

    % Initialize parameters
    m = size(A, 2);
    n = size(B, 2);
    tolerance = 1e-1; % Threshold for overlap detection
    errors = zeros(size(d_values));
    overlap_flags = false(size(d_values)); % Tracks overlap between convex hulls

    %iterate through possible d's
    for i = 1:length(d_values)
        d = d_values(i);
        %finding d for cvx
        if strcmp(method, 'cvx')
            % CVX-based optimization
            cvx_begin quiet
                variables u(m) v(n)
                minimize(1/2 * square_pos(norm(A * u - B * v, 2)))
                subject to
                    sum(u) == 1;
                    sum(v) == 1;
                    0 <= u <= d;
                    0 <= v <= d;
            cvx_end
        %finding d for projected gradient
        elseif strcmp(method, 'proj')
           [u,v] = projectedGradient(train_data,d,1000);
        %finding d for nesterov
        elseif strcmp(method, 'nesterov')
            [u,v] = projectedNesterov(train_data,d, 1000);
        %finding d for admm
        elseif strcmp(method, 'admm')
            [u,v] = admm(train_data,d, 1000,1000);
        else
            error('Invalid method. Choose "cvx" or "gradient_descent".');
        end

        % Calculate closest points
        closest_point_A = A * u;
        closest_point_B = B * v;

        % Compute distance and check for overlap
        distance = norm(closest_point_A - closest_point_B, 2);
        if distance < tolerance
            overlap_flags(i) = true;
            errors(i) = NaN;
            continue;
        end

        % Compute classification error
        normal_vector = closest_point_A - closest_point_B;
        normal_vector = normal_vector / norm(normal_vector);

        decision_boundary = (closest_point_A + closest_point_B) / 2;
        predicted_labels = sign((X_test' - decision_boundary') * (normal_vector));
        classification_error = sum(predicted_labels ~= true_labels') / length(true_labels);
        errors(i) = classification_error;
    end

    % Select optimal d based on minimum classification error (ignoring overlaps)
    valid_indices = ~overlap_flags;
    valid_d_values = d_values(valid_indices);
    valid_errors = errors(valid_indices);

    [optimal_error, optimal_index] = min(valid_errors);
    opt_d = valid_d_values(optimal_index);

    fprintf('Optimal d: %.6f with Classification Error: %.6f%%\n', opt_d, optimal_error * 100);

    figure;
    plot(valid_d_values, valid_errors * 100, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    xlabel('d');
    ylabel('Classification Error (%)');
    title(['Classification Error vs. d (Method: ', method, ')']);
    grid on;
end
