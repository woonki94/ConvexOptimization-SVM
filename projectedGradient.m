
function [u,v,objective_values,time_measure] = projectedGradient(train_data,d, max_iter)
    A= train_data.A;
    B= train_data.B;
    objective_values = [];
    time_measure = [];
    cumulate_time = 0;

    m = size(A, 2);
    n = size(B, 2);

    % Fixed learning rate alpha_u, alpha_v.
    gram_A = A' * A;
    gram_B = B' * B;

    max_eigen_A = max(eig(gram_A));
    max_eigen_B = max(eig(gram_B));

    alpha_u = 1/max_eigen_A;
    alpha_v = 1/max_eigen_B;

    u = ones(m, 1) / m;
    v = ones(n, 1) / n;


    % Gradient Descent with Projection
    for iter = 1:max_iter
        tic;

        grad_u = A' * (A * u - B * v);
        grad_v = -B' * (A * u - B * v);
        
        % Gradient descent updates
        u_new = u - alpha_u * grad_u;
        v_new = v - alpha_v * grad_v;
        
        % Project u and v/
        u_new = simplexProjectionD(u_new,d);
        v_new = simplexProjectionD(v_new,d);
        
    
        current_objective = (1/2 * square_pos(norm(A * u - B * v, 2)));
        objective_values = [objective_values; current_objective]; 
        
        elapsed_time = toc;
        cumulate_time =cumulate_time + elapsed_time;
        time_measure = [time_measure; cumulate_time, current_objective];

        %Stopping criteria
        if (norm(u_new-u,2)) <  1e-6 && (norm(v_new-v,2)) <  1e-6
            break
        end
        
        % Update u and v
        u = u_new;
        v = v_new;
    end
    
end