function [u,v,objective_values,time_measure] = projectedNesterov(train_data,d, max_iter)
    %Initialization
    A= train_data.A;
    B= train_data.B;
    objective_values = [];
    time_measure = [];
    cumulate_time = 0;

    m = size(A, 2);
    n = size(B, 2);

    gram_A = A' * A;
    gram_B = B' * B;

    max_eigen_A = max(eig(gram_A));
    max_eigen_B = max(eig(gram_B));
    %learning rate for u,v
    alpha_u = 1/max_eigen_A;
    alpha_v = 1/max_eigen_B;

    u = ones(m, 1) / m;
    v = ones(n, 1) / n;
    u_prev = u;
    v_prev = v;

    t_u = 1;
    t_v = 1;
    a_r = 0;

    for iter = 1:max_iter
        tic;
        %compute a_r
        a_new = (1 + sqrt(1 + 4 * a_r^2)) / 2;
        %compute t_r
        t_r = (a_r - 1) / a_new;
        a_r = a_new; 

        %compute extrapolated point y
        y_u = u + t_r * (u - u_prev);
        y_v = v + t_r * (v - v_prev);
        %compute gradient
        grad_u = A' * (A * y_u - B * y_v);
        grad_v = -B' * (A * y_u - B * y_v);
        %update u
        u_tmp = y_u - alpha_u * grad_u;
        v_tmp = y_v - alpha_v * grad_v;
    
        u_tmp = simplexProjectionD(u_tmp,d);
        v_tmp = simplexProjectionD(v_tmp,d);
        %update t
        t_u_next = (1 + sqrt(1 + 4 * t_u^2)) / 2;
        t_v_next = (1 + sqrt(1 + 4 * t_v^2)) / 2;
        
        current_objective = (1/2 * norm(A * u - B * v, 2)^2); % Objective function 1/2 ||Au - Bv||_2^2
        objective_values = [objective_values; current_objective];

        elapsed_time = toc;
        cumulate_time =cumulate_time + elapsed_time;
        time_measure = [time_measure; cumulate_time, current_objective];
        
        % Update variables
        u_prev = u;
        v_prev = v;
        u = u_tmp;
        v = v_tmp;

        % Update t
        t_u = t_u_next;
        t_v = t_v_next;


    end
end