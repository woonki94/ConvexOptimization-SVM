function [u,v,objective_values,time_measure] = admm(train_data,d, max_iter,RHO)

    A= train_data.A;
    B= train_data.B;

    objective_values = [];
    time_measure = [];
    cumulate_time = 0;

    m = size(A, 2); 
    n = size(B, 2); 

    %RHO = 1; % ADMM penalty parameter
    objective_values = [];

    x = ones(m+n, 1)/(m+n); % x= [u,v] (200,1)
    z = x; % z= x (200,1)
    z_old = z;
    y = zeros(m+n + 2, 1); %(202,1) 

    one = ones(m, 1); %(100,1)
    zero = zeros(m, 1); %(100,1)

    C = [eye(m+n); [one', zero'; zero', one']]; % (202,200)
    D = [-eye(m+n); zeros(2, m+n)]; % (202,200)
    E = [zeros(m+n, 1); 1; 1]; % (202,1)
    F = [A, -B]; % (202,200)
    
    Q = 2 * F' * F + RHO * (C' * C); 
    Q_inv = inv(Q);

    primal = inf;
    dual = inf;

    for iter = 1:max_iter
        tic;
        % Update x
        x = Q_inv * (-RHO * C' * (D * z - E + y / RHO));
       
        %only use y(1~ m+n)
        %removing m+n ~ m+n+2 part where was constraints of sum of z = 1 
        %so have to project again.
        z = max(0, x + y(1:m+n) / RHO);
        z(1:m) = simplexProjectionD(z(1:m),d);
        z(m+1:m+n) = simplexProjectionD(z(m+1:m+n),d);

        %update y
        y = y + RHO * ( C * x + D * z - E );
        

        % Compute primal problem
        primal = C * x + D * z - E;
        
        % Compute dual problem
        dual = RHO * C' * (D * (z - z_old));
        z_old = z;

        u = z(1:m); % Extract u from z
        v = z(m+1:m+n); % Extract v from z
        
        %storing objective value for plot
        current_objective = (1/2 * norm(A * u - B * v, 2)^2); % Objective function ||Au - Bv||_2^2
        objective_values = [objective_values; current_objective]; 

        %storing time:objective value for plot
        elapsed_time = toc;
        cumulate_time =cumulate_time + elapsed_time;
        time_measure = [time_measure; cumulate_time, current_objective];
        
        %stopping criteria
        if norm(primal, 2) < 1e-6 && norm(dual, 2) < 1e-6
            break;
        end
        
    end
end
