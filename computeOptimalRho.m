function [rho] = computeOptimalRho(train_data,d, max_iter)
    fprintf("\nSearching for rho with Least iteration...\n");

    %searching range
    rho_values = logspace(-3, 3, 10); 
    min_iterations = inf; 
    optimal_rho = rho_values(1); 
    
    for i = 1:length(rho_values)
        rho = rho_values(i); 
        
        [~, ~, obj_sep] = admm(train_data, d, max_iter, rho); 
        
        num_iterations = length(obj_sep); 

        %convert rho with lowest iteration.
        if num_iterations < min_iterations
            min_iterations = num_iterations;
            optimal_rho = rho;
        end
    end
    
    rho = optimal_rho;
end