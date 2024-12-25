train_sep = load('separable_case/train_separable.mat', 'A', 'B');
test_sep = load('separable_case/test_separable.mat', 'X_test', 'true_labels');

train_ov = load('overlap_case/train_overlap.mat', 'A', 'B');
test_ov = load('overlap_case/test_overlap.mat', 'X_test', 'true_labels');

max_iter = 1000;
rho = 1000; %penalty param for admm
d_sep =1;
d_ov = 0.04; % all algorithms had least error when d =0.001
rho_sep = computeOptimalRho(train_sep,d_sep,max_iter );
rho_ov = computeOptimalRho(train_ov,d_ov,max_iter );


[u_sep,v_sep,obj_sep_pg,time_sep_pg] = projectedGradient(train_sep,d_sep,max_iter);
[u_ov, v_ov,obj_ov_pg,time_ov_pg] = projectedGradient(train_ov,d_ov,max_iter);


[u_sep,v_sep,obj_sep_ns,time_sep_ns] = projectedNesterov(train_sep,d_sep,max_iter);
[u_ov, v_ov,obj_ov_ns,time_ov_ns] = projectedNesterov(train_ov,d_ov,max_iter);


[u_sep,v_sep,obj_sep_admm,time_sep_admm] = admm(train_sep,d_sep,max_iter,rho_sep);
[u_ov, v_ov,obj_ov_admm,time_ov_admm] = admm(train_ov,d_ov,max_iter,rho_ov);


figure;
hold on;
plot(obj_sep_pg,'-o', 'LineWidth', 1, 'MarkerSize', 3, 'DisplayName', 'PG Separable');
plot(obj_sep_ns,'-o', 'LineWidth', 1, 'MarkerSize', 3, 'DisplayName', 'Nesterov Separable');
plot(obj_sep_admm, '-o', 'LineWidth', 1, 'MarkerSize', 3, 'DisplayName', 'ADMM Separable');
title('Objective Function Values Over Iterations', 'FontSize', 14);
xlabel('Iteration', 'FontSize', 12);
ylabel('Objective Value', 'FontSize', 12);
grid on;
legend;
hold off;

figure;
hold on;
plot(obj_ov_pg, '-o', 'LineWidth', 1, 'MarkerSize', 3,'DisplayName', 'PG Overlapping');
plot(obj_ov_ns,'-o', 'LineWidth', 1, 'MarkerSize', 3, 'DisplayName', 'Nesterov Overlapping');
plot(obj_ov_admm, '-o', 'LineWidth', 1, 'MarkerSize', 3, 'DisplayName', 'ADMM Overlapping');
title('Objective Function Values Over Iterations', 'FontSize', 14);
xlabel('Iteration', 'FontSize', 12);
ylabel('Objective Value', 'FontSize', 12);
grid on;
legend;
hold off;


figure;
hold on;
plot(time_sep_pg(:,1),time_sep_pg(:,2),'-o', 'LineWidth', 1, 'MarkerSize', 3, 'DisplayName', 'PG Separable');
plot(time_sep_ns(:,1),time_sep_ns(:,2),'-o', 'LineWidth', 1, 'MarkerSize', 3, 'DisplayName', 'Nesterov Separable');
plot(time_sep_admm(:,1),time_sep_admm(:,2), '-o', 'LineWidth', 1, 'MarkerSize', 3, 'DisplayName', 'ADMM Separable');
title('Objective Function Values Over Time', 'FontSize', 14);
xlabel('Time', 'FontSize', 12);
ylabel('Objective Value', 'FontSize', 12);
grid on;
legend;
hold off;


figure;
hold on;
plot(time_ov_pg(:,1),time_ov_pg(:,2),'-o', 'LineWidth', 1, 'MarkerSize', 3, 'DisplayName', 'PG Overlapping');
plot(time_ov_ns(:,1),time_ov_ns(:,2),'-o', 'LineWidth', 1, 'MarkerSize', 3, 'DisplayName', 'Nesterov Overlapping');
plot(time_ov_admm(:,1),time_ov_admm(:,2), '-o', 'LineWidth', 1, 'MarkerSize', 3, 'DisplayName', 'ADMM Overlapping');
title('Objective Function Values Over Time', 'FontSize', 14);
xlabel('Time', 'FontSize', 12);
ylabel('Objective Value', 'FontSize', 12);
grid on;
legend;
hold off;
