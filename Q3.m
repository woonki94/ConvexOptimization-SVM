% Load data

train_sep = load('separable_case/train_separable.mat', 'A', 'B');
test_sep = load('separable_case/test_separable.mat', 'X_test', 'true_labels');

train_ov = load('overlap_case/train_overlap.mat', 'A', 'B');
test_ov = load('overlap_case/test_overlap.mat', 'X_test', 'true_labels');

max_iter = 1000;
d_values = 0.01:0.01:1;
%Set d
d_sep =1;
d_ov = computeOptimalD(train_ov, test_ov,d_values,'admm' );
rho_sep = computeOptimalRho(train_sep,d_sep,max_iter );
display(rho_sep);
rho_ov = computeOptimalRho(train_ov,d_ov,max_iter );
display(rho_ov);
[u_sep,v_sep,obj_sep] = admm(train_sep,d_sep,max_iter,rho_sep);
[u_ov, v_ov,obj_ov] = admm(train_ov,d_ov,max_iter,rho_ov);

plotSvm(u_sep,v_sep,train_sep, test_sep)
plotSvm(u_ov,v_ov, train_ov, test_ov);

