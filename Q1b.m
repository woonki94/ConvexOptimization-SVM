% Load data
train_data =load('overlap_case/train_overlap.mat', 'A', 'B');
test_data = load('overlap_case/test_overlap.mat', 'X_test', 'true_labels');
A = train_data.A;
B = train_data.B;

m = size(A, 2); % Number of features in A
n = size(B, 2); % Number of features in B
d_values = 0.01:0.01:1;

objective_values = [];

d = computeOptimalD(train_data,test_data,d_values,'cvx');

%solving optimization problem using cvx 
cvx_begin
    variables u(m) v(n)
    minimize(1/2 * square_pos(norm(A * u - B * v, 2)))
    subject to
        sum(u) == 1;
        sum(v) == 1;
        0 <= u <= d;
        0 <= v <= d;
cvx_end


fprintf('Optimal d: %.6f \n', d);
plotSvm(u,v, train_data,test_data)
