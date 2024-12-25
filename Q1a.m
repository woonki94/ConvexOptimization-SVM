whos -file separable_case/train_separable.mat %A, B
whos -file separable_case/test_separable.mat %X_test, true_labels

train_data =load('separable_case/train_separable.mat', 'A', 'B');
test_data = load('separable_case/test_separable.mat', 'X_test', 'true_labels');

A = train_data.A;
B = train_data.B;

m = size(A, 2);
n = size(B, 2);

u = ones(m, 1) / m;
v = ones(n, 1) / n;

objective_values= [];
%solving optimization problem using cvx.
cvx_begin
    variables u(m) v(n)
    minimize(1/2 * square_pos(norm(A * u - B * v, 2))) 
    subject to
        sum(u) == 1;
        sum(v) == 1;
        u >= 0;
        v >= 0;
cvx_end


plotSvm(u,v, train_data, test_data)


