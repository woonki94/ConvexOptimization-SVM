function plotSvm(u,v, train_data, test_data)
     
     A = train_data.A;
     B = train_data.B;
     X_test = test_data.X_test;
     true_labels = test_data.true_labels;     

    closest_point_A = A * u;
    closest_point_B = B * v;


    fprintf('Au: [%f, %f]\n', closest_point_A(1), closest_point_A(2));
    fprintf('Bv: [%f, %f]\n', closest_point_B(1), closest_point_B(2));
    fprintf('Objective value: %f\n', 1/2 * square_pos(norm(closest_point_A - closest_point_B, 2)));
    
    % Calculate normal vector and classifier boundary
    normal_vector = closest_point_A - closest_point_B;
    classifier_boundary = (closest_point_A + closest_point_B) / 2;
    
    % Visualization for Training Data
    figure;
    axis equal;
    scatter(A(1, :), A(2, :), 'r', 'filled'); hold on;
    scatter(B(1, :), B(2, :), 'b', 'filled');
    scatter(closest_point_A(1), closest_point_A(2), 'ro', 'LineWidth', 2); % Closest point in Class A
    scatter(closest_point_B(1), closest_point_B(2), 'bo', 'LineWidth', 2); % Closest point in Class B
    
    % Line segment between closest points
    axis equal;
    plot([closest_point_A(1), closest_point_B(1)], ...
         [closest_point_A(2), closest_point_B(2)], 'k-', 'LineWidth', 1.5);
    
    % Classifier boundary
    axis equal;
    plot([classifier_boundary(1) - normal_vector(2), classifier_boundary(1) + normal_vector(2)], ...
         [classifier_boundary(2) + normal_vector(1), classifier_boundary(2) - normal_vector(1)], ...
         'k--', 'LineWidth',2);

    legend('A', 'B', 'Au', 'Bv', 'Line Segment', 'Classifier');
    title('Training Data');
    hold off;
    
    figure;
    axis equal;
    scatter(X_test(1, true_labels == 1), X_test(2, true_labels == 1), 'r', 'filled'); hold on;
    scatter(X_test(1, true_labels == -1), X_test(2, true_labels == -1), 'b', 'filled');
    plot([classifier_boundary(1) - normal_vector(2), classifier_boundary(1) + normal_vector(2)], ...
         [classifier_boundary(2) + normal_vector(1), classifier_boundary(2) - normal_vector(1)], ...
         'k--', 'LineWidth', 1.5);
    legend('A', 'B', 'Classifier');
    title('Testing Data');
    hold off;
    
    % Predict labels on the test data
    predicted_labels = sign((X_test' - classifier_boundary') * (normal_vector));
    
    % Calculate classification error
    classification_error = sum(predicted_labels ~= true_labels') / length(true_labels);
    fprintf('Classification Error on Testing Data: %.3f%%\n', classification_error * 100);
    

end