function proj = simplexProjection(z)

    z_sorted = sort(z, 'descend');
    
    sum_z = cumsum(z_sorted);
    
    t = find(z_sorted - (sum_z - 1) ./ (1:length(z_sorted))' > 0, 1, 'last');
    
    theta = (sum_z(t) - 1) / t;
    
    proj = max(z - theta, 0);
end