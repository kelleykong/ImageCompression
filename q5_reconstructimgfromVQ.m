function recI = q5_reconstructimgfromVQ(prototypes, tilesize, tileidx, num_x_tiles, num_y_tiles)
% Reconstructs an image starting from the VQ model.
%
% INPUT:
%  prototypes: [n x K] matrix, containing the n-dimensional centroids of the K clusters.
%  tilesize: [1 x 1] scalar, indicating the size of the tiles.
%  tileidx: [m x 1] vector, containing the labels that the Kmeans algorithm assigned to the data.
%           tileidx(i) is an element of {1 ... K} and it indicates the
%           cluster/prototype ID associated to the i-th example/tile;
%           not that tileidx stores the tiles in raster order (see comments in file q5_splitimgintiles.m)
%  num_x_tiles: [1 x 1] scalar value, indicating the number of tiles along the x axis.
%  num_y_tiles: [1 x 1] scalar value, indicating the number of tiles along the y axis.
% 
% OUTPUT:
%  recI: [r x c] matrix, corresponding to the reconstructed gray-scale image

recI = zeros(num_y_tiles * tilesize, num_x_tiles * tilesize);

for j = 1:num_x_tiles
    for i = 1:num_y_tiles
        recI((i-1)*tilesize+1:i*tilesize, (j-1)*tilesize+1:j*tilesize) = reshape(prototypes(:,tileidx((j-1)*num_y_tiles+i,1)), tilesize,tilesize);
    end
end

end