import numpy as np
from scipy.signal import convolve, convolve2d


def dot_productA(grid_width, grid_height, layer_1_output_size, transposed_grid, weights):
    layer_1_outputs = np.zeros((grid_width, grid_height, layer_1_output_size))

    for width in range(grid_width):
        for height in range(grid_height):
            for layer_1_depth in range(layer_1_output_size):
                dot_product = np.dot(transposed_grid[width][height], weights[layer_1_depth])
                layer_1_outputs[width][height][layer_1_depth] = dot_product
    
    return layer_1_outputs


def dot_productB(grid_width, grid_height, layer_1_output_size, transposed_grid, weights):
    # the transposed grid is in the shape (width, height, depth)
    layer_1_outputs = np.zeros((grid_width, grid_height, layer_1_output_size))
    for width in range(grid_width):
        dot_product = np.dot(transposed_grid[width], weights.T)
        layer_1_outputs[width] = dot_product

    return layer_1_outputs


def dot_productC(grid_width, grid_height, layer_1_output_size, transposed_grid, weights):
    # Transposed grid shape: (grid_width, grid_height, layer_1_output_size)
    # Reshape transposed_grid to (grid_width * grid_height, layer_1_output_size)
    reshaped_grid = transposed_grid.reshape(grid_width * grid_height, weights[0].size)

    dot_product = np.dot(reshaped_grid, weights.T) # dot product of reshaped_grid and weights.T

    # Reshape dot_product back to original shape (grid_width, grid_height, layer_1_output_size)
    layer_1_outputs = dot_product.reshape(grid_width, grid_height, layer_1_output_size)

    return layer_1_outputs


def dot_productD(grid_width, grid_height, layer_1_output_size, transposed_grid, weights):
    # Transposed grid shape: (grid_width, grid_height, layer_1_output_size)
    # Reshape transposed_grid to (grid_width * grid_height, layer_1_output_size)
    reshaped_grid = transposed_grid.reshape(grid_width * grid_height, weights[0].size)

    dot_product = reshaped_grid @ weights.T # dot product of reshaped_grid and weights.T

    # Reshape dot_product back to original shape (grid_width, grid_height, layer_1_output_size)
    layer_1_outputs = dot_product.reshape(grid_width, grid_height, layer_1_output_size)

    return layer_1_outputs

if __name__ == '__main__':
    depth, width, height = 3, 10, 10
    grid = np.random.uniform(-1, 1, size=(depth, width, height))
    transposed = np.transpose(grid, (1, 2, 0))

    l = 10
    weights = np.random.uniform(-1, 1, size=(l, 3))

    updatedA = dot_productA(width, height, l, transposed, weights)
    updatedB = dot_productB(width, height, l, transposed, weights)
    updatedC = dot_productC(width, height, l, transposed, weights)
    updatedD = dot_productD(width, height, l, transposed, weights)

    print(np.round(updatedD - updatedA, 8))



"""
    def update_all_states(self):
        # first we need to apply our kernal operation to each cell and their immidate neighbours in their 2d slice
        convolved_grid = np.zeros(self.grid.shape)
        grid_depth, grid_width, grid_height = self.grid.shape
        for depth in range(grid_depth):
            for width in range(grid_width):
                for height in range(grid_height):
                    dot_product = 0
                    for idx in range(-1, 1):
                        for idy in range(-1, 1):
                            dot_product += self.grid[depth][idx][idy] * self.pre_weights[0][idx+1][idy+1]
                    
                    convolved_grid[depth][width][height] = dot_product

        # next we need to transpose the convolved_grid so that all the convolved items for each 2d cell
        # depthwise are placed together, shape (A, B, C) to shape (B, C, A).
        # or from (depth, width, height) to (width, height, depth)
        transposed_grid = np.zeros((grid_width, grid_height, grid_depth))
        for width in range(grid_width):
            for height in range(grid_height):
                for depth in range(grid_depth):
                    transposed_grid[width][height][depth] = convolved_grid[depth][width][height]
        
        # now we have our neural inputs grouped together, we can apply the first layer operations to it
        layer_1_outputs = np.zeros((grid_width, grid_height, self.layer_1_output_size))
        for width in range(grid_width):
            for height in range(grid_height):
                for layer_1_depth in range(self.layer_1_output_size):
                    dot_product = 0

                    for layer_0_depth in range(grid_depth):
                        dot_product += transposed_grid[width][height][layer_0_depth] * self.net_weights_layer_1[layer_1_depth][layer_0_depth]

                    layer_1_outputs[width][height][layer_1_depth] = dot_product
        

        layer_2_outputs = np.zeros((grid_width, grid_height, self.layer_2_output_size))
        for width in range(grid_width):
            for height in range(grid_height):
                for layer_2_depth in range(self.layer_2_output_size):
                    dot_product = 0

                    for layer_1_depth in range(self.layer_1_output_size):
                        dot_product += transposed_grid[width][height][layer_1_depth] * self.net_weights_layer_2[layer_2_depth][layer_1_depth]

                    layer_2_outputs[width][height][layer_2_depth] = dot_product
        
        # finally we transpose the (B, C, A) matrix back into an (A, B, C) matrix
        detransposed = np.zeros(self.grid.shape)
        for depth in range(grid_depth):
            for width in range(grid_width):
                for height in range(grid_height):
                    detransposed[depth][width][height] = layer_2_outputs[width][height][depth]
        
        self.grid = detransposed"""