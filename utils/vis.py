import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import os
def visualize_and_save_3d_matrix(data, output_path='3d_matrix.png', grid_resolution=None, scene_box=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Assuming data is a 3D numpy array
    x, y, z = np.indices(np.array(data.shape) + 1)
    x = x[:-1, :-1, :-1].flatten()  # reduce each dimension to align with data
    y = y[:-1, :-1, :-1].flatten()
    z = z[:-1, :-1, :-1].flatten()
    intensity = data.flatten()  # Use data value for color mapping
    
    # Normalize color map
    norm = plt.Normalize(0, intensity.max())
    # colors = plt.cm.viridis(norm(intensity))
    colors = plt.cm.jet(norm(intensity))
    alphas = np.clip(norm(intensity), 0, 1)

    # cmmapable = cm.ScalarMappable(norm, cm.get_cmap('viridis'))
    cmmapable = cm.ScalarMappable(norm, cm.get_cmap('jet'))
    cmmapable.set_array(np.linspace(intensity.min(), intensity.max(), 3))
    
    # Setting the alpha based on the value (example: normalized value)
    # alphas = (colors[:, 0] + colors[:, 1] + colors[:, 2]) / 3  # Simple average to compute alpha
    colors[:, 3] = alphas  # Set the alpha channel
    
    # Scatter plot
    sc = ax.scatter(x, y, z, c=colors, marker='o')
    if scene_box is not None:
        ax.set_xlim(scene_box[0][0], scene_box[1][0])
        ax.set_ylim(scene_box[0][1], scene_box[1][1])
        ax.set_zlim(scene_box[0][2], scene_box[1][2])
    
    # Customizations
    if grid_resolution is not None:
        ax.set_box_aspect([grid_resolution[0],grid_resolution[1],grid_resolution[2]])
    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.set_zlabel('Z Coordinates')
    # plt.colorbar(sc, ax=ax, shrink=0.6, aspect=20)  # Optional: Colorbar
    # plt.colorbar(cmmapable)
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()  # Close the plot window 
    # print(f"Plot saved to {output_path}")

def visualize_matrix(mat, out_dir, file_name='dpp_S.png'):
    plt.matshow(mat)
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, file_name))
    plt.close()