import imageio


def gif(image_path_list, output_path, fps=8):
    data = [imageio.read(x) for x in image_path_list]
    imageio.mimsave(output_path, data, fps=fps)
    return output_path
