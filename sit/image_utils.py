import imageio

all_images = [f"test_out/ddim_intermediate_projection_step_{981 - i*20}.png" for i in range(49)]

images = []
for image_path in all_images:
    images.append(imageio.imread(image_path))

imageio.mimsave('test_out/ddim_intermediate.gif', images)