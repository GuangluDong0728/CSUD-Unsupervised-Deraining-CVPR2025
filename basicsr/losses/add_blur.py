from basicsr.losses.GaussionSmoothLayer import GaussionSmoothLayer

BGBlur_kernel = [3, 9, 15]
BlurWeight = [0.01,0.1,1.]

BlurNet = [GaussionSmoothLayer(3, k_size, 25).cuda() for k_size in BGBlur_kernel]

for index, weight in enumerate(BlurWeight):
    out_b1 = BlurNet[index](img_A)