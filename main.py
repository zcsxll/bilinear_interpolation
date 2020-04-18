import torch
import numpy as np

def bilinear_interpolation(img_in, scale_factor=2):
    shape = img_in.shape
    assert(len(shape) == 4)
    assert(shape[0] == 1)
    img_in = img_in[0]
    height = shape[2] * scale_factor
    width = shape[3] * scale_factor

    img_out = np.zeros((shape[1], height, width)).astype(np.float32)

    for dy in range(height):
        for dx in range(width):
            x = float(dx / (width - 1)) * (shape[3] - 1)
            y = float(dy / (height - 1)) * (shape[2] - 1)
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = int(np.ceil(x))
            y1 = int(np.ceil(y))
            # print(src_x0, src_y0, src_x1, src_y1, src_x, src_y)
            # distance0 = np.sqrt((src_x - src_x0) ** 2 + (src_y - src_y0) ** 2)
            # distance1 = np.sqrt((src_x - src_x1) ** 2 + (src_y - src_y1) ** 2)
            # distance2 = np.sqrt((src_x0 - src_x1) ** 2 + (src_y0 - src_y1) ** 2)
            # img_out[:, y, x] = distance0 * img_in[:, src_y1, src_x1] + distance1 * img_in[:, src_y0, src_x0]
            if x0 == x1:
                temp0 = img_in[:, y0, x0]
                temp1 = img_in[:, y1, x0]
            else:
                temp0 = (x - x0) * img_in[:, y0, x1] + (x1 - x) * img_in[:, y0, x0]
                temp1 = (x - x0) * img_in[:, y1, x1] + (x1 - x) * img_in[:, y1, x0]
            if y0 == y1:
                img_out[:, dy, dx] = temp0
            else:
                img_out[:, dy, dx] = (y - y0) * temp1 + (y1 - y) * temp0
        # break
    return img_out

if __name__ == "__main__":
    upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    img_in = np.arange(1, 5).reshape(1, 1, 2, 2).astype(np.float32)
    print(img_in)
    img_out = upsample(torch.from_numpy(img_in))
    print(img_out)
    img_out2 = bilinear_interpolation(img_in, scale_factor=2)
    print(img_out2)


    img_in = np.arange(1, 10).reshape(1, 1, 3, 3).astype(np.float32)
    print(img_in)
    img_out = upsample(torch.from_numpy(img_in))
    print(img_out)
    img_out2 = bilinear_interpolation(img_in, scale_factor=2)
    print(img_out2)