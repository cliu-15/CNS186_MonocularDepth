import torch
import torch.nn as nn
import torch.nn.functional as F

print('our loss')
class MonodepthLoss(nn.modules.Module):
    def __init__(self, n=4, SSIM_w=0.85, disp_gradient_w=1.0, lr_w=1.0):
        super(MonodepthLoss, self).__init__()
        self.SSIM_w = SSIM_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w
        self.n = n

    def scale_pyramid(self, img, num_scales):
        pyramid = [img]
        h, w = img.size(2), img.size(3)
        for i in range(1, num_scales):
            scaled_h = h // (2 ** i)
            scaled_w = w // (2 ** i)
            pyramid.append(nn.functional.interpolate(img,
                               size=[scaled_h, scaled_w], mode='bilinear',
                               align_corners=True))
        return pyramid

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        return (img[:, :, :, :-1] - img[:, :, :, 1:])

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        return (img[:, :, :-1, :] - img[:, :, 1:, :])

    def generate_img(self, img, disp):
        # - disp to generate left image
        # + disp to generate right image
        batch_size, _, h, w = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, w).repeat(batch_size, h, 1).type_as(img)
        y_base = torch.linspace(0, 1, h).repeat(batch_size, w, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        flow_field = torch.stack((x_base + disp[:, 0, :, :], y_base), dim=3)

        return (F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros'))

    def compute_SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x.pow(2)
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y.pow(2)
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def disp_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d)  for d in disp]
        disp_gradients_y = [self.gradient_y(d)  for d in disp]

        image_gradients_x = [self.gradient_x(i)  for i in pyramid]
        image_gradients_y = [self.gradient_y(i)  for i in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True))  for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True))  for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i]  for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i]  for i in range(self.n)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])  for i in range(self.n)]

    def forward(self, input, target):
        """
        Args:
            input [disp1, disp2, disp3, disp4]
            target [left, right]

        Return:
            (float): The loss
        """
        left, right = target
        left_pyramid = self.scale_pyramid(left, self.n)
        right_pyramid = self.scale_pyramid(right, self.n)

        # Prepare disparities
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in input]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in input]

        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est
        # Generate images
        left_est = [self.generate_img(right_pyramid[i], -disp_left_est[i]) for i in range(self.n)]
        right_est = [self.generate_img(left_pyramid[i], disp_right_est[i]) for i in range(self.n)]
        self.left_est = left_est
        self.right_est = right_est

        # L-R Consistency
        right_left_disp = [self.generate_img(disp_right_est[i], -disp_left_est[i]) for i in range(self.n)]
        left_right_disp = [self.generate_img(disp_left_est[i], disp_right_est[i]) for i in range(self.n)]

        # Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_left_est, left_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_right_est, right_pyramid)

        # L1
        l1_left = [torch.mean(torch.abs(left_est[i] - left_pyramid[i]))  for i in range(self.n)]
        l1_right = [torch.mean(torch.abs(right_est[i] - right_pyramid[i]))  for i in range(self.n)]


        # SSIM
        ssim_left = [torch.mean(self.compute_SSIM(left_est[i], left_pyramid[i]))  for i in range(self.n)]
        ssim_right = [torch.mean(self.compute_SSIM(right_est[i], right_pyramid[i]))  for i in range(self.n)]

        image_loss = sum(
                        [self.SSIM_w * ssim_left[i] + (1 - self.SSIM_w) * l1_left[i]
                        + self.SSIM_w * ssim_right[i] + (1 - self.SSIM_w) * l1_right[i]
                        for i in range(self.n)]
        )

        # L-R Consistency
        lr_loss = sum(
                        [torch.mean(torch.abs(right_left_disp[i] - disp_left_est[i]))
                        + torch.mean(torch.abs(left_right_disp[i] - disp_right_est[i]))
                        for i in range(self.n)]
        )


        # Disparities smoothness
        disp_gradient_loss = sum(
                            [torch.mean(torch.abs(disp_left_smoothness[i])) / 2 ** i
                            + torch.mean(torch.abs(disp_right_smoothness[i])) / 2 ** i
                            for i in range(self.n)]
        )


        loss = image_loss + self.disp_gradient_w * disp_gradient_loss\
               + self.lr_w * lr_loss
        self.image_loss = image_loss
        self.disp_gradient_loss = disp_gradient_loss
        self.lr_loss = lr_loss
        return loss