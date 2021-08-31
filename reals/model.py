import torch.nn as nn
import torch.nn.functional as F
import torch


class REALS_model(nn.Module):
    def __init__(self, data_shape, tau, device, k=1):
        """
        Model of REALS. It consists of geometric transformation and linear layer.
        In case of euclidean transformation, you can control clamping values for better optimization.
        """
        super().__init__()
        self.w, self.h, self.t = data_shape
        self.inp = self.w * self.h
        self.ln = nn.Linear(self.inp, k, bias=False)
        self.device = device
        self.tau = tau

        if self.tau == 'affine':
            theta = torch.tensor([[1, 0, 0],
                                  [0, 1, 0]]
                                 , dtype=torch.float, device=device)
            self.theta = theta.unsqueeze(0).repeat(self.t, 1, 1).detach().requires_grad_(True)
        elif self.tau == 'euclidean':
            self.theta_ro = torch.zeros((self.t, 1), dtype=torch.float, device=self.device).requires_grad_(True)
            self.theta_tr = torch.zeros((self.t, 2), dtype=torch.float, device=self.device).requires_grad_(True)
            self.theta = [self.theta_ro, self.theta_tr]

    def forward(self, x):
        if self.tau == 'affine':
            theta = self.theta
        elif self.tau == 'euclidean':
            theta = torch.zeros((self.t, 2, 3), dtype=torch.float, device=self.device)
            theta[:, 0, 0] = torch.cos(self.theta_ro[:, 0])
            theta[:, 1, 1] = torch.cos(self.theta_ro[:, 0])
            theta[:, 0, 1] = -torch.sin(self.theta_ro[:, 0])
            theta[:, 1, 0] = torch.sin(self.theta_ro[:, 0])
            theta[:, 0, 2] = self.theta_tr[:, 0]
            theta[:, 1, 2] = self.theta_tr[:, 1]
        else:
            exit()
        x_reshape = x.view(self.t, 1, self.w, self.h)  # 60x1x512x512
        grid = F.affine_grid(theta, x_reshape.size())
        x_reg = F.grid_sample(x_reshape, grid).view(self.t, self.inp)  # 60x(512x512)
        L = self.ln(x_reg) @ self.ln.weight  # 60x(512x512)
        return x_reg, L

    def clamp_theta(self):
        if self.tau == 'affine':
            self.theta[:, 0, 0].data.clamp_(max=1.0)
            self.theta[:, 1, 1].data.clamp_(max=1.0)
        if self.tau == 'euclidean':
            ''' # large displacement data setting
            '''
            # self.theta_ro.data.clamp_(min=-1.0, max=1.0)
            # self.theta_tr.data.clamp_(min=-0.4, max=0.4)
            ''' # original setting
            '''
            self.theta_ro.data.clamp_(min=-4e-1, max=4e-1)
            self.theta_tr.data.clamp_(min=-25e-2, max=25e-2)


class REALS_model_3d(nn.Module):
    def __init__(self, data_shape, tau, device, k=1):
        """
        3D Model of REALS. It consists of 3D geometric transformation and linear layer.
        affine transformation is available. Controlling clamping value is important since
        optimization may lead to minification of images.
        """
        super().__init__()
        self.w, self.h, self.d, self.t = data_shape
        self.inp = self.w * self.h * self.d
        self.ln = nn.Linear(self.inp, k, bias=False)
        self.device = device
        self.tau = tau

        if self.tau == 'affine':
            theta = torch.tensor([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0]]
                                 , dtype=torch.float, device=device)
            self.theta = theta.unsqueeze(0).repeat(self.t, 1, 1).detach().requires_grad_(True)
        else:
            exit()

    def forward(self, x, r_index):
        if self.tau == 'affine':
            theta = self.theta[r_index, ...]
        else:
            exit()

        x_reshape = x.view(len(r_index), 1, self.w, self.h, self.d)
        grid = F.affine_grid(theta, x_reshape.size())
        x_reg = F.grid_sample(x_reshape, grid).view(len(r_index), self.inp)  # t, whd
        L = self.ln(x_reg) @ self.ln.weight  # t, whd

        return x_reg, L

    def clamp_theta(self):
        '''
        the images can be minified upto 1.02.
        '''
        self.theta[:, 0, 0].data.clamp_(max=1.02)
        self.theta[:, 1, 1].data.clamp_(max=1.02)
        self.theta[:, 2, 2].data.clamp_(max=1.02)


class REALS_model_minibatch(nn.Module):
    """
    Model of REALS mini-batch version. It consists of geometric transformation and linear layer.
    """
    def __init__(self, data_shape, tau, device, k=1):
        super().__init__()
        self.w, self.h, self.t = data_shape
        self.inp = self.w * self.h
        self.ln = nn.Linear(self.inp, k, bias=False)
        self.device = device
        self.tau = tau

        if self.tau == 'affine':
            theta = torch.tensor([[1, 0, 0],
                                  [0, 1, 0]]
                                 , dtype=torch.float, device=device)
            self.theta = theta.unsqueeze(0).repeat(self.t, 1, 1).detach().requires_grad_(True)
        elif self.tau == 'euclidean':
            self.theta_ro = torch.zeros((self.t, 1), dtype=torch.float, device=self.device).requires_grad_(True)
            self.theta_tr = torch.zeros((self.t, 2), dtype=torch.float, device=self.device).requires_grad_(True)
            self.theta = [self.theta_ro, self.theta_tr]

    def forward(self, x, r_index):
        if self.tau == 'affine':
            theta = self.theta[r_index]
        elif self.tau == 'euclidean':
            theta = torch.zeros((len(r_index), 2, 3), dtype=torch.float, device=self.device)
            theta[:, 0, 0] = torch.cos(self.theta_ro[r_index, 0])
            theta[:, 1, 1] = torch.cos(self.theta_ro[r_index, 0])
            theta[:, 0, 1] = -torch.sin(self.theta_ro[r_index, 0])
            theta[:, 1, 0] = torch.sin(self.theta_ro[r_index, 0])
            theta[:, 0, 2] = self.theta_tr[r_index, 0]
            theta[:, 1, 2] = self.theta_tr[r_index, 1]

        x_reshape = x.view(len(r_index), 1, self.w, self.h)  # 60x1x512x512
        grid = F.affine_grid(theta, x_reshape.size())
        x_reg = F.grid_sample(x_reshape, grid).view(len(r_index), self.inp)  # 60x(512x512)
        L = self.ln(x_reg) @ self.ln.weight  # 60x(512x512)

        return x_reg, L

    def clamp_theta(self):
        if self.tau == 'euclidean':
            self.theta_ro.data.clamp_(min=-6e-1, max=6e-1)
            self.theta_tr.data.clamp_(min=-25e-2, max=25e-2)


class REALS_model_multi(nn.Module):
    def __init__(self, data_shape, tau, device, k=1):
        """
        Model of multi-resolution REALS. It consists of geometric transformation and linear layer.
        Multi-resolution consists of x1, x2, x4, x8, which is 4-levels.
        """
        super().__init__()
        self.w, self.h, self.t = data_shape
        self.inp = self.w * self.h
        self.ln = nn.Linear(self.inp, k, bias=False)
        self.ln_x2 = nn.Linear(self.inp // 4, k, bias=False)
        self.ln_x4 = nn.Linear(self.inp // 16, k, bias=False)
        self.ln_x8 = nn.Linear(self.inp // 64, k, bias=False)
        self.device = device
        self.tau = tau
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        if self.tau == 'affine':
            theta = torch.tensor([[1, 0, 0],
                                  [0, 1, 0]]
                                 , dtype=torch.float, device=device)
            self.theta = theta.unsqueeze(0).repeat(self.t, 1, 1).detach().requires_grad_(True)
        elif self.tau == 'euclidean':
            self.theta = torch.zeros((self.t, 3),
                                     dtype=torch.float, device=self.device).requires_grad_(True)

    def forward(self, x):
        if self.tau == 'affine':
            theta = self.theta
        elif self.tau == 'euclidean':
            theta = torch.zeros((self.t, 2, 3), dtype=torch.float, device=self.device)
            theta[:, 0, 0] = torch.cos(self.theta[:, 0])
            theta[:, 1, 1] = torch.cos(self.theta[:, 0])
            theta[:, 0, 1] = -torch.sin(self.theta[:, 0])
            theta[:, 1, 0] = torch.sin(self.theta[:, 0])
            theta[:, 0, 2] = self.theta[:, 1]
            theta[:, 1, 2] = self.theta[:, 2]

        x_reshape = x.view(self.t, 1, self.w, self.h)  # 60x1x512x512
        x_reshape_x2 = self.avgpool(x_reshape)
        x_reshape_x4 = self.avgpool(x_reshape_x2)
        x_reshape_x8 = self.avgpool(x_reshape_x4)
        # x1
        grid = F.affine_grid(theta, x_reshape.size())
        x_reg = F.grid_sample(x_reshape, grid).view(self.t, self.inp)  # 60x(512x512)
        L = self.ln(x_reg) @ self.ln.weight  # 60x(512x512)
        # x2
        grid = F.affine_grid(theta, x_reshape_x2.size())
        x_reg_x2 = F.grid_sample(x_reshape_x2, grid).view(self.t, self.inp // 4)  # 60x(256x256)
        L_x2 = self.ln_x2(x_reg_x2) @ self.ln_x2.weight  # 60x(256x256)
        # x4
        grid = F.affine_grid(theta, x_reshape_x4.size())
        x_reg_x4 = F.grid_sample(x_reshape_x4, grid).view(self.t, self.inp // 16)  # 60x(128x128)
        L_x4 = self.ln_x4(x_reg_x4) @ self.ln_x4.weight  # 60x(128x128)
        # x8
        grid = F.affine_grid(theta, x_reshape_x8.size())
        x_reg_x8 = F.grid_sample(x_reshape_x8, grid).view(self.t, self.inp // 64)  # 60x(64x64)
        L_x8 = self.ln_x8(x_reg_x8) @ self.ln_x8.weight  # 60x(64x64)

        return x_reg, L, x_reg_x2, L_x2, x_reg_x4, L_x4, x_reg_x8, L_x8

    def clamp_theta(self):
        if self.tau == 'euclidean':
            self.theta[:, 0].data.clamp_(min=-0.6, max=0.6)
            self.theta[:, 1:3].data.clamp_(min=-0.25, max=0.25)

