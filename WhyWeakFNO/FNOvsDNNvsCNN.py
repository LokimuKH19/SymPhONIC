import torch
import torch.nn as nn
import torch.fft
import numpy as np


# -----------------------------
# Select the sampling points
# -----------------------------
def generate_data(num_samples=100, grid_size=64):
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y, indexing='ij')

    data_u = []
    data_v = []
    for _ in range(num_samples):
        u0 = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.random.rand()
        v0 = np.cos(np.pi * X) * np.cos(np.pi * Y) * np.random.rand()
        data_u.append(u0)
        data_v.append(v0)
    data_u = torch.tensor(np.array(data_u), dtype=torch.float32)
    data_v = torch.tensor(np.array(data_u), dtype=torch.float32)
    return data_u.unsqueeze(1), data_v.unsqueeze(1)  # [N,1,H,W]


train_u, train_v = generate_data(50)
test_u, test_v = generate_data(10)

# -----------------------------
# MLP
# -----------------------------
class MLP2d(nn.Module):
    def __init__(self, grid_size=64):
        super().__init__()
        self.H = grid_size
        self.W = grid_size
        self.input_dim = 2 * grid_size * grid_size
        self.output_dim = 2 * grid_size * grid_size
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.output_dim)
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)           # [B, 2*H*W]
        x = self.net(x)             # [B, 2*H*W]
        x = x.view(B, 2, self.H, self.W)
        return x


# -----------------------------
# FNO
# -----------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, modes, 2)
        )

    def compl_mul2d(self, input, weights):
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bixy, ioxy -> boxy", input, cweights)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes, :self.modes] = self.compl_mul2d(
            x_ft[:, :, :self.modes, :self.modes], self.weights
        )
        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return x


class FNO2d(nn.Module):
    def __init__(self, modes=12, width=32):
        super().__init__()
        self.modes = modes
        self.width = width
        self.fc0 = nn.Linear(2, width)

        self.conv0 = SpectralConv2d(width, width, modes)
        self.conv1 = SpectralConv2d(width, width, modes)
        self.conv2 = SpectralConv2d(width, width, modes)
        self.conv3 = SpectralConv2d(width, width, modes)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x) + self.w0(x)
        x2 = self.conv1(torch.relu(x1)) + self.w1(torch.relu(x1))
        x3 = self.conv2(torch.relu(x2)) + self.w2(torch.relu(x2))
        x4 = self.conv3(torch.relu(x3)) + self.w3(torch.relu(x3))

        x = x4.permute(0, 2, 3, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x


# -----------------------------
# CNN
# -----------------------------
class CNN2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# PDE residual (numerical differential)
# -----------------------------
def burgers_residual(u, v, dx=1/64, dy=1/64, nu=0.01):
    u_x = (u[:,:,2:,:] - u[:,:,:-2,:]) / (2*dx)
    u_y = (u[:,:,:,2:] - u[:,:,:,:-2]) / (2*dy)
    u_xx = (u[:,:,2:,:] - 2*u[:,:,1:-1,:] + u[:,:,:-2,:]) / dx**2
    u_yy = (u[:,:,:,2:] - 2*u[:,:,:,1:-1] + u[:,:,:,:-2]) / dy**2

    v_x = (v[:,:,2:,:] - v[:,:,:-2,:]) / (2*dx)
    v_y = (v[:,:,:,2:] - v[:,:,:,:-2]) / (2*dy)
    v_xx = (v[:,:,2:,:] - 2*v[:,:,1:-1,:] + v[:,:,:-2,:]) / dx**2
    v_yy = (v[:,:,:,2:] - 2*v[:,:,:,1:-1] + v[:,:,:,:-2]) / dy**2

    # 裁剪 u_y 和 u_yy 对齐 u_x 和 u_xx
    min_h = u_x.shape[2]
    min_w = u_y.shape[3]
    u_x = u_x[:,:, :min_h, :min_w]
    u_xx = u_xx[:,:, :min_h, :min_w]
    u_y = u_y[:,:, :min_h, :min_w]
    u_yy = u_yy[:,:, :min_h, :min_w]

    v_x = v_x[:,:, :min_h, :min_w]
    v_xx = v_xx[:,:, :min_h, :min_w]
    v_y = v_y[:,:, :min_h, :min_w]
    v_yy = v_yy[:,:, :min_h, :min_w]

    r_u = u_x + u_y - nu * (u_xx + u_yy)
    r_v = v_x + v_y - nu * (v_xx + v_yy)

    return r_u, r_v


def loss_fn_physics(pred_u, pred_v):
    r_u, r_v = burgers_residual(pred_u, pred_v)
    phys_loss = (r_u ** 2).mean() + (r_v ** 2).mean()    # use mse loss
    return phys_loss


# -----------------------------
# train
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

fno = FNO2d().to(device)
cnn = CNN2d().to(device)
mlp = MLP2d(grid_size=64).to(device)
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
optimizer_fno = torch.optim.Adam(fno.parameters(), lr=1e-3)
optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=1e-3)

train_data = torch.cat([train_u, train_v], dim=1).to(device)

epochs = 300
for epoch in range(epochs):
    # ---------------- FNO ----------------
    fno.train()
    optimizer_fno.zero_grad()
    pred_fno = fno(train_data)
    loss_fno = loss_fn_physics(pred_fno[:, 0:1], pred_fno[:, 1:2])
    loss_fno.backward()
    optimizer_fno.step()

    # ---------------- CNN ----------------
    cnn.train()
    optimizer_cnn.zero_grad()
    pred_cnn = cnn(train_data)
    loss_cnn = loss_fn_physics(pred_cnn[:, 0:1], pred_cnn[:, 1:2])
    loss_cnn.backward()
    optimizer_cnn.step()

    # ---------------- MLP ----------------
    mlp.train()
    optimizer_mlp.zero_grad()
    pred_mlp = mlp(train_data)
    loss_mlp = loss_fn_physics(pred_mlp[:, 0:1], pred_mlp[:, 1:2])
    loss_mlp.backward()
    optimizer_mlp.step()

    # ---------------- result ----------------
    print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"FNO Loss: {loss_fno.item():.12f} | "
            f"CNN Loss: {loss_cnn.item():.12f} | "
            f"MLP Loss: {loss_mlp.item():.12f}"
        )

