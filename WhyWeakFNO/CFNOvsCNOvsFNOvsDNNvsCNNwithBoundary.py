import torch
import torch.nn as nn
import torch.fft
import numpy as np
from matplotlib import pyplot as plt
import random
from CFNOvsCNOvsFNO_PINN import CNO2d_small, CFNO2d_small, FNO2d_small, CFNOBlock


def seed_everything(seed):
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=16)
    np.set_printoptions(precision=16)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 1000
seed_everything(SEED)
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
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)           # [B, 2*H*W]
        x = self.net(x)             # [B, 2*H*W]
        x = x.view(B, 2, self.H, self.W)
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

    # To Clip
    min_h = u_x.shape[2]
    min_w = u_y.shape[3]
    u = u[:,:, :min_h, :min_w]
    u_x = u_x[:,:, :min_h, :min_w]
    u_xx = u_xx[:,:, :min_h, :min_w]
    u_y = u_y[:,:, :min_h, :min_w]
    u_yy = u_yy[:,:, :min_h, :min_w]
    v = v[:, :, :min_h, :min_w]
    v_x = v_x[:,:, :min_h, :min_w]
    v_xx = v_xx[:,:, :min_h, :min_w]
    v_y = v_y[:,:, :min_h, :min_w]
    v_yy = v_yy[:,:, :min_h, :min_w]

    r_u = u*u_x + v*u_y - nu * (u_xx + u_yy)    # todo change the PDE form
    r_v = u*v_x + v*v_y - nu * (v_xx + v_yy)

    return r_u, r_v


def loss_fn_physics(pred_u, pred_v):
    r_u, r_v = burgers_residual(pred_u, pred_v)
    phys_loss = (r_u ** 2).mean() + (r_v ** 2).mean()    # use mse loss
    return phys_loss


def apply_boundary_conditions(u, v):
    # u, v: [B,1,H,W]
    B, _, H, W = u.shape

    # Top and Bottom no-slip
    u[:, :, 0, :] = 0.0
    u[:, :, -1, :] = 0.0
    v[:, :, 0, :] = 0.0
    v[:, :, -1, :] = 0.0

    # Inlet: Dirichlet
    u[:, :, :, 0] = 10.0    # todo control the inlet flow rate/Re
    v[:, :, :, 0] = 0.0

    # Outlet: Neumann
    u[:, :, :, -1] = u[:, :, :, -2]
    v[:, :, :, -1] = v[:, :, :, -2]

    return u, v


# -----------------------------
# train
# -----------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cnn = CNN2d().to(device)   # 56k, high performance, highest para efficiency, could be added to CFNO
mlp = MLP2d(grid_size=64).to(device)   # 1060k paras, parameter efficiency not good

# -----------------------------
# Add CNO and CFNO
# -----------------------------
fno = FNO2d_small(modes=16, width=16, depth=3, input_features=2, output_features=2).to(device)   # 198k
cno = CNO2d_small(cheb_modes=(8, 8), width=16, depth=3, input_features=2, output_features=2).to(device)   # 50k, also high benefits
cfno = CFNO2d_small(modes=16, cheb_modes=(8, 8), width=16, depth=3, alpha_init=0.5, input_features=2, output_features=2).to(device)   # 249k, And I'm considering add cnn layer to cfno

lr = 5e-4
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=lr)
optimizer_fno = torch.optim.Adam(fno.parameters(), lr=lr)
optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=lr)
optimizer_cno = torch.optim.Adam(cno.parameters(), lr=lr)
optimizer_cfno = torch.optim.Adam(cfno.parameters(), lr=lr)

train_data = torch.cat([train_u, train_v], dim=1).to(device)

epochs = 600
result_fno, result_cnn, result_mlp, result_cno, result_cfno = [], [], [], [], []
for epoch in range(epochs):
    # ---------------- FNO ----------------
    fno.train()
    optimizer_fno.zero_grad()
    pred_fno = fno(train_data)
    pred_fno_u, pred_fno_v = pred_fno[:, 0:1], pred_fno[:, 1:2]
    pred_fno_u, pred_fno_v = apply_boundary_conditions(pred_fno_u, pred_fno_v)
    loss_fno = loss_fn_physics(pred_fno_u, pred_fno_v)
    loss_fno.backward()
    optimizer_fno.step()

    # ---------------- CNN ----------------
    cnn.train()
    optimizer_cnn.zero_grad()
    pred_cnn = cnn(train_data)

    pred_cnn_u, pred_cnn_v = pred_cnn[:, 0:1], pred_cnn[:, 1:2]
    pred_cnn_u, pred_cnn_v = apply_boundary_conditions(pred_cnn_u, pred_cnn_v)
    loss_cnn = loss_fn_physics(pred_cnn_u, pred_cnn_v)
    loss_cnn.backward()
    optimizer_cnn.step()

    # ---------------- MLP ----------------
    mlp.train()
    optimizer_mlp.zero_grad()
    pred_mlp = mlp(train_data)
    pred_mlp_u, pred_mlp_v = pred_mlp[:, 0:1], pred_mlp[:, 1:2]
    pred_mlp_u, pred_mlp_v = apply_boundary_conditions(pred_mlp_u, pred_mlp_v)
    loss_mlp = loss_fn_physics(pred_mlp_u, pred_mlp_v)
    loss_mlp.backward()
    optimizer_mlp.step()

    # ---------------- CNO ----------------
    cno.train()
    optimizer_cno.zero_grad()
    pred_cno = cno(train_data)
    pred_cno_u, pred_cno_v = pred_cno[:, 0:1], pred_cno[:, 1:2]
    pred_cno_u, pred_cno_v = apply_boundary_conditions(pred_cno_u, pred_cno_v)
    loss_cno = loss_fn_physics(pred_cno_u, pred_cno_v)
    loss_cno.backward()
    optimizer_cno.step()

    # ---------------- CFNO ----------------
    cfno.train()
    optimizer_cfno.zero_grad()
    pred_cfno = cfno(train_data)
    pred_cfno_u, pred_cfno_v = pred_cfno[:, 0:1], pred_cfno[:, 1:2]
    pred_cfno_u, pred_cfno_v = apply_boundary_conditions(pred_cfno_u, pred_cfno_v)
    loss_cfno = loss_fn_physics(pred_cfno_u, pred_cfno_v)
    loss_cfno.backward()
    optimizer_cfno.step()


    result_fno.append(loss_fno.cpu().detach().numpy())
    result_cnn.append(loss_cnn.cpu().detach().numpy())
    result_mlp.append(loss_mlp.cpu().detach().numpy())
    result_cno.append(loss_cno.cpu().detach().numpy())
    result_cfno.append(loss_cfno.cpu().detach().numpy())


    # ---------------- result ----------------
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"FNO Loss: {loss_fno.item():.12f} | "
        f"CNO Loss: {loss_cno.item():.12f} | "
        f"CFNO Loss: {loss_cfno.item():.12f} | "
        f"CNN Loss: {loss_cnn.item():.12f} | "
        f"MLP Loss: {loss_mlp.item():.12f}"
    )

def get_alphas(model):
    alphas = []
    for blk in model.blocks:
        if isinstance(blk, CFNOBlock):
            alphas.append(torch.sigmoid(blk.alpha).item())
    return alphas

print(f"CFNO final alpha: {get_alphas(cfno)}")

plt.title(f"Loss Curve Comparison (LearningRate {lr} RandomSeed {SEED} Burgers Flow in 2D Channel)")
plt.plot(result_cnn, label="CNN")
plt.plot(result_mlp, label="MLP")
plt.plot(result_fno, label="FNO")
plt.plot(result_cno, label="CNO")
plt.plot(result_cfno, label="CFNO")
plt.xlabel("Epoch")
plt.ylabel("MSE Residual")
plt.yscale("log")
plt.legend()
plt.show()


