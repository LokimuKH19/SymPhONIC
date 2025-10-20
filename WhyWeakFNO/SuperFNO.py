import torch
import torch.nn as nn
import torch.fft
import numpy as np
from matplotlib import pyplot as plt
import random
from CFNOvsCNOvsFNO_PINN import CNO2d_small, CFNO2d_small, FNO2d_small, CFNOBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Explored the NeuralOperator(FieldEncoder(Coord)) Structure


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


# use FDM
def generate_burgers_data(num_samples=50, grid_size=64, T=0.1, dt=0.001, nu=0.01):
    """
    Generate Burgers Equation Data
    u_t + u u_x + v u_y = nu (u_xx + u_yy)
    """
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y, indexing='ij')

    data_u0, data_v0, data_uT, data_vT = [], [], [], []

    for _ in range(num_samples):
        u = np.sin(np.pi*X)*np.sin(np.pi*Y) * np.random.rand()
        v = np.cos(np.pi*X)*np.cos(np.pi*Y) * np.random.rand()
        u0, v0 = u.copy(), v.copy()
        nt = int(T/dt)
        dx, dy = 1/(grid_size-1), 1/(grid_size-1)

        for _ in range(nt):
            u_x = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx)
            u_y = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dy)
            u_xx = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2
            u_yy = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dy**2

            v_x = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2*dx)
            v_y = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2*dy)
            v_xx = (np.roll(v, -1, axis=0) - 2*v + np.roll(v, 1, axis=0)) / dx**2
            v_yy = (np.roll(v, -1, axis=1) - 2*v + np.roll(v, 1, axis=1)) / dy**2

            u = u - dt*(u*u_x + v*u_y) + dt*nu*(u_xx + u_yy)
            v = v - dt*(u*v_x + v*v_y) + dt*nu*(v_xx + v_yy)

        data_u0.append(u0)
        data_v0.append(v0)
        data_uT.append(u)
        data_vT.append(v)

    # Turns to a torch tensor
    data_u0 = torch.tensor(np.array(data_u0), dtype=torch.float32).unsqueeze(1)
    data_v0 = torch.tensor(np.array(data_v0), dtype=torch.float32).unsqueeze(1)
    data_uT = torch.tensor(np.array(data_uT), dtype=torch.float32).unsqueeze(1)
    data_vT = torch.tensor(np.array(data_vT), dtype=torch.float32).unsqueeze(1)

    return data_u0, data_v0, data_uT, data_vT


# Encoding the Coordinate
class CoordMLP(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, coords):
        B,H,W,_ = coords.shape
        x = coords.view(B*H*W, 2)
        out = self.net(x)
        out = out.view(B,H,W,-1).permute(0,3,1,2)  # [B,C,H,W]
        return out


# New CoordEncoder
class CoordCNN(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        # The Input [B, 2, H, W]
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1),  # Keep H,W
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, coords):
        """
        coords: [B,H,W,2] -> shifted into [B,2,H,W] -> CNN
        Output: [B,C,H,W]
        """
        coords = coords.permute(0,3,1,2)  # [B,2,H,W]
        out = self.net(coords)            # [B,hidden_dim,H,W]
        return out


def burgers_residual(u, v, dx=1/64, dy=1/64, nu=0.01):
    u_x = (u[:,:,2:,:] - u[:,:,:-2,:]) / (2*dx)
    u_y = (u[:,:,:,2:] - u[:,:,:,:-2]) / (2*dy)
    u_xx = (u[:,:,2:,:] - 2*u[:,:,1:-1,:] + u[:,:,:-2,:]) / dx**2
    u_yy = (u[:,:,:,2:] - 2*u[:,:,:,1:-1] + u[:,:,:,:-2]) / dy**2

    v_x = (v[:,:,2:,:] - v[:,:,:-2,:]) / (2*dx)
    v_y = (v[:,:,:,2:] - v[:,:,:,:-2]) / (2*dy)
    v_xx = (v[:,:,2:,:] - 2*v[:,:,1:-1,:] + v[:,:,:-2,:]) / dx**2
    v_yy = (v[:,:,:,2:] - 2*v[:,:,:,1:-1] + v[:,:,:,:-2]) / dy**2

    # Cut & Margin
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


# avoid trivial solution
def apply_boundary_conditions(u, v):
    # u, v: [B,1,H,W]
    B, _, H, W = u.shape

    # Top and Bottom no-slip
    u[:, :, 0, :] = 0.0
    u[:, :, -1, :] = 0.0
    v[:, :, 0, :] = 0.0
    v[:, :, -1, :] = 0.0

    # Inlet: Dirichlet
    u[:, :, :, 0] = 1    # todo control the inlet flow rate/Re=U*L/nu
    v[:, :, :, 0] = 0.0

    # Outlet: Neumann
    u[:, :, :, -1] = u[:, :, :, -2]
    v[:, :, :, -1] = v[:, :, :, -2]

    return u, v


if __name__ == '__main__':
    # Data
    train_u0, train_v0, train_uT, train_vT = generate_burgers_data(num_samples=50)
    train_data = torch.cat([train_u0, train_v0], dim=1).to(device)
    train_label = torch.cat([train_uT, train_vT], dim=1).to(device)

    # CoordPINN embedding
    coord_pinn_fno = CoordCNN(hidden_dim=32).to(device)
    coord_pinn_cno = CoordCNN(hidden_dim=32).to(device)
    coord_pinn_cfno = CoordCNN(hidden_dim=32).to(device)

    # Initialization
    fno = FNO2d_small(modes=16, width=16, depth=3, input_features=32, output_features=2).to(device)
    cno = CNO2d_small(cheb_modes=(8, 8), width=16, depth=3, input_features=32, output_features=2).to(device)
    cfno = CFNO2d_small(modes=16, cheb_modes=(8, 8), width=16, depth=3, alpha_init=0.5,
                        input_features=32, output_features=2).to(device)

    # Optimizer
    optimizer_fno = torch.optim.Adam(list(fno.parameters()) + list(coord_pinn_fno.parameters()), lr=1e-3)
    optimizer_cno = torch.optim.Adam(list(cno.parameters()) + list(coord_pinn_cno.parameters()), lr=1e-3)
    optimizer_cfno = torch.optim.Adam(list(cfno.parameters()) + list(coord_pinn_cfno.parameters()), lr=1e-3)

    epochs = 600
    beta = 0.0  # Weight of data loss
    H, W = train_u0.shape[2], train_u0.shape[3]
    coords = torch.stack(torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij'), dim=-1)
    coords = coords.unsqueeze(0).repeat(train_data.shape[0], 1, 1, 1).to(device)  # [B,H,W,2]

    # Record Loss
    history_fno, history_cno, history_cfno = [], [], []

    for epoch in range(epochs):
        # ---------------- FNO ----------------
        optimizer_fno.zero_grad()
        pinn_feat = coord_pinn_fno(coords)
        u_pred = fno(pinn_feat)
        pred_u, pred_v = apply_boundary_conditions(u_pred[:, 0:1], u_pred[:, 1:2])
        r_u, r_v = burgers_residual(pred_u, pred_v)
        loss_phys = (r_u**2).mean() + (r_v**2).mean()
        loss_sup = ((torch.cat([pred_u, pred_v], dim=1) - train_label)**2).mean()
        loss_fno = loss_phys + beta * loss_sup
        loss_fno.backward()
        optimizer_fno.step()
        history_fno.append((loss_phys.item(), loss_sup.item()))

        # ---------------- CNO ----------------
        optimizer_cno.zero_grad()
        pinn_feat = coord_pinn_cno(coords)
        u_pred = cno(pinn_feat)
        pred_u, pred_v = apply_boundary_conditions(u_pred[:, 0:1], u_pred[:, 1:2])
        r_u, r_v = burgers_residual(pred_u, pred_v)
        loss_phys = (r_u**2).mean() + (r_v**2).mean()
        loss_sup = ((torch.cat([pred_u, pred_v], dim=1) - train_label)**2).mean()
        loss_cno_val = loss_phys + beta * loss_sup
        loss_cno_val.backward()
        optimizer_cno.step()
        history_cno.append((loss_phys.item(), loss_sup.item()))

        # ---------------- CFNO ----------------
        optimizer_cfno.zero_grad()
        pinn_feat = coord_pinn_cfno(coords)
        u_pred = cfno(pinn_feat)
        pred_u, pred_v = apply_boundary_conditions(u_pred[:, 0:1], u_pred[:, 1:2])
        r_u, r_v = burgers_residual(pred_u, pred_v)
        loss_phys = (r_u**2).mean() + (r_v**2).mean()
        loss_sup = ((torch.cat([pred_u, pred_v], dim=1) - train_label)**2).mean()
        loss_cfno_val = loss_phys + beta * loss_sup
        loss_cfno_val.backward()
        optimizer_cfno.step()
        history_cfno.append((loss_phys.item(), loss_sup.item()))

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"FNO Phys={history_fno[-1][0]:.6f}, Sup={history_fno[-1][1]:.6f} | "
                  f"CNO Phys={history_cno[-1][0]:.6f}, Sup={history_cno[-1][1]:.6f} | "
                  f"CFNO Phys={history_cfno[-1][0]:.6f}, Sup={history_cfno[-1][1]:.6f}")

