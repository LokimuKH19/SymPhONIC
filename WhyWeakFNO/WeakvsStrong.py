import torch
import torch.nn as nn
import torch.fft
import numpy as np

# -----------------------------
# Select Sampling points
# -----------------------------
def generate_data(num_samples=50, grid_size=64):
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y, indexing='ij')
    data_u, data_v = [], []
    for _ in range(num_samples):
        u0 = np.sin(np.pi*X)*np.sin(np.pi*Y)*np.random.rand()
        v0 = np.cos(np.pi*X)*np.cos(np.pi*Y)*np.random.rand()
        data_u.append(u0)
        data_v.append(v0)
    data_u = torch.tensor(np.array(data_u), dtype=torch.float32).unsqueeze(1)
    data_v = torch.tensor(np.array(data_v), dtype=torch.float32).unsqueeze(1)
    return data_u, data_v

train_u, train_v = generate_data(50)
test_u, test_v = generate_data(10)

# -----------------------------
# FNO Layer
# -----------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_channels, out_channels, modes, modes, 2) / (in_channels*out_channels))
        self.modes = modes

    def compl_mul2d(self, input, weights):
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bixy, ioxy -> boxy", input, cweights)

    def forward(self, x):
        B,C,H,W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B,self.weights.shape[1],H,W//2+1,dtype=torch.cfloat,device=x.device)
        out_ft[:,:, :self.modes, :self.modes] = self.compl_mul2d(x_ft[:,:, :self.modes, :self.modes], self.weights)
        x = torch.fft.irfft2(out_ft, s=(H,W), norm="ortho")
        return x

class FNO2d(nn.Module):
    def __init__(self, modes=12, width=32):
        super().__init__()
        self.fc0 = nn.Linear(2,width)
        self.conv0 = SpectralConv2d(width,width,modes)
        self.conv1 = SpectralConv2d(width,width,modes)
        self.conv2 = SpectralConv2d(width,width,modes)
        self.conv3 = SpectralConv2d(width,width,modes)
        self.w0 = nn.Conv2d(width,width,1)
        self.w1 = nn.Conv2d(width,width,1)
        self.w2 = nn.Conv2d(width,width,1)
        self.w3 = nn.Conv2d(width,width,1)
        self.fc1 = nn.Linear(width,128)
        self.fc2 = nn.Linear(128,2)

    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.fc0(x)
        x = x.permute(0,3,1,2)
        x1 = self.conv0(x) + self.w0(x)
        x2 = self.conv1(torch.relu(x1)) + self.w1(torch.relu(x1))
        x3 = self.conv2(torch.relu(x2)) + self.w2(torch.relu(x2))
        x4 = self.conv3(torch.relu(x3)) + self.w3(torch.relu(x3))
        x = x4.permute(0,2,3,1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0,3,1,2)
        return x

# -----------------------------
# Weak Form Residual (Maybe wrong???)
# -----------------------------
def burgers_weak_residual(u, v, phi_u, phi_v, dx=1/64, dy=1/64, nu=0.01):
    u_x = (u[:,:,1:,:] - u[:,:,:-1,:]) / dx
    u_y = (u[:,:,:,1:] - u[:,:,:,:-1]) / dy
    v_x = (v[:,:,1:,:] - v[:,:,:-1,:]) / dx
    v_y = (v[:,:,:,1:] - v[:,:,:,:-1]) / dy

    H = min(u_x.shape[2], u_y.shape[2])
    W = min(u_x.shape[3], u_y.shape[3])
    u_x = u_x[:,:,:H,:W]; u_y = u_y[:,:,:H,:W]
    v_x = v_x[:,:,:H,:W]; v_y = v_y[:,:,:H,:W]

    grad_u = torch.stack([u_x, u_y], dim=-1)
    grad_v = torch.stack([v_x, v_y], dim=-1)

    phi_u_x = (phi_u[:,:,1:,:] - phi_u[:,:,:-1,:]) / dx
    phi_u_y = (phi_u[:,:,:,1:] - phi_u[:,:,:,:-1]) / dy
    phi_v_x = (phi_v[:,:,1:,:] - phi_v[:,:,:-1,:]) / dx
    phi_v_y = (phi_v[:,:,:,1:] - phi_v[:,:,:,:-1]) / dy

    phi_u_x = phi_u_x[:,:,:H,:W]; phi_u_y = phi_u_y[:,:,:H,:W]
    phi_v_x = phi_v_x[:,:,:H,:W]; phi_v_y = phi_v_y[:,:,:H,:W]

    grad_phi_u = torch.stack([phi_u_x, phi_u_y], dim=-1)
    grad_phi_v = torch.stack([phi_v_x, phi_v_y], dim=-1)

    diffusion_u = (nu * grad_u * grad_phi_u).mean()
    diffusion_v = (nu * grad_v * grad_phi_v).mean()

    u_c = u[:,:,1:1+H,1:1+W]; v_c = v[:,:,1:1+H,1:1+W]
    u_x_c = u_x; u_y_c = u_y
    v_x_c = v_x; v_y_c = v_y
    phi_u_c = phi_u[:,:,1:1+H,1:1+W]; phi_v_c = phi_v[:,:,1:1+H,1:1+W]

    convection_u = ((u_c*u_x_c + v_c*u_y_c)*phi_u_c).mean()
    convection_v = ((u_c*v_x_c + v_c*v_y_c)*phi_v_c).mean()

    loss_u = diffusion_u + convection_u
    loss_v = diffusion_v + convection_v

    return loss_u, loss_v

# -----------------------------
# Strong Form Residual
# -----------------------------
def burgers_strong_residual(u, v, dx=1/64, dy=1/64, nu=0.01):
    u_x = (u[:,:,2:,:] - u[:,:,:-2,:]) / (2*dx)
    u_y = (u[:,:,:,2:] - u[:,:,:,:-2]) / (2*dy)
    u_xx = (u[:,:,2:,:] - 2*u[:,:,1:-1,:] + u[:,:,:-2,:]) / dx**2
    u_yy = (u[:,:,:,2:] - 2*u[:,:,:,1:-1] + u[:,:,:,:-2]) / dy**2

    v_x = (v[:,:,2:,:] - v[:,:,:-2,:]) / (2*dx)
    v_y = (v[:,:,:,2:] - v[:,:,:,:-2]) / (2*dy)
    v_xx = (v[:,:,2:,:] - 2*v[:,:,1:-1,:] + v[:,:,:-2,:]) / dx**2
    v_yy = (v[:,:,:,2:] - 2*v[:,:,:,1:-1] + v[:,:,:,:-2]) / dy**2

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

    r_u = u_x + u_y + nu * (u_xx + u_yy)
    r_v = v_x + v_y + nu * (v_xx + v_yy)

    return r_u, r_v

# -----------------------------
# Training
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fno = FNO2d().to(device)
optimizer = torch.optim.Adam(fno.parameters(), lr=1e-3)

train_data = torch.cat([train_u, train_v], dim=1).to(device)
phi_u = torch.randn_like(train_u).to(device)
phi_v = torch.randn_like(train_v).to(device)

epochs = 300
num_elements = 1
for epoch in range(epochs):
    optimizer.zero_grad()
    pred = fno(train_data)
    loss_u, loss_v = burgers_weak_residual(pred[:,0:1], pred[:,1:2], phi_u, phi_v)
    loss = loss_u**2 + loss_v**2
    loss_strong_u, loss_strong_v = burgers_strong_residual(pred[:, 0:1], pred[:, 1:2])
    # adding several points randomly... into the loss function
    indices = torch.randint(0, loss_strong_u.size(0), (num_elements,))
    loss += (loss_strong_u[indices]**2).mean() + (loss_strong_v[indices]**2).mean()
    # the real loss
    strong_loss = (loss_strong_u**2).mean() + (loss_strong_v**2).mean()
    print(f"Epoch {epoch+1}: Test Residual MSE {strong_loss}, Used Strong Points {num_elements}")
    loss.backward()
    optimizer.step()
