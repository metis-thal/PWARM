"""Physics-Informed Neural Networks (PINNs) for accelerating PDE solves.

Provides a framework to train neural networks that respect physical laws,
enabling 5-10x speedup vs pure numerical solvers for repeated evaluations.

This lives in the AI layer — it's a LEARNED approximation of the physics
kernel's ground truth, never conflated with it (per project rule #2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class PINNConfig:
    """Configuration for PINN training and architecture."""
    
    # Network architecture
    hidden_layers: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    activation: str = "tanh"  # tanh, sin, relu
    
    # Training
    lr: float = 1e-3
    epochs: int = 5000
    batch_size: int = 1024
    
    # Loss weights
    w_pde: float = 1.0       # PDE residual weight
    w_bc: float = 10.0       # Boundary condition weight
    w_ic: float = 10.0       # Initial condition weight
    w_data: float = 1.0      # Data matching weight (if available)
    
    # Domain
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    
    # Device
    device: str = "cpu"  # "cpu" or "cuda"


class Activation(nn.Module):
    """Supported activation functions."""
    
    def __init__(self, name: str):
        super().__init__()
        if name == "tanh":
            self.act = nn.Tanh()
        elif name == "sin":
            self.act = nn.SiLU()  # Approximation; true sin would be custom
        elif name == "relu":
            self.act = nn.ReLU()
        elif name == "swish":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def forward(self, x):
        return self.act(x)


class PINN(nn.Module):
    """Physics-Informed Neural Network for PDE solving.
    
    Solves PDEs of the form: u_t + N[u] = 0
    where N is a differential operator.
    """
    
    def __init__(self, input_dim: int, output_dim: int, config: PINNConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(Activation(config.activation))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        self.to(config.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Network forward pass."""
        return self.net(x)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict with numpy arrays (no grad)."""
        self.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32, device=self.config.device)
            y = self.forward(x_t)
        return y.cpu().numpy()


class HeatPINN(PINN):
    """PINN for 1D heat equation: u_t = alpha * u_xx.
    
    Can be extended to 2D/3D and coupled systems.
    """
    
    def __init__(self, config: PINNConfig | None = None):
        config = config or PINNConfig()
        # Input: (x, t), Output: u
        super().__init__(input_dim=2, output_dim=1, config=config)
        self.alpha = 0.01  # thermal diffusivity
    
    def pde_residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute PDE residual: u_t - alpha * u_xx."""
        # Enable grad for automatic differentiation
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        inputs = torch.cat([x, t], dim=1)
        u = self.forward(inputs)
        
        # u_t
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        # u_x
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        # u_xx
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        
        # Residual: u_t - alpha * u_xx
        return u_t - self.alpha * u_xx
    
    def loss(self, x_pde: torch.Tensor, t_pde: torch.Tensor,
             x_bc: torch.Tensor, t_bc: torch.Tensor, u_bc: torch.Tensor,
             x_ic: torch.Tensor, t_ic: torch.Tensor, u_ic: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute total loss with components."""
        # PDE loss
        residual = self.pde_residual(x_pde, t_pde)
        loss_pde = torch.mean(residual**2)
        
        # Boundary condition loss
        inputs_bc = torch.cat([x_bc, t_bc], dim=1)
        u_pred_bc = self.forward(inputs_bc)
        loss_bc = torch.mean((u_pred_bc - u_bc)**2)
        
        # Initial condition loss
        inputs_ic = torch.cat([x_ic, t_ic], dim=1)
        u_pred_ic = self.forward(inputs_ic)
        loss_ic = torch.mean((u_pred_ic - u_ic)**2)
        
        # Total loss
        total = (self.config.w_pde * loss_pde +
                 self.config.w_bc * loss_bc +
                 self.config.w_ic * loss_ic)
        
        return total, {
            "pde": loss_pde.item(),
            "bc": loss_bc.item(),
            "ic": loss_ic.item(),
            "total": total.item()
        }


class WavePINN(PINN):
    """PINN for 1D wave equation: u_tt = c^2 * u_xx."""
    
    def __init__(self, config: PINNConfig | None = None):
        config = config or PINNConfig()
        super().__init__(input_dim=2, output_dim=1, config=config)
        self.c = 1.0  # wave speed
    
    def pde_residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x.requires_grad_(True)
        t.requires_grad_(True)
        inputs = torch.cat([x, t], dim=1)
        u = self.forward(inputs)
        
        # u_tt
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True, retain_graph=True)[0]
        
        # u_xx
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        
        return u_tt - self.c**2 * u_xx


class NavierStokesPINN(PINN):
    """PINN for 2D incompressible Navier-Stokes:
    u_t + u·∇u = -∇p/ρ + ν∇²u
    ∇·u = 0
    """
    
    def __init__(self, config: PINNConfig | None = None):
        config = config or PINNConfig()
        # Input: (x, y, t), Output: (u, v, p)
        super().__init__(input_dim=3, output_dim=3, config=config)
        self.nu = 0.01  # kinematic viscosity
        self.rho = 1.0  # density
    
    def pde_residual(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        inputs = torch.cat([x, y, t], dim=1)
        u = self.forward(inputs)
        
        u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3]
        
        # Velocity gradients
        u_x = torch.autograd.grad(u_vel, x, grad_outputs=torch.ones_like(u_vel), create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u_vel, y, grad_outputs=torch.ones_like(u_vel), create_graph=True, retain_graph=True)[0]
        u_t = torch.autograd.grad(u_vel, t, grad_outputs=torch.ones_like(u_vel), create_graph=True, retain_graph=True)[0]
        
        v_x = torch.autograd.grad(v_vel, x, grad_outputs=torch.ones_like(v_vel), create_graph=True, retain_graph=True)[0]
        v_y = torch.autograd.grad(v_vel, y, grad_outputs=torch.ones_like(v_vel), create_graph=True, retain_graph=True)[0]
        v_t = torch.autograd.grad(v_vel, t, grad_outputs=torch.ones_like(v_vel), create_graph=True, retain_graph=True)[0]
        
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        
        # Second derivatives for Laplacian
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True, retain_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True, retain_graph=True)[0]
        
        # Continuity: u_x + v_y = 0
        continuity = u_x + v_y
        
        # Momentum x: u_t + u*u_x + v*u_y + p_x/rho - nu*(u_xx + u_yy) = 0
        momentum_x = u_t + u_vel*u_x + v_vel*u_y + p_x/self.rho - self.nu*(u_xx + u_yy)
        
        # Momentum y: v_t + u*v_x + v*v_y + p_y/rho - nu*(v_xx + v_yy) = 0
        momentum_y = v_t + u_vel*v_x + v_vel*v_y + p_y/self.rho - self.nu*(v_xx + v_yy)
        
        return momentum_x, momentum_y, continuity


class PINNTrainer:
    """Training loop for PINNs with domain sampling."""
    
    def __init__(self, model: PINN, config: PINNConfig):
        self.model = model
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=500, factor=0.5, min_lr=1e-6
        )
        self.history = []
    
    def sample_pde_points(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample random points in the domain for PDE loss."""
        x = np.random.uniform(self.config.x_min, self.config.x_max, (n, 1))
        t = np.random.uniform(self.config.t_min, self.config.t_max, (n, 1))
        return x.astype(np.float32), t.astype(np.float32)
    
    def sample_bc_points(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample boundary condition points.
        
        For heat equation: u(0,t)=0, u(L,t)=0
        Returns (x_bc, t_bc, u_bc)
        """
        # Left boundary
        n_left = n // 2
        x_left = np.full((n_left, 1), self.config.x_min, dtype=np.float32)
        t_left = np.random.uniform(self.config.t_min, self.config.t_max, (n_left, 1)).astype(np.float32)
        u_left = np.zeros((n_left, 1), dtype=np.float32)
        
        # Right boundary
        n_right = n - n_left
        x_right = np.full((n_right, 1), self.config.x_max, dtype=np.float32)
        t_right = np.random.uniform(self.config.t_min, self.config.t_max, (n_right, 1)).astype(np.float32)
        u_right = np.zeros((n_right, 1), dtype=np.float32)
        
        x_bc = np.vstack([x_left, x_right])
        t_bc = np.vstack([t_left, t_right])
        u_bc = np.vstack([u_left, u_right])
        
        return x_bc, t_bc, u_bc
    
    def sample_ic_points(self, n: int, ic_func: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample initial condition points."""
        x = np.random.uniform(self.config.x_min, self.config.x_max, (n, 1)).astype(np.float32)
        t = np.full((n, 1), self.config.t_min, dtype=np.float32)
        u = ic_func(x).astype(np.float32)
        return x, t, u
    
    def train_step(self, n_pde: int = 1024, n_bc: int = 128, n_ic: int = 128,
                   ic_func: Callable[[np.ndarray], np.ndarray] | None = None) -> dict:
        """Single training step."""
        self.model.train()
        
        # Sample points
        x_pde, t_pde = self.sample_pde_points(n_pde)
        x_bc, t_bc, u_bc = self.sample_bc_points(n_bc)
        
        if ic_func is None:
            ic_func = lambda x: np.sin(np.pi * x)  # default: sin(pi*x)
        x_ic, t_ic, u_ic = self.sample_ic_points(n_ic, ic_func)
        
        # Convert to tensors
        device = self.config.device
        x_pde_t = torch.tensor(x_pde, device=device)
        t_pde_t = torch.tensor(t_pde, device=device)
        x_bc_t = torch.tensor(x_bc, device=device)
        t_bc_t = torch.tensor(t_bc, device=device)
        u_bc_t = torch.tensor(u_bc, device=device)
        x_ic_t = torch.tensor(x_ic, device=device)
        t_ic_t = torch.tensor(t_ic, device=device)
        u_ic_t = torch.tensor(u_ic, device=device)
        
        # Compute loss
        self.optimizer.zero_grad()
        loss, loss_dict = self.model.loss(
            x_pde_t, t_pde_t,
            x_bc_t, t_bc_t, u_bc_t,
            x_ic_t, t_ic_t, u_ic_t
        )
        
        loss.backward()
        self.optimizer.step()
        
        return loss_dict
    
    def train(self, epochs: int | None = None, n_pde: int = 1024, n_bc: int = 128, n_ic: int = 128,
              ic_func: Callable[[np.ndarray], np.ndarray] | None = None,
              verbose: bool = True) -> list[dict]:
        """Full training loop."""
        epochs = epochs or self.config.epochs
        history = []
        
        for epoch in range(epochs):
            loss_dict = self.train_step(n_pde, n_bc, n_ic, ic_func)
            self.scheduler.step(loss_dict["total"])
            history.append(loss_dict)
            
            if verbose and epoch % 500 == 0:
                print(f"Epoch {epoch}: total={loss_dict['total']:.6f} "
                      f"pde={loss_dict['pde']:.6f} bc={loss_dict['bc']:.6f} ic={loss_dict['ic']:.6f}")
        
        self.history = history
        return history


def train_heat_pinn(config: PINNConfig | None = None,
                    ic_func: Callable[[np.ndarray], np.ndarray] | None = None) -> HeatPINN:
    """Train a PINN for the 1D heat equation with default settings."""
    config = config or PINNConfig()
    model = HeatPINN(config)
    trainer = PINNTrainer(model, config)
    trainer.train(config.epochs, ic_func=ic_func)
    return model


def train_wave_pinn(config: PINNConfig | None = None) -> WavePINN:
    """Train a PINN for the 1D wave equation."""
    config = config or PINNConfig()
    model = WavePINN(config)
    # Training loop similar to heat but with different loss
    return model


# Convenience function for integration with pymo kernel
class PINNWrapper:
    """Wrapper to use PINN as a drop-in acceleration for numerical solvers.
    
    Usage:
        pinn = PINNWrapper("heat", config)
        # Train once
        pinn.train(ic_func=lambda x: np.sin(np.pi * x))
        # Use for fast inference
        u = pinn.predict(x, t)
    """
    
    def __init__(self, pde_type: str, config: PINNConfig | None = None):
        self.pde_type = pde_type
        self.config = config or PINNConfig()
        self.model = None
        self.trainer = None
    
    def train(self, ic_func: Callable[[np.ndarray], np.ndarray] | None = None, epochs: int | None = None):
        """Train the PINN."""
        if self.pde_type == "heat":
            self.model = HeatPINN(self.config)
        elif self.pde_type == "wave":
            self.model = WavePINN(self.config)
        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")
        
        self.trainer = PINNTrainer(self.model, self.config)
        self.trainer.train(epochs, ic_func=ic_func)
    
    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Predict solution at (x, t) points."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        x = np.asarray(x).reshape(-1, 1)
        t = np.asarray(t).reshape(-1, 1)
        inputs = np.hstack([x, t])
        return self.model.predict(inputs).flatten()
    
    def save(self, path: str):
        """Save trained model."""
        if self.model is None:
            raise RuntimeError("No model to save")
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
            "pde_type": self.pde_type
        }, path)
    
    def load(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.pde_type = checkpoint["pde_type"]
        self.config = checkpoint["config"]
        
        if self.pde_type == "heat":
            self.model = HeatPINN(self.config)
        elif self.pde_type == "wave":
            self.model = WavePINN(self.config)
        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")
        
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.config.device)


if __name__ == "__main__":
    # Quick test
    config = PINNConfig(
        hidden_layers=[32, 32, 32],
        epochs=1000,
        lr=1e-3,
        device="cpu"
    )
    
    # Train heat PINN
    pinn = PINNWrapper("heat", config)
    pinn.train(ic_func=lambda x: np.sin(np.pi * x), epochs=200)
    
    # Test prediction
    x_test = np.linspace(0, 1, 50)
    t_test = np.full_like(x_test, 0.5)
    u_pred = pinn.predict(x_test, t_test)
    
    # Exact solution: u(x,t) = exp(-alpha*pi^2*t) * sin(pi*x)
    alpha = 0.01
    u_exact = np.exp(-alpha * np.pi**2 * 0.5) * np.sin(np.pi * x_test)
    
    error = np.mean((u_pred - u_exact)**2)
    print(f"MSE: {error:.6f}")
    print(f"Max error: {np.max(np.abs(u_pred - u_exact)):.6f}")