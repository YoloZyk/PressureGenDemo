"""
Flow Matching implementation for continuous normalizing flows.

Flow matching learns a velocity field v(x, t) that transports samples from a noise
distribution to the data distribution via ordinary differential equations (ODEs).

Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2023)
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class FlowMatching(nn.Module):
    """
    Flow Matching model that learns to transport noise to data via velocity fields.

    Args:
        model: Neural network that predicts velocity field v(x, t)
               Should take (x, t) and return predicted velocity of same shape as x
        sigma: Standard deviation for conditional flow matching (default: 0.0)
               When sigma > 0, uses conditional flow matching with Gaussian paths
    """

    def __init__(self, model, sigma=0.0):
        super().__init__()
        self.model = model
        self.sigma = sigma

    def forward(self, x_0, return_loss=True):
        """
        Compute flow matching loss for a batch of data.

        Forward process:
            - Sample t uniformly from [0, 1]
            - Sample noise x_1 ~ N(0, I)
            - Interpolate: x_t = t * x_0 + (1 - t) * x_1
            - True velocity: v_t = x_0 - x_1
            - Loss: MSE(predicted_velocity, true_velocity)

        Args:
            x_0: Clean data samples (B, D)
            return_loss: If True, return scalar loss. If False, return per-sample losses

        Returns:
            loss: Scalar loss if return_loss=True, else (B,) tensor of per-sample losses
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps uniformly from [0, 1]
        t = torch.rand(batch_size, device=device)

        # Sample noise from standard normal
        x_1 = torch.randn_like(x_0)

        # Interpolate between noise and data
        # x_t = t * x_0 + (1 - t) * x_1
        t_expanded = t.view(batch_size, *([1] * (x_0.ndim - 1)))  # (B, 1, 1, ...)
        x_t = t_expanded * x_0 + (1 - t_expanded) * x_1

        # True velocity field: dx_t/dt = x_0 - x_1
        # This is the derivative of the linear interpolation
        true_velocity = x_0 - x_1

        # Add Gaussian noise for conditional flow matching (if sigma > 0)
        if self.sigma > 0:
            x_t = x_t + self.sigma * torch.randn_like(x_t)

        # Predict velocity using the model
        predicted_velocity = self.model(x_t, t)

        # Compute MSE loss
        loss = (predicted_velocity - true_velocity) ** 2

        if return_loss:
            return loss.mean()
        else:
            # Return per-sample loss (averaged over dimensions)
            return loss.view(batch_size, -1).mean(dim=1)

    @torch.no_grad()
    def sample(self,
               sample_shape,
               device='cuda',
               num_steps=100,
               method='euler',
               return_intermediates=False,
               save_interval=10,
               verbose=True):
        """
        Generate samples by solving the ODE: dx/dt = v(x, t) from t=0 to t=1.

        Args:
            sample_shape: Shape of samples to generate (B, D)
            device: Device to generate samples on
            num_steps: Number of integration steps (default: 100)
            method: ODE solver method ('euler', 'midpoint', 'rk4')
            return_intermediates: If True, return intermediate states
            save_interval: Save intermediate states every N steps (if return_intermediates=True)
            verbose: Show progress bar

        Returns:
            samples: Generated samples (B, D)
            intermediates: List of (t, x_t) tuples (if return_intermediates=True)
        """
        # Start from noise at t=0
        x = torch.randn(sample_shape, device=device)

        # Time steps from 0 to 1
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)
        dt = 1.0 / num_steps

        intermediates = []
        if return_intermediates:
            intermediates.append((0.0, x.cpu().clone()))

        # Progress bar
        iterator = tqdm(range(num_steps), desc='Sampling') if verbose else range(num_steps)

        for i in iterator:
            t = timesteps[i]

            # Create batch of timesteps
            t_batch = torch.full((sample_shape[0],), t, device=device)

            if method == 'euler':
                # Euler method: x_{t+dt} = x_t + dt * v(x_t, t)
                v = self.model(x, t_batch)
                x = x + dt * v

            elif method == 'midpoint':
                # Midpoint method (RK2)
                # k1 = v(x_t, t)
                # k2 = v(x_t + 0.5*dt*k1, t + 0.5*dt)
                # x_{t+dt} = x_t + dt * k2
                k1 = self.model(x, t_batch)
                t_mid = t + 0.5 * dt
                t_mid_batch = torch.full((sample_shape[0],), t_mid, device=device)
                k2 = self.model(x + 0.5 * dt * k1, t_mid_batch)
                x = x + dt * k2

            elif method == 'rk4':
                # Classic RK4 method
                t_half = t + 0.5 * dt
                t_next = t + dt

                t_batch_half = torch.full((sample_shape[0],), t_half, device=device)
                t_batch_next = torch.full((sample_shape[0],), t_next, device=device)

                k1 = self.model(x, t_batch)
                k2 = self.model(x + 0.5 * dt * k1, t_batch_half)
                k3 = self.model(x + 0.5 * dt * k2, t_batch_half)
                k4 = self.model(x + dt * k3, t_batch_next)

                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            else:
                raise ValueError(f"Unknown method: {method}. Choose from 'euler', 'midpoint', 'rk4'")

            # Save intermediate states
            if return_intermediates and (i + 1) % save_interval == 0:
                intermediates.append((timesteps[i + 1].item(), x.cpu().clone()))

        # Final state at t=1
        if return_intermediates:
            if len(intermediates) == 0 or intermediates[-1][0] != 1.0:
                intermediates.append((1.0, x.cpu().clone()))
            return x, intermediates

        return x

    @torch.no_grad()
    def sample_ode(self,
                   sample_shape,
                   device='cuda',
                   rtol=1e-5,
                   atol=1e-5,
                   method='dopri5',
                   return_intermediates=False,
                   verbose=True):
        """
        Generate samples using adaptive ODE solvers from torchdiffeq.

        This method uses scipy-style adaptive solvers that automatically
        adjust step sizes for accuracy. Requires: pip install torchdiffeq

        Args:
            sample_shape: Shape of samples to generate (B, D)
            device: Device to generate samples on
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver
            method: Solver method ('dopri5', 'dopri8', 'adams', 'rk4', etc.)
            return_intermediates: If True, return trajectory
            verbose: Print status messages

        Returns:
            samples: Generated samples (B, D)
            trajectory: Full trajectory if return_intermediates=True
        """
        try:
            from torchdiffeq import odeint
        except ImportError:
            raise ImportError(
                "torchdiffeq is required for adaptive ODE solvers. "
                "Install with: pip install torchdiffeq"
            )

        if verbose:
            print(f"Sampling with adaptive ODE solver: {method}")

        # Start from noise at t=0
        x_0 = torch.randn(sample_shape, device=device)

        # Define velocity field function
        def velocity_fn(t, x):
            # t is a scalar tensor, need to broadcast to batch
            t_batch = torch.full((sample_shape[0],), t.item(), device=device)
            return self.model(x, t_batch)

        # Integration time points
        if return_intermediates:
            # Return full trajectory with 100 points
            t_span = torch.linspace(0, 1, 100, device=device)
        else:
            # Just start and end points
            t_span = torch.tensor([0.0, 1.0], device=device)

        # Solve ODE
        trajectory = odeint(
            velocity_fn,
            x_0,
            t_span,
            rtol=rtol,
            atol=atol,
            method=method
        )

        # Extract final state
        x_final = trajectory[-1]

        if return_intermediates:
            # Convert trajectory to list of (t, x) tuples
            intermediates = [(t.item(), x.cpu()) for t, x in zip(t_span, trajectory)]
            return x_final, intermediates

        return x_final
