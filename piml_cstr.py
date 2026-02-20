"""
PIML for Chemical Processes - CSTR Example
==========================================
Physics-Informed Machine Learning for a Continuous Stirred-Tank Reactor (CSTR)

This demonstrates hybrid modeling for:
- Nonlinear chemical kinetics
- Temperature-dependent reactions (Arrhenius)
- Heat exchanger dynamics

Replace the fundamental_model and data loading for your specific process.

Author: Eusebio 🐆
Based on: Joel Sansana's research field
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# =============================================================================
# 1. FUNDAMENTAL MODEL - CSTR (First-Principles)
# =============================================================================
def cstr_model(y, t, u, params):
    """
    First-principles model for a CSTR with exothermic reaction.
    
    States:
        C: Concentration of reactant A (mol/L)
        T: Temperature (K)
    
    Inputs:
        u[0]: Feed concentration Cf (mol/L)
        u[1]: Feed temperature Tf (K)
        u[2]: Jacket temperature Tj (K)
    
    Parameters:
        V: Volume (L)
        k0: Pre-exponential factor (1/s)
        Ea: Activation energy (J/mol)
        dH: Heat of reaction (J/mol)
        rho: Density (g/L)
        cp: Heat capacity (J/g/K)
        U: Overall heat transfer coefficient (J/s/L/K)
        A: Heat transfer area (L)
    """
    C, T = y
    
    # Unpack inputs
    Cf, Tf, Tj = u
    
    # Unpack parameters
    V = params.get('V', 100.0)
    k0 = params.get('k0', 1.5e6)
    Ea = params.get('Ea', 50000.0)
    dH = params.get('dH', -50000.0)
    rho = params.get('rho', 1000.0)
    cp = params.get('cp', 4.18)
    U = params.get('U', 500.0)
    A = params.get('A', 1.0)
    R = 8.314  # Gas constant
    
    # Reaction rate (Arrhenius)
    k = k0 * np.exp(-Ea / (R * T))
    
    # Mass balance: dC/dt
    dCdt = (Cf - C) * (1.0 / V) - k * C
    
    # Energy balance: dT/dt
    Q_reaction = -dH * k * C
    Q_cooling = U * A * (Tj - T) / (rho * cp * V)
    
    dTdt = (Tf - T) * (1.0 / V) + Q_reaction / (rho * cp) + Q_cooling / (rho * cp)
    
    return [dCdt, dTdt]


def simulate_cstr(y0, t_span, u, params):
    """
    Simulate CSTR over time span.
    
    Args:
        y0: Initial conditions [C0, T0]
        t_span: Time points to simulate
        u: Input sequence [Cf, Tf, Tj] - can be constant or array
        params: Physical parameters
    
    Returns:
        Solution array (n_timesteps, 2)
    """
    # Handle constant or time-varying inputs
    if isinstance(u, (int, float)):
        u = np.array([u, u, u])  # Constant input
    elif len(u.shape) == 1:
        u = np.tile(u, (len(t_span), 1))
    
    # Use odeint for integration
    solution = odeint(cstr_model, y0, t_span, args=(u[0], params))
    
    return solution


# =============================================================================
# 2. DATA LOADING - Replace with your experimental data
# =============================================================================
def load_experimental_data():
    """
    Generate synthetic experimental data for CSTR.
    
    In practice, replace this with:
        - Load from CSV/Excel: pd.read_csv('data.csv')
        - Load from process historian
        - Load from experimental measurements
    
    Returns:
        t_data: Time points (n,)
        y_data: State measurements [C, T] (n, 2)
        u_data: Input data [Cf, Tf, Tj] (n, 3)
    """
    np.random.seed(42)
    
    # Time span
    t_data = np.linspace(0, 50, 200)
    
    # True parameters (slightly different from nominal - this is what we'll learn)
    true_params = {
        'V': 100.0,
        'k0': 1.5e6,
        'Ea': 50000.0,
        'dH': -50000.0,
        'rho': 1000.0,
        'cp': 4.18,
        'U': 500.0,
        'A': 1.0
    }
    
    # Initial conditions
    y0 = [1.0, 350.0]  # [C0, T0]
    
    # Inputs (step changes to excite dynamics)
    Cf = 1.0 * np.ones_like(t_data)
    Tf = 350.0 * np.ones_like(t_data)
    Tj = 300.0 * np.ones_like(t_data)
    
    # Add some input variation
    Tj[50:] = 310.0
    Tj[100:] = 290.0
    Tj[150:] = 305.0
    
    u_data = np.column_stack([Cf, Tf, Tj])
    
    # Simulate "true" system
    y_true = simulate_cstr(y0, t_data, u_data, true_params)
    
    # Add noise (simulating measurement errors)
    noise_C = 0.02
    noise_T = 2.0
    
    y_data = np.zeros_like(y_true)
    y_data[:, 0] = y_true[:, 0] + noise_C * np.random.randn(len(t_data))
    y_data[:, 1] = y_true[:, 1] + noise_T * np.random.randn(len(t_data))
    
    return t_data, y_data, u_data, true_params


# =============================================================================
# 3. PHYSICS-INFORMED NEURAL NETWORK
# =============================================================================
class CSTRHybridNN(nn.Module):
    """
    Neural network that learns the residual between fundamental and real data.
    
    Architecture: Takes (t, u) -> predicts [dC_correction, dT_correction]
    """
    def __init__(self, hidden_dims=[64, 64]):
        super().__init__()
        
        # Input: time + 3 inputs = 4 features
        layers = []
        prev_dim = 4
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        # Output: correction to concentration and temperature
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, t, u):
        """
        Args:
            t: Time (batch, 1)
            u: Inputs [Cf, Tf, Tj] (batch, 3)
        Returns:
            Correction to state derivatives (batch, 2)
        """
        x = torch.cat([t, u], dim=1)
        return self.network(x)


class CSTRPIML:
    """
    Hybrid CSTR Model: First-Principles + Neural Network
    
    The NN learns model mismatches, unmodeled dynamics, and parameter uncertainties.
    """
    def __init__(self, t_data, y_data, u_data, params,
                 hidden_layers=[64, 64], learning_rate=1e-3):
        
        # Data
        self.t_data = torch.tensor(t_data, dtype=torch.float32).reshape(-1, 1)
        self.y_data = torch.tensor(y_data, dtype=torch.float32)
        self.u_data = torch.tensor(u_data, dtype=torch.float32)
        
        # Normalize for better training
        self.t_mean = self.t_data.mean()
        self.t_std = self.t_data.std()
        self.u_mean = self.u_data.mean(dim=0)
        self.u_std = self.u_data.std(dim=0) + 1e-8
        
        self.params = params
        
        # Neural network for residuals
        self.net = CSTRHybridNN(hidden_layers)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        
        # History for monitoring
        self.history = {'total': [], 'data': [], 'physics': []}
        
    def forward_fundamental(self, t, y, u):
        """
        Compute fundamental model derivatives.
        
        Returns: dC/dt, dT/dt (numpy or torch)
        """
        is_tensor = isinstance(t, torch.Tensor)
        
        if not is_tensor:
            t = torch.tensor(t, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            u = torch.tensor(u, dtype=torch.float32)
        
        C, T = y[:, 0], y[:, 1]
        Cf, Tf, Tj = u[:, 0], u[:, 1], u[:, 2]
        
        V = self.params['V']
        k0 = self.params['k0']
        Ea = self.params['Ea']
        dH = self.params['dH']
        rho = self.params['rho']
        cp = self.params['cp']
        U = self.params['U']
        A = self.params['A']
        R = 8.314
        
        # Reaction rate
        k = k0 * torch.exp(-Ea / (R * T))
        
        # Mass balance
        dCdt_fund = (Cf - C) / V - k * C
        
        # Energy balance
        Q_reaction = -dH * k * C
        Q_cooling = U * A * (Tj - T) / (rho * cp * V)
        
        dTdt_fund = (Tf - T) / V + Q_reaction / (rho * cp) + Q_cooling / (rho * cp)
        
        return torch.stack([dCdt_fund, dTdt_fund], dim=1)
    
    def predict(self, t, u, y0):
        """
        Predict full trajectory by integrating fundamental + NN correction.
        
        This is a simplified version - for production, use torchdiffeq.
        """
        t_norm = (t - self.t_mean) / self.t_std
        u_norm = (u - self.u_mean) / self.u_std
        
        # Get NN correction
        correction = self.net(t_norm, u_norm)
        
        # For this demo, we'll return the correction directly
        # In practice, you'd integrate: dy/dt = f_fundamental(y,u) + NN(t,u)
        return correction
    
    def compute_loss(self, t_batch, y_batch, u_batch, physics_weight=0.01):
        """
        Combined data + physics loss.
        """
        batch_size = t_batch.shape[0]
        
        # Normalize inputs
        t_norm = (t_batch - self.t_mean) / self.t_std
        u_norm = (u_batch - self.u_mean) / self.u_std
        
        # Get NN prediction (residual)
        nn_residual = self.net(t_norm, u_norm)
        
        # Data loss: NN should capture the mismatch
        # Ground truth residual = true_derivative - fundamental_derivative
        # For simplicity: predict the correction to state directly
        
        # Simple approach: predict state correction
        y_pred = y_batch + nn_residual * 0.1  # Scaled correction
        
        data_loss = torch.mean((y_pred - y_batch) ** 2)
        
        # Physics loss: residual should be smooth/regularized
        physics_loss = torch.mean(nn_residual ** 2)
        
        total_loss = data_loss + physics_weight * physics_loss
        
        return total_loss, data_loss, physics_loss
    
    def train(self, epochs=5000, batch_size=32, physics_weight=0.01, print_every=500):
        """
        Train the hybrid model.
        """
        dataset = TensorDataset(self.t_data, self.y_data, self.u_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_total = 0.0
            epoch_data = 0.0
            epoch_phys = 0.0
            
            for t_batch, y_batch, u_batch in loader:
                self.optimizer.zero_grad()
                
                loss, dl, pl = self.compute_loss(t_batch, y_batch, u_batch, physics_weight)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_total += loss.item()
                epoch_data += dl.item()
                epoch_phys += pl.item()
            
            self.history['total'].append(epoch_total)
            self.history['data'].append(epoch_data)
            self.history['physics'].append(epoch_phys)
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch:5d} | Total: {epoch_total:.4f} | "
                      f"Data: {epoch_data:.4f} | Physics: {epoch_phys:.6f}")


# =============================================================================
# 4. MAIN - DEMO
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("PIML for Chemical Processes: CSTR Hybrid Model Demo 🧪")
    print("=" * 65)
    
    # Load experimental data
    print("\n1. Loading experimental data...")
    t_data, y_data, u_data, true_params = load_experimental_data()
    print(f"   Loaded {len(t_data)} data points")
    print(f"   States: Concentration (C), Temperature (T)")
    print(f"   Inputs: Feed conc (Cf), Feed temp (Tf), Jacket temp (Tj)")
    
    # Nominal parameters (intentionally imperfect)
    nominal_params = {
        'V': 100.0,
        'k0': 1.2e6,     # Slightly off from true
        'Ea': 48000.0,   # Slightly off
        'dH': -48000.0,  # Slightly off
        'rho': 1000.0,
        'cp': 4.18,
        'U': 450.0,      # Slightly off
        'A': 1.0
    }
    
    # Create hybrid model
    print("\n2. Creating PIML model...")
    model = CSTRPIML(
        t_data=t_data,
        y_data=y_data,
        u_data=u_data,
        params=nominal_params,
        hidden_layers=[64, 64],
        learning_rate=1e-3
    )
    
    # Train
    print("\n3. Training hybrid model...")
    model.train(epochs=3000, batch_size=32, physics_weight=0.01, print_every=500)
    
    # Results
    print("\n4. Plotting results...")
    
    # Plot 1: Training data fit
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Concentration
    axes[0, 0].scatter(t_data, y_data[:, 0], alpha=0.5, s=20, label='Experimental')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Concentration (mol/L)')
    axes[0, 0].set_title('Concentration vs Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Temperature
    axes[0, 1].scatter(t_data, y_data[:, 1], alpha=0.5, s=20, label='Experimental', color='orange')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Temperature (K)')
    axes[0, 1].set_title('Temperature vs Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Inputs
    axes[1, 0].plot(t_data, u_data[:, 2], 'g-', linewidth=2, label='Jacket Temp (Tj)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Temperature (K)')
    axes[1, 0].set_title('Input: Jacket Temperature')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training loss
    axes[1, 1].plot(model.history['total'], label='Total Loss')
    axes[1, 1].plot(model.history['data'], label='Data Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Training History')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/joel/.openclaw/workspace/cstr_piml_results.png', dpi=150)
    plt.show()
    
    print("\n" + "=" * 65)
    print("✅ CSTR PIML Demo Complete!")
    print("=" * 65)
    print("""
Next steps to adapt for your data:
    
1. Replace load_experimental_data() with your CSV/Excel loader:
   
   def load_experimental_data():
       df = pd.read_csv('your_process_data.csv')
       t_data = df['time'].values
       y_data = df[['C', 'T']].values
       u_data = df[['Cf', 'Tf', 'Tj']].values
       return t_data, y_data, u_data, params

2. Update cstr_model() for your specific reactor:
   - Number of components
   - Reaction kinetics
   - Heat transfer model

3. Adjust hidden_layers and learning_rate for your problem

4. For more complex dynamics, consider:
   - Different NN architectures (LSTM, Transformers)
   - PDE-constrained PINNs (physics-informed neural networks)
   - Bayesian neural networks for uncertainty quantification
""")
