import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import t, norm

def solve_optimal_pnl_all(loc=40.0, scale=8.0):
    """
    Compares the optimal 'Invest & Expand' allocation across three different 
    statistical distributions for competitor behavior.
    
    loc: Expected Median (e.g., 40.0%).
    scale: Spread/Dispersion of the crowd.
    """
    # Define the three different probability models
    scenarios = [
        {"name": "Extreme Fat Tails (Cauchy, df=1)", "dist": lambda z: t.cdf(z, 1, loc=loc, scale=scale)},
        {"name": "Heavy Fat Tails (Student-t, df=2)", "dist": lambda z: t.cdf(z, 2, loc=loc, scale=scale)},
        {"name": "Normal Distribution (Gaussian)", "dist": lambda z: norm.cdf(z, loc=loc, scale=scale)}
    ]
    
    print(f"ASSUMPTION: The crowd's median Speed is {loc}% with a spread of {scale}%\n")
    print("="*60)
    
    for scenario in scenarios:
        best_z = 0
        best_pnl = -np.inf
        best_x = 0
        best_y = 0
        
        # Test Speed (z) from 0 to 100
        zs = np.linspace(0, 100, 1000)
        
        for z in zs:
            C = 100 - z
            if C <= 0:
                continue
                
            # Optimize Research (x) and Scale (y) for the remaining budget
            res = minimize_scalar(lambda x: -np.log(1+x)*(C-x), bounds=(0, C), method='bounded')
            x = res.x
            y = C - x
            
            # Calculate PnL components
            research = 200000 * np.log(1+x)/np.log(101)
            scale_val = 7 * y / 100
            
            # Calculate the expected rank multiplier using the specific distribution
            perc = scenario["dist"](z)
            mult = 0.1 + 0.8 * perc
            
            # Net PnL calculation
            pnl = (research * scale_val * mult) - 50000
            
            if pnl > best_pnl:
                best_pnl = pnl
                best_z = z
                best_x = x
                best_y = y
                
        # Output results
        print(f"--- {scenario['name']} ---")
        print(f"Optimal Speed:    {best_z:.2f}%")
        print(f"Optimal Research: {best_x:.2f}%")
        print(f"Optimal Scale:    {best_y:.2f}%")
        print(f"Expected Net PnL: {best_pnl:,.0f} XIRECs\n")

# Execute the simulation
solve_optimal_pnl_all(loc=40.0, scale=8.0)