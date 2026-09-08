import numpy as np

# call the needed functions from the module
# note: get_running_couplings etc. are excluded since they are not needed for the check itself
from thermal_np import scalar_masses_AE

def check_mass_positivity(t_min=-2.0, t_max=0.0, num_t=50,
                          rho_min=0.0, rho_max=2.0, num_rho=100):
    """
    Function that checks whether 1 + mH^2 (Higgs mass) stays non-negative over the given range of t.

    Parameters:
    -----------
    t_min, t_max : float
        search range in t (default: -2.0 to 0.0)
    num_t : int
        number of t subdivisions (increase for higher precision)
    rho_min, rho_max, num_rho : float, float, int
        rho grid settings

    Returns:
    --------
    dict
        - 'is_all_positive': True if 1 + mH^2 >= 0 at every grid point and t
        - 'global_min': the minimum of 1 + mH^2 over the whole search range
        - 'negative_points': list of t values where it went negative and the corresponding minimum
    """

    # build the rho grid (t-independent, so build once outside the loop and reuse)
    rho_vals = np.linspace(rho_min, rho_max, num_rho).reshape(1, -1)

    # array of t
    t_vals = np.linspace(t_min, t_max, num_t)

    global_min = np.inf
    negative_points = []

    for t in t_vals:
        T_VAL = np.full_like(rho_vals, t)

        # compute the mass eigenvalues
        (mG2, mH2), _ = scalar_masses_AE(rho_vals, T_VAL)

        # compute the minimum of 1 + mH2
        current_min = np.min(1.0 + mH2)

        if current_min < global_min:
            global_min = current_min

        # if it went negative, record the t and the corresponding minimum
        if current_min < 0:
            negative_points.append({'t': t, 'min_val': current_min})

    is_all_positive = (global_min >= 0)

    return {
        'is_all_positive': is_all_positive,
        'global_min': global_min,
        'negative_points': negative_points
    }


# ============================================================
# test code for when run directly, not when imported as a module
# ============================================================
if __name__ == "__main__":
    print("Starting positivity check for 1 + m^2 in t = [-2.0, 0.0]...")

    # run the function (adjust num_t, num_rho to change the subdivisions)
    results = check_mass_positivity(
        t_min=-2.0, t_max=0.0, num_t=100,
        rho_min=0.0, rho_max=2.0, num_rho=100,
    )

    print("-" * 50)
    if results['is_all_positive']:
        print(f"OK. 1 + m^2 >= 0 for all t.")
        print(f"   (global minimum over the search range: {results['global_min']:.6f})")
    else:
        print(f"WARNING: found a region where 1 + m^2 becomes negative.")
        print(f"   (global minimum over the search range: {results['global_min']:.6f})")
        print("[t values where it went negative and their minima]:")
        for pt in results['negative_points']:
            print(f"  t = {pt['t']:>7.4f}  |  min(1+m^2) = {pt['min_val']:.6f}")
