import pickle
import gzip
from scipy.interpolate import splev
from scipy.stats import norm
import os

# --- Load the pre-trained spline models and metadata upon module import ---
MODELS_PATH = 'spline_models/models_t_cdf.pkl.gz'
try:
    with gzip.open(MODELS_PATH, 'rb') as f:
        _model_data = pickle.load(f)
    _models = _model_data['models']
    _df_max = _model_data['df_max']
    _essential_dfs = sorted(_model_data['essential_dfs'])
    _application_tolerance = _model_data['application_tolerance']
except FileNotFoundError:
    raise FileNotFoundError(f"Models file not found at {MODELS_PATH}.")

def t_cdf(x, df):
    """
    Calculates the t-distribution CDF for a given x and df.

    Uses a pre-computed collection of spline models for df <= df_max and
    the standard normal CDF for df > df_max. It also leverages the point
    symmetry of the t-distribution.

    Args:
        x (float): The t-value at which to evaluate the CDF.
        df (float or int): The degrees of freedom.

    Returns:
        float: The cumulative probability.
    """
    df = int(round(df))

    if df > _df_max:
        return norm.cdf(x)
    
    low, high = 0, len(_essential_dfs) - 1
    model_df = _essential_dfs[0]
    while low <= high:
        mid_idx = (low + high) // 2
        mid_val = _essential_dfs[mid_idx]
        if mid_val <= df:
            model_df = mid_val
            low = mid_idx + 1
        else:
            high = mid_idx - 1
    
    model_to_use = _models[model_df]
    tck = model_to_use['tck']
    lower_bound = model_to_use['lower_bound']
    boundary_val = model_to_use['boundary_val']

    # --- Evaluate using the selected spline, handling boundaries and symmetry ---
    if x > 0:
        neg_x = -x
        if neg_x <= lower_bound:
            # For the far-right tail, return 1.0 minus the tiny boundary value
            #return 1.0 - boundary_val
            return 1.0
        # Use splev for values within the trained range.
        return 1.0 - float(splev(neg_x, tck))
    else: # x <= 0
        if x <= lower_bound:
            # For the far-left tail, return the tiny boundary value directly
            #return boundary_val
            return 0.0
        # Use splev for values within the trained range.
        return float(splev(x, tck))
