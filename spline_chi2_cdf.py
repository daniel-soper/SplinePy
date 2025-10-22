import pickle
import gzip
from scipy.interpolate import splev
from spline_normal_cdf import normal_cdf

# --- Load the pre-trained spline models and metadata upon module import ---
MODELS_PATH = 'spline_models/models_chi2_cdf.pkl.gz'
try:
    with gzip.open(MODELS_PATH, 'rb') as f:
        _model_data = pickle.load(f)
    _models = _model_data['models']
    _df_max = _model_data['df_max']
    _essential_dfs = sorted(_model_data['essential_dfs'])
    _application_tolerance = _model_data['application_tolerance']
except FileNotFoundError:
    raise FileNotFoundError(f"Models file not found at {MODELS_PATH}.")

#a constant for one-third
ONE_THIRD = 1.0 / 3.0

def _wilson_hilferty_transform(x: float, df: int) -> float:
    """Applies the Wilson-Hilferty transformation."""
    x_safe = max(x, 1e-12)
    term1 = (x_safe / df) ** ONE_THIRD
    term2 = 1.0 - 2.0 / (9.0 * df)
    term3 = (2.0 / (9.0 * df)) ** 0.5
    return (term1 - term2) / term3

def _wilson_hilferty_approx(x: float, df: int) -> float:
    """Approximates the chi-squared CDF using the Wilson-Hilferty transform."""
    z = _wilson_hilferty_transform(x, df)
    return normal_cdf(z)

def chi2_cdf(x: float, df: int) -> float:
    """
    Approximates the chi-squared CDF for a single given x and df value.
    
    Args:
        x (float): The value at which to evaluate the CDF.
        df (int): The degrees of freedom.

    Returns:
        float: The approximated CDF value.
    """
    #calculate the base approximation.
    wh_approx = _wilson_hilferty_approx(x, df)
    
    #if df is outside the range of the trained spline models, fall back to Wilson-Hilferty.
    if df > _df_max: return wh_approx

    #find the appropriate spline model to use for the given df.
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
    model = _models[model_df]
    
    #get model boundaries and boundary residual values.
    lower_bound, upper_bound = model['bounds']
    lower_val, upper_val = model['boundary_residuals']
        
    #determine the residual based on x's position relative to the spline bounds.
    spline_residual: float
    if x < lower_bound:
        spline_residual = lower_val
    elif x > upper_bound:
        spline_residual = upper_val
    else:
        spline_residual = float(splev(x, model['tck']))
    
    #combine the approximation and the residual.
    result = wh_approx + spline_residual
    
    #clamp the result to the valid probability range [0.0, 1.0].
    return max(0.0, min(1.0, result))
