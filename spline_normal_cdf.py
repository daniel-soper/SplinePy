import pickle
import gzip
from scipy.interpolate import splev

# --- Load the pre-trained spline model upon module import ---

MODEL_PATH = 'spline_models/model_standard_normal_cdf.pkl.gz'

try:
    with gzip.open(MODEL_PATH, 'rb') as f:
        _model_data = pickle.load(f)
    _tck = _model_data['tck']
    _a = _model_data['a']
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")

def normal_cdf(x, mean=0.0, std_dev=1.0):
    """
    Calculates the CDF for any normal distribution using a pre-trained spline.

    This function handles a single value by first standardizing it and then
    using the pre-computed spline. It also leverages the point symmetry
    of the normal distribution for efficiency.

    Args:
        x (float): The value at which to evaluate the CDF.
        mean (float): The mean of the normal distribution.
        std_dev (float): The standard deviation of the normal distribution.

    Returns:
        float: The cumulative probability.
    """
    # 1. Standardize the input value to a z-score
    # Handle case where std_dev is zero to avoid division error
    if std_dev == 0:
        return 1.0 if x >= mean else 0.0
    z = (x - mean) / std_dev

    # 2. Evaluate the CDF based on the z-score
    # Condition 1: Value is to the left of the trained interval
    if z <= _a:
        return 0.0
    # Condition 2: Value is within the trained interval (left tail)
    elif z <= 0:
        return splev(z, _tck)
    # Condition 3: Value is in the right tail (use symmetry: 1 - CDF(-z))
    else:
        neg_z = -z
        # Handle cases where -z falls outside the trained interval
        if neg_z <= _a:
            return 1.0
        else:
            return 1.0 - splev(neg_z, _tck)
