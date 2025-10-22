import numpy as np
from scipy.stats import chi2
import time
import os
import pandas as pd
from spline_chi2_cdf import chi2_cdf

# Define the tolerance for verifying the results
MAX_ERROR_TOLERANCE = 1e-8

def run_benchmark():
    """
    Performs a rigorous benchmark of the spline chi-squared CDF vs. SciPy's chi-squared CDF.
    """
    print("--- Verifying Chi-Squared Distribution Spline Accuracy and Performance ---")
    num_test_points = 10_000_000
    num_trials = 30
    
    all_trial_results = []

    # Use a realistic range for x and df in testing
    df_range = (1, 1_000_000)
    x_range = (0, 1_000_000)

    for trial_num in range(1, num_trials + 1):
        print(f"\n--- Starting Trial {trial_num}/{num_trials} ---")
        
        # Generate random values for this trial
        test_x_vals = np.random.uniform(x_range[0], x_range[1], num_test_points)
        test_df_vals = np.random.randint(df_range[0], df_range[1] + 1, num_test_points)
        
        print(f"Generated {len(test_x_vals):,} random values for this trial...")
        
        # --- SciPy Benchmark ---
        start_scipy = time.time()
        scipy_cdf_vals = np.array([chi2.cdf(x, df) for x, df in zip(test_x_vals, test_df_vals)])
        end_scipy = time.time()
        scipy_time = end_scipy - start_scipy
        print(f"SciPy Time:   {scipy_time:.4f} seconds")

        # --- Spline Benchmark ---
        start_spline = time.time()
        spline_cdf_vals = np.array([chi2_cdf(x, df) for x, df in zip(test_x_vals, test_df_vals)])
        end_spline = time.time()
        spline_time = end_spline - start_spline
        print(f"Spline Time:  {spline_time:.4f} seconds")
        
        # --- Error Calculation ---
        final_error = np.abs(scipy_cdf_vals - spline_cdf_vals)
        min_abs_error = np.min(final_error)
        mean_abs_error = np.mean(final_error)
        max_abs_error = np.max(final_error)
        
        print(f"Min, Mean, & Max Absolute Error for Trial: {min_abs_error:.2e}\t{mean_abs_error:.2e}\t{max_abs_error:.2e}")
        
        # Store results for this trial
        trial_data = {
            'trial': trial_num,
            'scipy_time_sec': scipy_time,
            'spline_time_sec': spline_time,
            'min_abs_error': min_abs_error,
            'mean_abs_error': mean_abs_error,
            'max_abs_error': max_abs_error
        }
        all_trial_results.append(trial_data)

    print("\n--- All trials complete. Saving results to Excel. ---")
    
    # --- Save Results to Excel ---
    results_df = pd.DataFrame(all_trial_results)
    
    results_path = 'splines/chi2/results_spline_chi2_cdf.xlsx'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    results_df.to_excel(results_path, index=False, engine='openpyxl')
    
    print(f"Results successfully saved to: {results_path}")

    # --- Final Summary ---
    print("\n--- Overall Summary (Averages Across 30 Trials) ---")
    avg_scipy_time = results_df['scipy_time_sec'].mean()
    avg_spline_time = results_df['spline_time_sec'].mean()
    overall_min_error = results_df['min_abs_error'].min()
    overall_mean_error = results_df['mean_abs_error'].mean()
    overall_max_error = results_df['max_abs_error'].max()
    
    print(f"Average SciPy Time:   {avg_scipy_time:.4f} seconds")
    print(f"Average Spline Time:  {avg_spline_time:.4f} seconds")
    print(f"Overall Min Error Across All Trials: {overall_min_error:.3e}")
    print(f"Overall Mean Error Across All Trials: {overall_mean_error:.3e}")
    print(f"Overall Max Error Across All Trials: {overall_max_error:.3e}")
    
    if overall_max_error <= MAX_ERROR_TOLERANCE:
        print("SUCCESS: Final model accuracy is within the tolerance for all trials.")
    else:
        print("FAILURE: Final model accuracy exceeded the tolerance in at least one trial.")
    
    speedup = avg_scipy_time / avg_spline_time
    print(f"The spline approach is {speedup:.2f}x faster on average.\n")

if __name__ == '__main__':
    run_benchmark()