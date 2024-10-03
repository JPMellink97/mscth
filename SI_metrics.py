from IPython.display import display, Math, Latex
import numpy as np
import matplotlib.pyplot as plt

def plot_coeff_grid(data, states=2, labels=None, true_coeff=None, save=False, save_name="Coefficients", precision=1, threshold=1e-3, linewidth=3, print_th_only=False):
    
    def closest_square_grid(n):
        # Find the closest integer greater than or equal to the square root of n
        sq_root = int(np.ceil(np.sqrt(n)))
        
        # Calculate rows and columns based on the closest square
        rows = sq_root
        cols = sq_root if sq_root * (sq_root - 1) < n else sq_root - 1
        
        return rows, cols

    if isinstance(data, list):
        data = np.array(data)
        
    n, m = data.shape
    if n > m:
        data = data.T
        n, m = data.shape

    nr_theta = n // states
    labels = [f"f{idx}(x, u)" for idx in range(nr_theta)] if labels is None else labels

    for i in range(states):
        data_i = data[i * nr_theta:(i + 1) * nr_theta, :]
        n, m = data_i.shape
        valid_plots = 0
        
        for j in range(n):
            estimated_coeff = np.round(np.mean(data_i[j, -1]), precision)
            if np.abs(estimated_coeff) > threshold:  # Only count if above threshold
                valid_plots += 1

        rows, cols = closest_square_grid(valid_plots)
        
        if valid_plots == 0:
            continue 
        
        # figure
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        axes_flat = np.array(axes).flatten() if valid_plots > 1 else [axes]
        
        plotted_count = 0

        for j in range(n):
            estimated_coeff = np.round(np.mean(data_i[j, -1]), precision)
            if np.abs(estimated_coeff) > threshold:  # plot if above threshold
                alpha = 1.0
                axes_flat[plotted_count].plot(data_i[j, :], label=f"Estimated = {estimated_coeff:.{precision}f}", alpha=alpha, linewidth=linewidth)
                axes_flat[plotted_count].grid()
                axes_flat[plotted_count].set_title(labels[j])
                axes_flat[plotted_count].set_xlim([0, m])

                y_range = np.max(data_i[j, :]) - np.min(data_i[j, :])
                y_last = data_i[j, -1]
                axes_flat[plotted_count].set_ylim([y_last - y_range / 2, y_last + y_range / 2])

                if true_coeff is not None:
                    axes_flat[plotted_count].plot(true_coeff[i, j] * np.ones(data[j, :].shape), label=f"True = {true_coeff[i, j]}")

                axes_flat[plotted_count].legend(loc='lower right')
                plotted_count += 1
        
        # hideextra subplots
        for j in range(plotted_count, rows * cols):
            axes_flat[j].axis('off')

        plt.tight_layout()
        if save:
            plt.savefig(f"{save_name}_x{i}.png", dpi=300, bbox_inches='tight')

        plt.show()

    return



def display_equation(coeff, labels, threshold=0.1, precision=2, verbose=True, convert_latex=False):
    n = coeff.shape

    equation = []

    if len(labels) is not n[0]:
        coeff = np.reshape(coeff, (n[0]//len(labels), len(labels)))

    for idx_i, row in enumerate(coeff):
        for idx_j, val in enumerate(row):
            coeff[idx_i, idx_j] = val if np.abs(val)>threshold else 0.0 
        
        nz = np.nonzero(row)
        
        eq_i = "x{i}[k+1] = ".format(i=idx_i)

        for idx, val in enumerate(nz[0]):
            cf = np.round(row[val], precision)
            c = str(cf) if (cf<0 or idx==0) else "+"+str(cf)
            f = str(labels[val])
            eq_i += "{c}*{f}".format(c=c, f=f)

        if verbose:
            print(eq_i)

        equation.append(eq_i)

        if convert_latex:
            pass

    return equation