import matplotlib.pyplot as plt


def plot_trajectory(leaspy_model, timepoints, reconstruction, observations=None, *,
                    xlabel='Reparametrized age', ylabel='Normalized feature value'):
    if observations is not None:
        ages = timepoints

    plt.figure(figsize=(10, 5))
    plt.ylim(0, 1)
    colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']
    for c, name, val in zip(colors, leaspy_model.model.features[0:4], reconstruction.T[0:4]):
        plt.plot(timepoints, val, label=name, c=c, linewidth=3)
        if observations is not None:
            # print("obs is not None", len(observations[name]))
            plt.plot(ages, observations[name], c=c, marker='o', markersize=12,
                     linewidth=1, linestyle=':')
    # plt.xlim(0,80)
    print(timepoints)
    plt.xlim(min(timepoints), max(timepoints))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    plt.show()
