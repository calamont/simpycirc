from scipy.optimize import minimize


def cost(params, circuit, data):
    sim = np.abs(circuit(10 ** params))
    return np.square(np.subtract(np.log10(data), np.log10(sim))).mean()


def fit_components(**kwargs):

    fit = minimize(cost, [10], args=(circuit, data), **kwargs)

    return 10 ** fit.x

