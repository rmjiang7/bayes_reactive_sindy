import numpy as _np

import readdy_learn.analyze.basis as _basis

INITIAL_STATES = _np.array([
    [20, 20, 20, 20, 20],
])

RATES = _np.zeros(260)
RATES[:8] = [0.5 , 1.  , 0.15, 1.  , 0.5 , 0.5 , 1.5 , 0.15]

SPECIES_NAMES = ['g', 'pt', 'g.pt', 'r', 'p']
TIMESTEP = 0.05


def bfc():
    result = _basis.BasisFunctionConfiguration(len(SPECIES_NAMES))

    result.add_fusion(0,1,2)
    result.add_fission(2,0,1)
    result.add_fission(0,0,3)
    result.add_fission(3,3,4)
    result.add_fusion(4,4,1)
    result.add_fission(1,4,4)
    result.add_decay(3)
    result.add_decay(4)
    result.add_decay(0)
    result.add_fission(0,0,0)
    result.add_conversion(0,1)
    result.add_fission(0,0,1)
    result.add_fusion(0,0,1)
    result.add_fission(1,0,0)
    result.add_conversion(0,2)
    result.add_fission(0,0,2)
    result.add_fusion(0,0,2)
    result.add_fission(2,0,0)
    result.add_conversion(0,3)
    result.add_fusion(0,0,3)
    result.add_fission(3,0,0)
    result.add_conversion(0,4)
    result.add_fission(0,0,4)
    result.add_fusion(0,0,4)
    result.add_fission(4,0,0)
    result.add_fission(0,1,2)
    result.add_fission(0,1,3)
    result.add_fission(0,1,4)
    result.add_fission(0,2,3)
    result.add_fission(0,2,4)
    result.add_fission(0,3,4)
    result.add_decay(1)
    result.add_conversion(1,0)
    result.add_fission(1,1,0)
    result.add_fusion(1,1,0)
    result.add_fission(0,1,1)
    result.add_fission(1,1,1)
    result.add_conversion(1,2)
    result.add_fission(1,1,2)
    result.add_fusion(1,1,2)
    result.add_fission(2,1,1)
    result.add_conversion(1,3)
    result.add_fission(1,1,3)
    result.add_fusion(1,1,3)
    result.add_fission(3,1,1)
    result.add_conversion(1,4)
    result.add_fission(1,1,4)
    result.add_fusion(1,1,4)
    result.add_fission(4,1,1)
    result.add_fission(1,0,2)
    result.add_fission(1,0,3)
    result.add_fission(1,0,4)
    result.add_fission(1,2,3)
    result.add_fission(1,2,4)
    result.add_fission(1,3,4)
    result.add_decay(2)
    result.add_conversion(2,0)
    result.add_fission(2,2,0)
    result.add_fusion(2,2,0)
    result.add_fission(0,2,2)
    result.add_conversion(2,1)
    result.add_fission(2,2,1)
    result.add_fusion(2,2,1)
    result.add_fission(1,2,2)
    result.add_fission(2,2,2)
    result.add_conversion(2,3)
    result.add_fission(2,2,3)
    result.add_fusion(2,2,3)
    result.add_fission(3,2,2)
    result.add_conversion(2,4)
    result.add_fission(2,2,4)
    result.add_fusion(2,2,4)
    result.add_fission(4,2,2)
    result.add_fission(2,0,3)
    result.add_fission(2,0,4)
    result.add_fission(2,1,3)
    result.add_fission(2,1,4)
    result.add_fission(2,3,4)
    result.add_conversion(3,0)
    result.add_fission(3,3,0)
    result.add_fusion(3,3,0)
    result.add_fission(0,3,3)
    result.add_conversion(3,1)
    result.add_fission(3,3,1)
    result.add_fusion(3,3,1)
    result.add_fission(1,3,3)
    result.add_conversion(3,2)
    result.add_fission(3,3,2)
    result.add_fusion(3,3,2)
    result.add_fission(2,3,3)
    result.add_fission(3,3,3)
    result.add_conversion(3,4)
    result.add_fusion(3,3,4)
    result.add_fission(4,3,3)
    result.add_fission(3,0,1)
    result.add_fission(3,0,2)
    result.add_fission(3,0,4)
    result.add_fission(3,1,2)
    result.add_fission(3,1,4)
    result.add_fission(3,2,4)
    result.add_conversion(4,0)
    result.add_fission(4,4,0)
    result.add_fusion(4,4,0)
    result.add_fission(0,4,4)
    result.add_conversion(4,1)
    result.add_fission(4,4,1)
    result.add_conversion(4,2)
    result.add_fission(4,4,2)
    result.add_fusion(4,4,2)
    result.add_fission(2,4,4)
    result.add_conversion(4,3)
    result.add_fission(4,4,3)
    result.add_fusion(4,4,3)
    result.add_fission(3,4,4)
    result.add_fission(4,4,4)
    result.add_fission(4,0,1)
    result.add_fission(4,0,2)
    result.add_fission(4,0,3)
    result.add_fission(4,1,2)
    result.add_fission(4,1,3)
    result.add_fission(4,2,3)
    result.add_double_conversion([0,1],[0,0])
    result.add_double_conversion([0,1],[1,1])
    result.add_fusion(0,1,3)
    result.add_fusion(0,1,4)
    result.add_double_conversion([0,1],[0,1])
    result.add_double_conversion([0,1],[0,1])
    result.add_double_conversion([0,1],[0,1])
    result.add_double_conversion([0,1],[0,1])
    result.add_double_conversion([0,1],[0,1])
    result.add_double_conversion([0,1],[0,1])
    result.add_double_conversion([0,1],[0,1])
    result.add_double_conversion([0,1],[0,1])
    result.add_double_conversion([0,1],[0,1])
    result.add_double_conversion([0,2],[0,0])
    result.add_fusion(0,2,1)
    result.add_double_conversion([0,2],[2,2])
    result.add_fusion(0,2,3)
    result.add_fusion(0,2,4)
    result.add_double_conversion([0,2],[0,2])
    result.add_double_conversion([0,2],[0,2])
    result.add_double_conversion([0,2],[0,2])
    result.add_double_conversion([0,2],[0,2])
    result.add_double_conversion([0,2],[0,2])
    result.add_double_conversion([0,2],[0,2])
    result.add_double_conversion([0,2],[0,2])
    result.add_double_conversion([0,2],[0,2])
    result.add_double_conversion([0,2],[0,2])
    result.add_double_conversion([0,3],[0,0])
    result.add_fusion(0,3,1)
    result.add_fusion(0,3,2)
    result.add_double_conversion([0,3],[3,3])
    result.add_fusion(0,3,4)
    result.add_double_conversion([0,3],[0,3])
    result.add_double_conversion([0,3],[0,3])
    result.add_double_conversion([0,3],[0,3])
    result.add_double_conversion([0,3],[0,3])
    result.add_double_conversion([0,3],[0,3])
    result.add_double_conversion([0,3],[0,3])
    result.add_double_conversion([0,3],[0,3])
    result.add_double_conversion([0,3],[0,3])
    result.add_double_conversion([0,3],[0,3])
    result.add_double_conversion([0,4],[0,0])
    result.add_fusion(0,4,1)
    result.add_fusion(0,4,2)
    result.add_fusion(0,4,3)
    result.add_double_conversion([0,4],[4,4])
    result.add_double_conversion([0,4],[0,4])
    result.add_double_conversion([0,4],[0,4])
    result.add_double_conversion([0,4],[0,4])
    result.add_double_conversion([0,4],[0,4])
    result.add_double_conversion([0,4],[0,4])
    result.add_double_conversion([0,4],[0,4])
    result.add_double_conversion([0,4],[0,4])
    result.add_double_conversion([0,4],[0,4])
    result.add_double_conversion([0,4],[0,4])
    result.add_fusion(1,2,0)
    result.add_double_conversion([1,2],[1,1])
    result.add_double_conversion([1,2],[2,2])
    result.add_fusion(1,2,3)
    result.add_fusion(1,2,4)
    result.add_double_conversion([1,2],[1,2])
    result.add_double_conversion([1,2],[1,2])
    result.add_double_conversion([1,2],[1,2])
    result.add_double_conversion([1,2],[1,2])
    result.add_double_conversion([1,2],[1,2])
    result.add_double_conversion([1,2],[1,2])
    result.add_double_conversion([1,2],[1,2])
    result.add_double_conversion([1,2],[1,2])
    result.add_double_conversion([1,2],[1,2])
    result.add_fusion(1,3,0)
    result.add_double_conversion([1,3],[1,1])
    result.add_fusion(1,3,2)
    result.add_double_conversion([1,3],[3,3])
    result.add_fusion(1,3,4)
    result.add_double_conversion([1,3],[1,3])
    result.add_double_conversion([1,3],[1,3])
    result.add_double_conversion([1,3],[1,3])
    result.add_double_conversion([1,3],[1,3])
    result.add_double_conversion([1,3],[1,3])
    result.add_double_conversion([1,3],[1,3])
    result.add_double_conversion([1,3],[1,3])
    result.add_double_conversion([1,3],[1,3])
    result.add_double_conversion([1,3],[1,3])
    result.add_fusion(1,4,0)
    result.add_double_conversion([1,4],[1,1])
    result.add_fusion(1,4,2)
    result.add_fusion(1,4,3)
    result.add_double_conversion([1,4],[4,4])
    result.add_double_conversion([1,4],[1,4])
    result.add_double_conversion([1,4],[1,4])
    result.add_double_conversion([1,4],[1,4])
    result.add_double_conversion([1,4],[1,4])
    result.add_double_conversion([1,4],[1,4])
    result.add_double_conversion([1,4],[1,4])
    result.add_double_conversion([1,4],[1,4])
    result.add_double_conversion([1,4],[1,4])
    result.add_double_conversion([1,4],[1,4])
    result.add_fusion(2,3,0)
    result.add_fusion(2,3,1)
    result.add_double_conversion([2,3],[2,2])
    result.add_double_conversion([2,3],[3,3])
    result.add_fusion(2,3,4)
    result.add_double_conversion([2,3],[2,3])
    result.add_double_conversion([2,3],[2,3])
    result.add_double_conversion([2,3],[2,3])
    result.add_double_conversion([2,3],[2,3])
    result.add_double_conversion([2,3],[2,3])
    result.add_double_conversion([2,3],[2,3])
    result.add_double_conversion([2,3],[2,3])
    result.add_double_conversion([2,3],[2,3])
    result.add_double_conversion([2,3],[2,3])
    result.add_fusion(2,4,0)
    result.add_fusion(2,4,1)
    result.add_double_conversion([2,4],[2,2])
    result.add_fusion(2,4,3)
    result.add_double_conversion([2,4],[4,4])
    result.add_double_conversion([2,4],[2,4])
    result.add_double_conversion([2,4],[2,4])
    result.add_double_conversion([2,4],[2,4])
    result.add_double_conversion([2,4],[2,4])
    result.add_double_conversion([2,4],[2,4])
    result.add_double_conversion([2,4],[2,4])
    result.add_double_conversion([2,4],[2,4])
    result.add_double_conversion([2,4],[2,4])
    result.add_double_conversion([2,4],[2,4])
    result.add_fusion(3,4,0)
    result.add_fusion(3,4,1)
    result.add_fusion(3,4,2)
    result.add_double_conversion([3,4],[3,3])
    result.add_double_conversion([3,4],[4,4])
    result.add_double_conversion([3,4],[3,4])
    result.add_double_conversion([3,4],[3,4])
    result.add_double_conversion([3,4],[3,4])
    result.add_double_conversion([3,4],[3,4])
    result.add_double_conversion([3,4],[3,4])
    result.add_double_conversion([3,4],[3,4])
    result.add_double_conversion([3,4],[3,4])
    result.add_double_conversion([3,4],[3,4])
    result.add_double_conversion([3,4],[3,4])

    return result


def derivative(times, counts):
    dcounts_dt = _np.empty_like(counts)
    for sp in range(len(SPECIES_NAMES)):
        x = counts[:, sp]

        indices = 1 + _np.where(x[:-1] != x[1:])[0]
        indices = _np.insert(indices, 0, 0)

        interpolated = _np.interp(times, times[indices], x[indices])
        interpolated = _np.gradient(interpolated) / TIMESTEP

        dcounts_dt[:, sp] = interpolated
    return dcounts_dt


def generate_lma(initial_state: int, target_time: float):
    import readdy_learn.analyze.generate as generate
    _, counts = generate.generate_continuous_counts(RATES, INITIAL_STATES[initial_state],
                                                    bfc(), TIMESTEP, int(target_time // TIMESTEP),
                                                    noise_variance=0, n_realizations=1)
    times = _np.linspace(0, counts.shape[0] * TIMESTEP, endpoint=False, num=counts.shape[0])

    for sp in range(len(SPECIES_NAMES)):
        x = counts[:, sp]
        indices = 1 + _np.where(x[:-1] != x[1:])[0]
        indices = _np.insert(indices, 0, 0)
        interpolated = _np.interp(times, times[indices], x[indices])
        counts[:, sp] = interpolated

    dcounts_dt = derivative(times, counts)

    return times, counts, dcounts_dt


def solve(counts, dcounts_dt, alpha, l1_ratio):
    import readdy_learn.analyze.estimator as rlas
    import readdy_learn.analyze.tools as tools

    tolerances_to_try = _np.logspace(-16, -1, num=16)

    if isinstance(counts, (list, tuple)):
        assert isinstance(dcounts_dt, (list, tuple))
        counts = _np.concatenate(counts, axis=0).squeeze()
        dcounts_dt = _np.concatenate(dcounts_dt, axis=0).squeeze()
    else:
        assert isinstance(counts, _np.ndarray) and isinstance(dcounts_dt, _np.ndarray)

    traj = tools.Trajectory(counts, time_step=TIMESTEP)
    traj.dcounts_dt = dcounts_dt

    estimator = None
    for tol in tolerances_to_try:
        print("Trying tolerance {}".format(tol))
        estimator = rlas.ReaDDyElasticNetEstimator([traj], bfc(), alpha=alpha, l1_ratio=l1_ratio,
                                                   maxiter=30000, method='SLSQP', verbose=True, approx_jac=False,
                                                   options={'ftol': tol}, rescale=False,
                                                   init_xi=_np.zeros_like(RATES),
                                                   constrained=True)

        estimator.fit(None)
        if estimator.success_:
            return estimator.coefficients_
    if estimator is not None:
        raise ValueError('*_*: {}, {}'.format(estimator.result_.status, estimator.result_.message))
    else:
        raise ValueError('-_-')


def solve_grid(counts, dcounts_dt, alphas, l1_ratios, njobs=1):
    import itertools
    import pathos.multiprocessing as multiprocessing
    from readdy_learn.analyze.progress import Progress

    alphas = _np.atleast_1d(_np.array(alphas).squeeze())
    lambdas = _np.atleast_1d(_np.array(l1_ratios).squeeze())
    params = itertools.product(alphas, lambdas)
    params = [(counts, dcounts_dt, p[0], p[1]) for p in params]

    progress = Progress(len(params), label="validation", nstages=1)

    def worker(args):
        c, dc, a, l = args
        return a, l, solve(c, dc, a, l)

    result = []
    with multiprocessing.Pool(processes=njobs) as p:
        for idx, res in enumerate(p.imap_unordered(worker, params, 1)):
            result.append(res)
            progress.increase()
    progress.finish()

    return result


def cv(counts, dcounts_dt, alphas=(1.,), l1_ratios=(1.,), n_splits=5, njobs=1):
    import readdy_learn.analyze.tools as tools
    import readdy_learn.analyze.cross_validation as cross_validation

    traj = tools.Trajectory(counts, time_step=TIMESTEP)
    traj.dcounts_dt = dcounts_dt
    cv = cross_validation.CrossValidation([traj], bfc())
    cv.splitter = 'kfold'
    cv.n_splits = n_splits
    cv.njobs = njobs
    cv.show_progress = True
    cv_result = cv.cross_validate(alphas, l1_ratios, realizations=1)
    result = {"cv_result": cv_result}
    return result

