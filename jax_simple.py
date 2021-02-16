from functools import partial
import time

import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental import optimizers
import matplotlib.pyplot as plt
import numpy as np

from pySDC.implementations.collocation_classes.gauss_radau_right import \
    CollGaussRadau_Right

jax.config.update('jax_enable_x64', True)

# Is buggy so don't use until fixed
use_jax_control_flow = False
# For full step: do not stop early, always iterate
# `max_episode_length` times
use_for_max_episode_length = False
# Whether to JIT loss and other functions that may not work
use_unstable_jax_jit = True
use_simple_loss = False


def get_initial(M, dt, rng):
    coll = CollGaussRadau_Right(M, 0, 1)
    Q = coll.Qmat[1:, 1:]
    u0 = jnp.ones(coll.num_nodes, dtype=jnp.complex128)

    lam = (1 * rng.uniform(low=-100.0, high=0.0)
           + 0j * rng.uniform(low=0.0, high=10.0))
    C = jnp.eye(coll.num_nodes) - lam * dt * Q
    u = jnp.ones(coll.num_nodes, dtype=jnp.complex128)

    initial_residual = u0 - C @ u
    return (
        coll,
        Q,
        u0,
        lam,
        C,
        u,
        initial_residual,
    )


# @partial(jax.jit, static_argnums=(2,))
@jax.jit
def _fill_diagonal(matrix, diagonal, offset=0):
    """Fill the main diagonal of `matrix` with the values in `diagonal`.
    A JAX-compatible variant of `np.fill_diagonal`.
    `offset` is an integer to get one of other diagonals the same offset
    from the main diagonal.
    If it is positive, fill one of the upper triangular matrices with
    the offset.
    If it is negative, fill one of the lower triangular matrices with the
    negated offset.
    """
    for i in range(len(diagonal)):
        matrix = matrix.at[i, i].set(diagonal[i])
    # if offset < 0:
    #     for i in range(len(diagonal) - offset):
    #         # Offset is negative, so subtract to add
    #         matrix = matrix.at[i - offset, i].set(diagonal[i])
    # else:
    #     for i in range(len(diagonal) - offset):
    #         matrix = matrix.at[i, i + offset].set(diagonal[i])
    return matrix


# @partial(jax.jit, static_argnums=(2,))
@jax.jit
def _get_prec(action, Q, M):
    Qdmat = jnp.zeros_like(Q)
    Qdmat = _fill_diagonal(Qdmat, action)
    return Qdmat


@partial(jax.jit, static_argnums=(0, 2))
def _get_pinv(coll_num_nodes, lam, dt, Qdmat):
    return jnp.linalg.inv(jnp.eye(coll_num_nodes) - lam * dt * Qdmat)


@jax.jit
def _update_u(u, Pinv, u0, C):
    return u + Pinv @ _residual(u, u0, C)


@jax.jit
def _residual(u, u0, C):
    return u0 - C @ u


@jax.jit
def _norm(res):
    return jnp.linalg.norm(res, jnp.inf)


@partial(jax.jit, static_argnums=(3, 4, 6, 9, 10))
def step(action, u, Q, M, coll_num_nodes, lam, dt, u0, C, _restol,
         _max_episode_length):
    Qdmat = _get_prec(action, Q, M)
    Pinv = _get_pinv(coll_num_nodes, lam, dt, Qdmat)
    # norm_res_old = _norm(init_resid)

    u = _update_u(u, Pinv, u0, C)
    residual = _residual(u, u0, C)
    norm_res = _norm(residual)
    return norm_res, u, residual, 1


@partial(jax.jit, static_argnums=(3, 4, 6, 9, 10))
def full_step(action, u, Q, M, coll_num_nodes, lam, dt, u0, C, restol,
              max_episode_length):
    Qdmat = _get_prec(action, Q, M)
    Pinv = _get_pinv(coll_num_nodes, lam, dt, Qdmat)
    residual = _residual(u, u0, C)
    norm_res = _norm(residual)
    # norm_res_old = _norm(init_resid)

    if use_for_max_episode_length:
        for niters in range(max_episode_length):
            u = _update_u(u, Pinv, u0, C)
            residual = _residual(u, u0, C)
            norm_res = _norm(residual) * niters
        return norm_res, u, residual, max_episode_length

    if use_jax_control_flow:
        init_val = (u, residual, norm_res, 0)

        def cond_fun(args):
            (_, _, norm_res, niters) = args
            # return jax.lax.cond(
            #     (norm_res > restol
            #      and niters < max_episode_length
            #      and not jnp.isnan(norm_res)),
            #     lambda _: True,
            #     lambda _: False,
            #     None,
            # )
            # return norm_res > 1E-10 and niters < max_episode_length
            return (norm_res > restol
                    and niters < max_episode_length
                    and not jnp.isnan(norm_res))

        def body_fun(args):
            (u, residual, norm_res, niters) = args
            u = _update_u(u, Pinv, u0, C)
            residual = _residual(u, u0, C)
            norm_res = _norm(residual)
            return (u, residual, norm_res, niters + 1)

        (u, residual, norm_res, niters) = jax.lax.while_loop(
            cond_fun, body_fun, init_val)
    else:
        niters = 0

        while (norm_res > restol
               and niters < max_episode_length
               and not jnp.isnan(norm_res)):
            u = _update_u(u, Pinv, u0, C)
            residual = _residual(u, u0, C)
            norm_res = _norm(residual)
            niters = niters + 1

    return norm_res, u, residual, niters


def build_model(M):
    (model_init, model_apply) = stax.serial(
        stax.elementwise(lambda x: jnp.float64(x)),
        stax.Dense(64),
        stax.Relu,
        stax.Dense(64),
        stax.Relu,
        stax.Dense(M),
        stax.Sigmoid,
    )
    return (model_init, model_apply)


def build_opt(lr, params):
    (opt_init, opt_update, opt_get_params) = optimizers.adam(lr)
    opt_state = opt_init(params)
    return (opt_state, opt_update, opt_get_params)


def init_input(input_shape):
    return jnp.zeros(input_shape, dtype=jnp.complex128)


@jax.jit
def build_input(inputs, u, residual, niters):
    new_input = jnp.concatenate((u, residual))
    offset = niters * len(new_input)
    # inputs = inputs.at[offset:offset + len(new_input)].set(new_input)
    inputs = jax.lax.dynamic_update_slice(inputs, new_input, [offset])
    return inputs


def main():
    seed = 0

    M = 3
    dt = 1.0
    restol = 1e-10

    max_episode_length = 50

    steps = 100000
    batch_size = 64
    # lr = 0.0001
    lr = 0.0003
    # lr = 0.001
    input_shape = (M * 2 * max_episode_length,)

    orig_step_fun = step
    step_fun = orig_step_fun

    rng = np.random.default_rng(seed)
    rng_key = jax.random.PRNGKey(seed)

    model_init, model_apply = build_model(M)
    _, params = model_init(rng_key, input_shape)
    opt_state, opt_update, opt_get_params = build_opt(lr, params)

    if use_unstable_jax_jit:
        pass
        # step_fun = jax.jit(step_fun, static_argnums=(3, 4, 6, 9, 10))
        # step_fun = jax.jit(step_fun)

    if not use_simple_loss:
        def loss(params, u, residual, Q, M, coll_num_nodes, lam, dt, u0, C,
                 restol, inputs, niters):
            # print(u, residual)
            # inputs = build_input(inputs, u, residual, niters)
            action = model_apply(params, inputs)
            norm_res = step_fun(action, u, Q, M, coll_num_nodes, lam, dt, u0,
                                C, restol, max_episode_length)[0]
            if use_jax_control_flow:
                return jax.lax.cond(
                    jnp.isnan(norm_res),
                    lambda _: float(max_episode_length),
                    lambda _: norm_res,
                    None,
                )

                # return jax.lax.cond(
                #     jnp.isnan(norm_res),
                #     None,
                #     lambda _: -float(max_episode_length),
                #     None,
                #     lambda _: norm_res,
                # )
            else:
                if use_for_max_episode_length:
                    return norm_res
                return norm_res + niters
                # if not jnp.isnan(norm_res):
                #     return -norm_res
                # return jnp.finfo('d').min
                # return -float(max_episode_length)

        if use_unstable_jax_jit:
            loss = jax.jit(loss, static_argnums=(4, 5, 7, 10))
            # loss = jax.jit(loss)

        grad_loss = jax.grad(loss)

        if use_unstable_jax_jit:
            grad_loss = jax.jit(grad_loss, static_argnums=(4, 5, 7, 10))
            # grad_loss = jax.jit(jax.grad(loss))

        def update(
                i, opt_state, u, residual, Q, M, coll_num_nodes, lam, dt, u0,
                C, restol, inputs, niters,
        ):
            params = opt_get_params(opt_state)
            return opt_update(
                i,
                grad_loss(
                    params, u, residual, Q, M, coll_num_nodes, lam, dt, u0, C,
                    restol, inputs, niters,
                    # params, norm_res,
                ),
                opt_state,
            )

        if use_unstable_jax_jit:
            update = jax.jit(update, static_argnums=(5, 6, 8, 11))
            # update = jax.jit(update)
    else:
        def loss(params, norm_res, niters):
            return norm_res + niters

        if use_unstable_jax_jit:
            loss = jax.jit(loss)

        grad_loss = jax.grad(loss)

        if use_unstable_jax_jit:
            grad_loss = jax.jit(jax.grad(loss))

        def update(i, opt_state, norm_res, niters):
            params = opt_get_params(opt_state)
            return opt_update(
                i,
                grad_loss(params, norm_res, niters),
                opt_state,
            )

        if use_unstable_jax_jit:
            update = jax.jit(update)

    def train(opt_state):
        start_time = time.perf_counter()
        steps_taken = 0
        episodes = 0
        norm_resids = []
        episode_norm_resids = []
        losses = []
        episode_losses = []
        all_niters = []

        while steps_taken < steps:
            (coll, Q, u0, lam, C, u, residual) = get_initial(M, dt, rng)
            norm_res = _norm(residual)
            coll_num_nodes = coll.num_nodes
            # prev_u = jnp.zeros_like(u)
            inputs = init_input(input_shape)

            niters = 0
            while (norm_res > restol
                   and niters < max_episode_length
                   and steps_taken < steps):
                params = opt_get_params(opt_state)

                inputs = build_input(inputs, u, residual, niters)
                action = model_apply(params, inputs)
                norm_res, u, residual, niters_ = step_fun(
                    action, u, Q, M, coll_num_nodes, lam, dt, u0, C, restol,
                    max_episode_length,
                )

                if jnp.isnan(norm_res):
                    print('ERR:', steps_taken)
                    break
                norm_resids.append(norm_res)
                if use_for_max_episode_length:
                    losses.append(norm_res)
                else:
                    losses.append(norm_res + niters)
                niters = niters_ if orig_step_fun is full_step else niters + 1
                all_niters.append(niters)

                if not use_simple_loss:
                    opt_state = update(
                        steps_taken, opt_state, u, residual, Q, M,
                        coll_num_nodes, lam, dt, u0, C, restol, inputs, niters,
                    )
                else:
                    opt_state = update(
                        steps_taken, opt_state, norm_res, niters,
                    )

                steps_taken += 1
                if steps_taken % 500 == 0:
                    num_last_entries = 500
                    last_losses = jnp.stack(losses[-num_last_entries:])
                    last_resids = jnp.stack(norm_resids[-num_last_entries:])
                    last_niters = jnp.stack(all_niters[-num_last_entries:])
                    print(f'Took {steps_taken} steps in '
                          f'{time.perf_counter() - start_time:.2f} seconds. '
                          f'{episodes} episodes. '
                          # f'Mean loss: {jnp.mean(resids)}. ')
                          f'Means over last {num_last_entries}: '
                          f'loss: {jnp.mean(last_losses)} '
                          f'resids: {jnp.mean(last_resids)} '
                          f'niters: {jnp.mean(last_niters):.2f}')

            if not jnp.isnan(norm_res):
                episode_norm_resids.append(norm_res)
                episode_losses.append(norm_res + niters)
            episodes += 1

        norm_resids = jnp.stack(norm_resids)
        with open('norm_resids.npy', 'wb') as f:
            jnp.save(f, norm_resids)

        losses = jnp.stack(losses)
        with open('losses.npy', 'wb') as f:
            jnp.save(f, losses)

        all_niters = jnp.stack(all_niters)
        with open('all_niters.npy', 'wb') as f:
            jnp.save(f, all_niters)

        trained_params = opt_get_params(opt_state)

        with open('model_params.npy', 'wb') as f:
            jnp.save(f, trained_params)

        return trained_params

    do_train = True
    if do_train:
        trained_params = train(opt_state)
    else:
        # Testing
        with open('model_params_100000_64_64_0003_sigmoid.npy', 'rb') as f:
            trained_params = jnp.load(f, allow_pickle=True)

    def test(trained_params):
        ntests = 2000
        mean_niter = 0
        nsucc = 0
        results = []

        for i in range(ntests):
            (coll, Q, u0, lam, C, u, residual) = get_initial(M, dt, rng)
            norm_res = _norm(residual)
            coll_num_nodes = coll.num_nodes
            inputs = init_input(input_shape)

            niters = 0
            err = False
            while (norm_res > restol
                   and niters < max_episode_length
                   and not err):
                inputs = build_input(inputs, u, residual, niters)
                action = model_apply(trained_params, inputs)
                norm_res, u, residual, niters_ = step_fun(
                    action, u, Q, M, coll_num_nodes, lam, dt, u0, C, restol,
                    max_episode_length,
                )

                err = jnp.isnan(norm_res)
                niters = niters_ if orig_step_fun is full_step else niters + 1

            if err:
                print('Test ERR')
            if norm_res <= restol and niters < max_episode_length and not err:
                nsucc += 1
                mean_niter += niters
                results.append((lam.real, niters))
            if i % 250 == 0:
                print(f'{i} tests done...')

        # Write out mean number of iterations (smaller is better) and the
        # success rate (target: 100 %)
        if nsucc > 0:
            mean_niter /= nsucc
        else:
            mean_niter = 666
        print(f'Mean number of iterations and success rate: '
              f'{mean_niter:4.2f}, {nsucc / ntests * 100} %')
        return results

    results = test(trained_params)
    sorted_results = sorted(results, key=lambda x: x[0])
    plt.plot(
        [i[0] for i in sorted_results],
        [i[1] for i in sorted_results],
        # color=color,
        # label=label,
    )
    plt.savefig('test_results.pdf')
    plt.show()


if __name__ == '__main__':
    main()
