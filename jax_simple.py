# from functools import partial
import datetime
import time

import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental import optimizers
import matplotlib.pyplot as plt
import numpy as np

from pySDC.implementations.collocation_classes.gauss_radau_right import \
    CollGaussRadau_Right

# Use double precision
jax.config.update('jax_enable_x64', True)

# Parameters

# Is buggy so don't use until fixed
use_jax_control_flow = False
# Whether to JIT loss and other functions that may not work
use_unstable_jax_jit = True
# Whether to use a loss function that does not call the step function.
# Probably does not work; the gradients would need to be tracked all the
# way into the batch.
use_simple_loss = False
# Whether to use the `full_step` function
use_full_step_env = False
# For full step: do not stop early, always iterate
# `max_episode_length` times. Enables us to JIT-compile the function.
unroll_full_step = False

seed = 0

M = 3
dt = 1.0
restol = 1e-10
lambda_real_range = [-100.0, 0.0]
lambda_imag_range = [0.0, 10.0]

max_episode_length = 50

steps = 1000000
batch_size = 256
hidden_layers = [32, 64, 32]
# lr = 0.0001
# lr = 0.0003
# lr = 0.001
# lr = 0.003
lr = 0.0003 * (np.log2(batch_size) + 1)
# lr = 0.0003 * batch_size
start_lr = 0.0003 * (np.log2(batch_size) + 1)
# start_lr = 0.0003 * batch_size
end_lr = 0.0003
steps_to_end_lr = 50000
schedule_polynomial_power = 1.0
# Whether to use learning rate scheduling. If `True`, `lr` is ignored.
# If `False`, `start_lr`, `end_lr`, `steps_to_end_lr` and
# `schedule_polynomial_power` are ignored.
use_lr_scheduling = True
time_step_weight = 1.0

# Whether to train or load a model for testing
do_train = True
# Whether to have only inputs of zeros
ignore_inputs = False

# How regularly to append data to our collections
collect_data_interval = 4


# Functions

def get_env(env_constants, rng):
    lam = (1 * rng.uniform(low=env_constants['lambda_real_range'][0],
                           high=env_constants['lambda_real_range'][1])
           + 0j * rng.uniform(low=env_constants['lambda_imag_range'][0],
                              high=env_constants['lambda_imag_range'][1]))
    C = (jnp.eye(env_constants['coll_num_nodes'])
         - lam * env_constants['dt'] * env_constants['Q'])
    u = jnp.ones(env_constants['coll_num_nodes'], dtype=jnp.complex128)

    initial_residual = env_constants['u0'] - C @ u
    norm_res = _norm(initial_residual)
    env = {
        **env_constants,
        'lam': lam,
        'C': C,
        'initial_residual': initial_residual,
        'initial_norm_res': norm_res,
    }
    return env, u


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


# @partial(jax.jit, static_argnums=(0,))
@jax.jit
def _get_pinv(coll_num_nodes_arr, lam, dt, Qdmat):
    # We use an array of the desired size to be able to JIT this function.
    # Stupid and unnecessary, but it works.
    return jnp.linalg.inv(jnp.eye(coll_num_nodes_arr.size) - lam * dt * Qdmat)


@jax.jit
def _update_u(u, Pinv, u0, C):
    return u + Pinv @ _residual(u, u0, C)


@jax.jit
def _residual(u, u0, C):
    return u0 - C @ u


@jax.jit
def _norm(res):
    return jnp.linalg.norm(res, jnp.inf)


# @partial(jax.jit, static_argnums=(2,))
@jax.jit
def step(action, u, env):
    Qdmat = _get_prec(action, env['Q'], env['M'])
    Pinv = _get_pinv(env['coll_num_nodes_arr'], env['lam'], env['dt'], Qdmat)
    # norm_res_old = _norm(init_resid)

    u = _update_u(u, Pinv, env['u0'], env['C'])
    residual = _residual(u, env['u0'], env['C'])
    norm_res = _norm(residual)
    return norm_res, u, residual, 1


# @partial(jax.jit, static_argnums=(2,))
@jax.jit
def full_step(action, u, env):
    Qdmat = _get_prec(action, env['Q'], env['M'])
    Pinv = _get_pinv(env['coll_num_nodes_arr'], env['lam'], env['dt'], Qdmat)
    residual = _residual(u, env['u0'], env['C'])
    norm_res = _norm(residual)
    # norm_res_old = _norm(init_resid)

    if unroll_full_step:
        for niters in range(env['max_episode_length']):
            u = _update_u(u, Pinv, env['u0'], env['C'])
            residual = _residual(u, env['u0'], env['C'])
            norm_res = _norm(residual)
        return (norm_res, u, residual, env['max_episode_length'])

    if use_jax_control_flow:
        init_val = (u, residual, norm_res, 0)

        def cond_fun(args):
            (_, _, norm_res, niters) = args
            # return jax.lax.cond(
            #     (norm_res > env['restol']
            #      and niters < env['max_episode_length']
            #      and not jnp.isnan(norm_res)),
            #     lambda _: True,
            #     lambda _: False,
            #     None,
            # )
            # return norm_res > 1E-10 and niters < max_episode_length
            return (norm_res > env['restol']
                    and niters < env['max_episode_length']
                    and not jnp.isnan(norm_res))

        def body_fun(args):
            (u, residual, norm_res, niters) = args
            u = _update_u(u, Pinv, env['u0'], env['C'])
            residual = _residual(u, env['u0'], env['C'])
            norm_res = _norm(residual)
            return (u, residual, norm_res, niters + 1)

        (u, residual, norm_res, niters) = jax.lax.while_loop(
            cond_fun, body_fun, init_val)
    else:
        niters = 0

        while (norm_res > env['restol']
               and niters < env['max_episode_length']
               and not jnp.isnan(norm_res)):
            u = _update_u(u, Pinv, env['u0'], env['C'])
            residual = _residual(u, env['u0'], env['C'])
            norm_res = _norm(residual)
            niters = niters + 1

    return norm_res, u, residual, niters


def build_model(input_shape, hidden_layers, M, seed):
    hidden_layers = [layer for num_hidden in hidden_layers
                     for layer in [stax.Dense(num_hidden), stax.Relu]]
    (model_init, model_apply) = stax.serial(
        stax.elementwise(
            lambda x: jax.lax.convert_element_type(x.real, jnp.float64)),
        *hidden_layers,
        stax.Dense(M),
        stax.Sigmoid,
    )

    rng_key = jax.random.PRNGKey(seed)
    _, params = model_init(rng_key, input_shape)
    return (model_apply, params)


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


@jax.jit
def calc_loss(norm_res, niters):
    return norm_res + niters * time_step_weight


def get_env_constants():
    coll = CollGaussRadau_Right(M, 0, 1)
    Q = coll.Qmat[1:, 1:]
    u0 = jnp.ones(coll.num_nodes, dtype=jnp.complex128)

    return {
        'M': M,
        'dt': dt,
        'restol': restol,
        'lambda_real_range': lambda_real_range,
        'lambda_imag_range': lambda_imag_range,
        'max_episode_length': max_episode_length,

        'coll_num_nodes': coll.num_nodes,
        # We use an array the size of `coll_num_nodes` to be able to JIT the
        # `_get_pinv` function.
        'coll_num_nodes_arr': jnp.empty(coll.num_nodes),
        'Q': Q,
        'u0': u0,
    }


def main():
    env_constants = get_env_constants()

    exp_start_time = str(
        datetime.datetime.today()).replace(':', '-').replace(' ', 'T')

    if use_full_step_env:
        step_fun = full_step
    else:
        step_fun = step

    rng = np.random.default_rng(seed)

    input_shape = (M * 2 * max_episode_length,)
    model_apply, params = build_model(input_shape, hidden_layers,
                                      env_constants['M'], seed)
    if use_lr_scheduling:
        opt_state, opt_update, opt_get_params = build_opt(
            optimizers.polynomial_decay(
                start_lr, steps_to_end_lr, end_lr, schedule_polynomial_power),
            params)
    else:
        opt_state, opt_update, opt_get_params = build_opt(lr, params)

    if use_unstable_jax_jit:
        pass
        # step_fun = jax.jit(step_fun, static_argnums=(3, 4, 6, 9, 10))
        # step_fun = jax.jit(step_fun)

    if not use_simple_loss:
        def loss(params, u, residual, env, input_batch, niters_batch):
            predict_batch = jax.vmap(
                lambda inputs: model_apply(params, inputs),
                -1,
            )
            action_batch = predict_batch(input_batch)
            step_batch = jax.vmap(lambda action: step_fun(action, u, env)[0])
            norm_res_batch = step_batch(action_batch)
            if use_jax_control_flow:
                return jax.lax.cond(
                    jnp.isnan(norm_res_batch),
                    lambda _: float(env['max_episode_length']),
                    lambda _: jnp.sum(norm_res_batch),
                    None,
                )
            else:
                return jnp.sum(calc_loss(norm_res_batch, niters_batch))
                # if not jnp.isnan(norm_res_batch):
                #     return -norm_res_batch
                # return jnp.finfo('d').min
                # return -float(env['max_episode_length'])

        if use_unstable_jax_jit:
            # loss = jax.jit(loss, static_argnums=(3,))
            loss = jax.jit(loss)

        grad_loss = jax.grad(loss)

        if use_unstable_jax_jit:
            # grad_loss = jax.jit(grad_loss, static_argnums=(3,))
            grad_loss = jax.jit(jax.grad(loss))

        def update(i, opt_state, u, residual, env, input_batch, niters_batch):
            params = opt_get_params(opt_state)
            return opt_update(
                i,
                grad_loss(params, u, residual, env, input_batch, niters_batch),
                opt_state,
            )

        if use_unstable_jax_jit:
            # update = jax.jit(update, static_argnums=(4,))
            update = jax.jit(update)
    else:
        def loss(params, norm_res_batch, niters_batch):
            return jnp.sum(calc_loss(norm_res_batch, niters_batch))

        if use_unstable_jax_jit:
            loss = jax.jit(loss)

        grad_loss = jax.grad(loss)

        if use_unstable_jax_jit:
            grad_loss = jax.jit(jax.grad(loss))

        def update(i, opt_state, norm_res_batch, niters_batch):
            params = opt_get_params(opt_state)
            return opt_update(
                i,
                grad_loss(params, norm_res_batch, niters_batch),
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
        episode_niters = []

        if not use_simple_loss:
            input_batch = jnp.empty(input_shape + (batch_size,),
                                    dtype=jnp.float64)
        else:
            input_batch = jnp.empty(batch_size)
        niters_batch = jnp.empty(batch_size)
        batch_index = 0

        while steps_taken < steps:
            env, u = get_env(env_constants, rng)
            residual = env['initial_residual']
            norm_res = env['initial_norm_res']
            inputs = init_input(input_shape)

            niters = 0
            while (norm_res > env['restol']
                   and niters < env['max_episode_length']
                   and steps_taken < steps):
                params = opt_get_params(opt_state)

                if not ignore_inputs:
                    inputs = build_input(inputs, u, residual, niters)
                action = model_apply(params, inputs)
                norm_res, u, residual, niters_ = step_fun(action, u, env)

                if jnp.isnan(norm_res):
                    print('ERR:', steps_taken)
                    break

                niters = niters_ if use_full_step_env else niters + 1
                if steps_taken % collect_data_interval == 0:
                    norm_resids.append(norm_res)
                    losses.append(calc_loss(norm_res, niters))
                    all_niters.append(niters)

                if not use_simple_loss:
                    input_batch = input_batch.at[:, batch_index].set(
                        inputs.real)
                else:
                    input_batch = input_batch.at[batch_index].set(norm_res)
                niters_batch = niters_batch.at[batch_index].set(niters)
                batch_index += 1

                if batch_index >= batch_size:
                    if not use_simple_loss:
                        opt_state = update(
                            steps_taken, opt_state, u, residual, env,
                            input_batch, niters_batch,
                        )
                    else:
                        opt_state = update(
                            steps_taken, opt_state, input_batch, niters_batch,
                        )
                    batch_index = 0

                steps_taken += 1
                if steps_taken % 500 == 0:
                    num_last_entries = 500
                    last_losses = jnp.stack(losses[-num_last_entries:])
                    last_resids = jnp.stack(norm_resids[-num_last_entries:])
                    last_niters = jnp.stack(episode_niters[-num_last_entries:])
                    print(f'Took {steps_taken} steps in '
                          f'{time.perf_counter() - start_time:.2f} seconds. '
                          f'{episodes} episodes. '
                          # f'Mean loss: {jnp.mean(resids)}. ')
                          f'Means over last {num_last_entries}: '
                          f'loss: {jnp.mean(last_losses)} '
                          f'resids: {jnp.mean(last_resids)} '
                          f'episode niters: {jnp.mean(last_niters):.2f}')

            if not jnp.isnan(norm_res):
                episode_norm_resids.append(norm_res)
                episode_losses.append(calc_loss(norm_res, niters))
                episode_niters.append(niters)
            episodes += 1

        norm_resids = jnp.stack(norm_resids)
        with open(f'norm_resids_{exp_start_time}.npy', 'wb') as f:
            jnp.save(f, norm_resids)

        losses = jnp.stack(losses)
        with open(f'losses_{exp_start_time}.npy', 'wb') as f:
            jnp.save(f, losses)

        all_niters = jnp.stack(all_niters)
        with open(f'all_niters_{exp_start_time}.npy', 'wb') as f:
            jnp.save(f, all_niters)

        trained_params = opt_get_params(opt_state)

        with open(f'model_params_{exp_start_time}.npy', 'wb') as f:
            jnp.save(f, trained_params)

        return trained_params

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
            env, u = get_env(env_constants, rng)
            residual = env['initial_residual']
            norm_res = _norm(residual)
            inputs = init_input(input_shape)

            niters = 0
            err = False
            while (
                    norm_res > env['restol']
                    and niters < env['max_episode_length']
                    and not err
            ):
                if not ignore_inputs:
                    inputs = build_input(inputs, u, residual, niters)
                action = model_apply(trained_params, inputs)
                norm_res, u, residual, niters_ = step_fun(action, u, env)

                err = jnp.isnan(norm_res)
                niters = niters_ if use_full_step_env else niters + 1

            if err:
                print('Test ERR')
            if (
                    norm_res <= env['restol']
                    and niters < env['max_episode_length']
                    and not err
            ):
                nsucc += 1
                mean_niter += niters
                results.append((env['lam'].real, niters))
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
    plt.savefig(f'test_results_{exp_start_time}.pdf')
    plt.show()


if __name__ == '__main__':
    main()
