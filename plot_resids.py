import jax.numpy as jnp
import matplotlib.pyplot as plt


def main():
    resids = jnp.load('norm_resids_10000_jit_2.npy', allow_pickle=True)
    resids[resids > 2] = 2
    plt.plot(jnp.arange(len(resids)), resids)
    plt.show()


if __name__ == '__main__':
    main()
