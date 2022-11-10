import jax
import jax.numpy as jnp

def _transfer_function_power_dB(x, poles):
    """x in kHz, poles in rad kHz"""
    def labs(x):
        return jnp.log10(jnp.abs(x))

    s = (1j)*x*2*jnp.pi # rad kHz
    G = jnp.sum(2*labs(poles))
    denom = jnp.sum(labs(s - poles) + labs(s - jnp.conjugate(poles)))
    return 20*(G - denom)

transfer_function_power_dB = jax.vmap(_transfer_function_power_dB, (0, None))