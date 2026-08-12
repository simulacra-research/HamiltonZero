# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import jax
import jax.numpy as jnp
import kfac_jax
from kfac_jax._src import utils as kfac_utils
from kfac_jax._src.layers_and_loss_tags import LayerMetaData, layer_tag

_FEATURIZER_OUTPUT_SPLIT_IDS = frozenset(
    {"featurizer.global_w1", "featurizer.combine_w1"}
)
_FEATURIZER_INPUT_SPLIT_IDS = frozenset(
    {"featurizer.global_w2", "featurizer.combine_w2"}
)


def _floor_matrix_avg_diag(mat, eps: float):
    d = mat.shape[-1]
    eps_arr = jnp.asarray(eps, dtype=mat.dtype)
    avg_diag = jnp.trace(mat) / d
    shift = jnp.maximum(eps_arr, eps_arr - avg_diag)
    return mat + shift * jnp.eye(d, dtype=mat.dtype)


def _balanced_axis_partition(shape: tuple[int, ...]):
    n = len(shape)
    full_mask = (1 << n) - 1
    best = None
    for mask in range(1, full_mask):
        if not mask & 1:
            continue
        left_axes = tuple((i for i in range(n) if mask & 1 << i))
        right_axes = tuple((i for i in range(n) if not mask & 1 << i))
        left_prod = _prod_int((shape[i] for i in left_axes))
        right_prod = _prod_int((shape[i] for i in right_axes))
        score = (max(left_prod, right_prod), abs(left_prod - right_prod))
        if best is None or score < best[0]:
            best = (score, left_axes, right_axes)
    assert best is not None
    return (best[1], best[2])


def _prod_int(vals) -> int:
    out = 1
    for v in vals:
        out *= int(v)
    return out


def _matricize(x, left_axes, right_axes):
    shape = tuple(x.shape)
    perm = tuple(left_axes) + tuple(right_axes)
    left_dim = _prod_int((shape[i] for i in left_axes))
    right_dim = _prod_int((shape[i] for i in right_axes))
    return jnp.transpose(x, perm).reshape(left_dim, right_dim)


def _unmatricize(x_mat, shape, left_axes, right_axes):
    left_shape = tuple((shape[i] for i in left_axes))
    right_shape = tuple((shape[i] for i in right_axes))
    perm = tuple(left_axes) + tuple(right_axes)
    inv_perm_list = [0] * len(perm)
    for pos, axis in enumerate(perm):
        inv_perm_list[axis] = pos
    inv_perm = tuple(inv_perm_list)
    x_perm = x_mat.reshape(left_shape + right_shape)
    return jnp.transpose(x_perm, inv_perm)


def _validate_approx_inverse_cache_request(
    exact_powers_to_cache, approx_powers_to_cache
):
    if exact_powers_to_cache:
        raise NotImplementedError(
            "Custom merge blocks do not implement exact cached powers."
        )
    unsupported = set(approx_powers_to_cache) - {-1}
    if unsupported:
        raise NotImplementedError(
            f"Unsupported approximate cached powers: {sorted(unsupported)}."
        )


def _init_two_kron_cache(
    left_dim,
    right_dim,
    dtype,
    exact_powers_to_cache,
    approx_powers_to_cache,
    cache_eigenvalues,
):
    _validate_approx_inverse_cache_request(
        exact_powers_to_cache, approx_powers_to_cache
    )
    cache = {}
    if -1 in approx_powers_to_cache:
        cache["-1"] = {
            "left_factor": jnp.eye(left_dim, dtype=dtype),
            "right_factor": jnp.eye(right_dim, dtype=dtype),
        }
    if cache_eigenvalues:
        cache["eigenvalues"] = jnp.zeros((left_dim * right_dim,), dtype=dtype)
    return cache


def _update_two_kron_cache(
    state,
    left_factor,
    right_factor,
    identity_weight,
    exact_powers,
    approx_powers,
    eigenvalues,
    *,
    inverse_epsilon=None,
):
    _validate_approx_inverse_cache_request(exact_powers, approx_powers)
    state = state.copy()
    if eigenvalues:
        s_left, _ = kfac_utils.safe_psd_eigh(left_factor)
        s_right, _ = kfac_utils.safe_psd_eigh(right_factor)
        state.cache["eigenvalues"] = jnp.einsum("p,q->pq", s_left, s_right).reshape(-1)
    if -1 in approx_powers:
        if inverse_epsilon is not None:
            left_for_inverse = _floor_matrix_avg_diag(left_factor, inverse_epsilon)
            right_for_inverse = _floor_matrix_avg_diag(right_factor, inverse_epsilon)
        else:
            left_for_inverse = left_factor
            right_for_inverse = right_factor
        inv_left, inv_right = kfac_utils.pi_adjusted_kronecker_inverse(
            left_for_inverse, right_for_inverse, damping=identity_weight
        )
        state.cache["-1"]["left_factor"] = inv_left
        state.cache["-1"]["right_factor"] = inv_right
    return state


def _two_kron_marginal_from_merge(dy_m, uA_m, uB_m, group_axes):
    lower = ("i", "j", "k", "l")
    upper = ("I", "J", "K", "L")
    group_axes = tuple(group_axes)
    group_set = set(group_axes)

    def _labels(axes, primed: bool):
        out = ["n"]
        for ax in axes:
            out.append(upper[ax] if primed and ax in group_set else lower[ax])
        return "".join(out)

    dy1 = _labels((0, 1), primed=False)
    uA1 = _labels((0, 2), primed=False)
    uB1 = _labels((0, 3), primed=False)
    dy2 = _labels((0, 1), primed=True)
    uA2 = _labels((0, 2), primed=True)
    uB2 = _labels((0, 3), primed=True)
    out = "".join((lower[ax] for ax in group_axes))
    out += "".join((upper[ax] for ax in group_axes))
    eqn = f"{dy1},{uA1},{uB1},{dy2},{uA2},{uB2}->{out}"
    gram = jnp.einsum(eqn, dy_m, uA_m, uB_m, dy_m, uA_m, uB_m)
    dims = (dy_m.shape[1], dy_m.shape[2], uA_m.shape[2], uB_m.shape[2])
    dim = _prod_int((dims[ax] for ax in group_axes))
    return gram.reshape(dim, dim)


def _merge_gradient_trace(dy_m, uA_m, uB_m, divisor):
    squared_norm_sum = jnp.einsum(
        "nij,nik,nil->", jnp.square(dy_m), jnp.square(uA_m), jnp.square(uB_m)
    )
    return squared_norm_sum / divisor


def _trace_normalize_two_kron_marginals(
    left_factor, right_factor, trace_mass, *, repeat_mass=1.0
):
    trace_mass = jnp.asarray(trace_mass, dtype=left_factor.dtype)
    repeat_mass = jnp.asarray(repeat_mass, dtype=left_factor.dtype)
    finite = jnp.isfinite(trace_mass) & jnp.isfinite(repeat_mass)
    no_mass = finite & ((trace_mass <= 0) | (repeat_mass <= 0))
    safe_trace = jnp.where(no_mass, jnp.ones_like(trace_mass), trace_mass)
    safe_repeat = jnp.where(no_mass, jnp.zeros_like(repeat_mass), repeat_mass)
    factor_scale = jnp.sqrt(safe_repeat / safe_trace)
    normalized_left = jnp.where(
        no_mass, jnp.zeros_like(left_factor), factor_scale * left_factor
    )
    normalized_right = jnp.where(
        no_mass, jnp.zeros_like(right_factor), factor_scale * right_factor
    )
    normalized_left = jnp.where(
        finite, normalized_left, jnp.full_like(left_factor, jnp.nan)
    )
    normalized_right = jnp.where(
        finite, normalized_right, jnp.full_like(right_factor, jnp.nan)
    )
    return (normalized_left, normalized_right)


def _identity_wma(dim, dtype):
    return kfac_utils.WeightedMovingAverage(
        value=jnp.eye(dim, dtype=dtype), weight=jnp.asarray(1.0, dtype=dtype)
    )


def _scalar_wma(value, dtype):
    return kfac_utils.WeightedMovingAverage(
        value=jnp.asarray(value, dtype=dtype), weight=jnp.asarray(1.0, dtype=dtype)
    )


def _poison_cached_inverse_on_failure(state, factor_key, certified):
    if "-1" in state.cache:
        cached = state.cache["-1"][factor_key]
        state.cache["-1"][factor_key] = jnp.where(
            certified, cached, jnp.full_like(cached, jnp.nan)
        )
    return state


STRUCTURAL_QUADRILINEAR_MERGE_TAG_VARIANT = "structural_quadrilinear_merge"


def _structural_name_kw(name: str | None) -> dict[str, str]:
    return {} if name is None else {"name": name}


def register_structural_quadrilinear_merge(
    y,
    x_l,
    x_r,
    T,
    structural_mask,
    *,
    scan_shared: bool,
    repeat_ndim: int,
    name: str | None = None,
):
    if tuple(x_l.shape) != tuple(x_r.shape):
        raise ValueError(
            f"quadrilinear input shapes differ: {x_l.shape} vs {x_r.shape}"
        )
    if tuple(structural_mask.shape) != tuple(x_l.shape[:-1]):
        raise ValueError(
            f"quadrilinear structural mask must match local leading shape: mask={structural_mask.shape}, input={x_l.shape}"
        )
    return layer_tag.bind(
        y,
        x_l,
        x_r,
        structural_mask,
        T,
        meta=LayerMetaData(
            variant=STRUCTURAL_QUADRILINEAR_MERGE_TAG_VARIANT,
            outputs_index=(0,),
            inputs_index=(1, 2, 3),
            params_index=(4,),
        ),
        scan_shared=bool(scan_shared),
        repeat_ndim=int(repeat_ndim),
        **_structural_name_kw(name),
    )


@kfac_utils.register_state_class
class _QuadrilinearMergeState(kfac_jax.CurvatureBlock.State):
    sigma_left: kfac_utils.WeightedMovingAverage
    sigma_right: kfac_utils.WeightedMovingAverage


class _QuadrilinearMergeBlock(kfac_jax.CurvatureBlock):
    State = _QuadrilinearMergeState

    @property
    def parameters_canonical_order(self) -> tuple[int, ...]:
        return (0,)

    def _init(
        self, rng, exact_powers_to_cache, approx_powers_to_cache, cache_eigenvalues
    ):
        del rng
        shape = tuple(self.parameters_shapes[0])
        left_axes, right_axes = _balanced_axis_partition(shape)
        left_dim = _prod_int((shape[i] for i in left_axes))
        right_dim = _prod_int((shape[i] for i in right_axes))

        def _eye_wma(d):
            return kfac_utils.WeightedMovingAverage(
                value=jnp.eye(d, dtype=self.dtype),
                weight=jnp.asarray(1.0, dtype=self.dtype),
            )

        return _QuadrilinearMergeState(
            cache=_init_two_kron_cache(
                left_dim,
                right_dim,
                self.dtype,
                exact_powers_to_cache,
                approx_powers_to_cache,
                cache_eigenvalues,
            ),
            sigma_left=_eye_wma(left_dim),
            sigma_right=_eye_wma(right_dim),
        )

    def sync(self, state, pmap_axis_name):
        state = state.copy()
        for f in (state.sigma_left, state.sigma_right):
            f.sync(pmap_axis_name)
        return state

    def update_curvature_matrix_estimate(
        self, state, estimation_data, ema_old, ema_new, identity_weight, batch_size
    ):
        del identity_weight, batch_size
        state = state.copy()
        u_a, u_b = estimation_data.primals.inputs
        [dy] = estimation_data.tangents.outputs
        [T_param] = estimation_data.primals.params
        G, d_j, d_k, d_l = T_param.shape
        d_m_eff = G * d_l

        def _find_last_feature_axis(arr, size):
            for i in range(arr.ndim - 1, -1, -1):
                if arr.shape[i] == size:
                    return i
            return arr.ndim - 1

        ax_dy = _find_last_feature_axis(dy, d_m_eff)
        ax_uA = _find_last_feature_axis(u_a, d_m_eff)
        ax_uB = _find_last_feature_axis(u_b, d_m_eff)
        dy_f = jnp.moveaxis(dy, ax_dy, -1).reshape(-1, G, d_j)
        uA_f = jnp.moveaxis(u_a, ax_uA, -1).reshape(-1, G, d_k)
        uB_f = jnp.moveaxis(u_b, ax_uB, -1).reshape(-1, G, d_l)
        is_active = 1.0 - jnp.all(dy_f == 0.0, axis=(-2, -1), keepdims=True).astype(
            dy_f.dtype
        )
        n_active = jnp.sum(is_active)
        normalizer = jnp.maximum(n_active, 1.0).astype(dy_f.dtype)
        inv_n = jnp.reciprocal(normalizer)
        dy_m = dy_f * is_active
        uA_m = uA_f * is_active
        uB_m = uB_f * is_active
        shape = tuple(T_param.shape)
        left_axes, right_axes = _balanced_axis_partition(shape)
        sigma_left_new = (
            _two_kron_marginal_from_merge(dy_m, uA_m, uB_m, left_axes) * inv_n
        )
        sigma_right_new = (
            _two_kron_marginal_from_merge(dy_m, uA_m, uB_m, right_axes) * inv_n
        )
        trace_mass = _merge_gradient_trace(dy_m, uA_m, uB_m, normalizer)
        sigma_left_new, sigma_right_new = _trace_normalize_two_kron_marginals(
            sigma_left_new, sigma_right_new, trace_mass
        )
        sigma_left_new = 0.5 * (sigma_left_new + sigma_left_new.T)
        sigma_right_new = 0.5 * (sigma_right_new + sigma_right_new.T)
        state.sigma_left.update(sigma_left_new, ema_old, ema_new)
        state.sigma_right.update(sigma_right_new, ema_old, ema_new)
        return state

    _MATPOWER_EPSILON_FLOOR: float = 1e-06

    def _multiply_matpower_unscaled(
        self, state, vector, identity_weight, power, exact_power, use_cached
    ):
        if exact_power and power != 1:
            raise NotImplementedError(
                "QuadrilinearMergeBlock implements approximate inverse powers only."
            )
        [grad_T] = vector
        shape = tuple(self.parameters_shapes[0])
        left_axes, right_axes = _balanced_axis_partition(shape)
        grad_mat = _matricize(grad_T, left_axes, right_axes)
        if power == -1:
            if use_cached:
                inv_left = state.cache["-1"]["left_factor"]
                inv_right = state.cache["-1"]["right_factor"]
            else:
                eps = self._MATPOWER_EPSILON_FLOOR
                inv_left, inv_right = kfac_utils.pi_adjusted_kronecker_inverse(
                    _floor_matrix_avg_diag(state.sigma_left.value, eps),
                    _floor_matrix_avg_diag(state.sigma_right.value, eps),
                    damping=identity_weight,
                )
            new_mat = jnp.einsum("pP,qQ,PQ->pq", inv_left, inv_right, grad_mat)
        elif power == 1:
            curvature_product = jnp.einsum(
                "pP,qQ,PQ->pq",
                state.sigma_left.value,
                state.sigma_right.value,
                grad_mat,
            )
            if use_cached:
                curvature_product = (
                    self.state_dependent_scale(state) * curvature_product
                )
            new_mat = curvature_product + identity_weight * grad_mat
        else:
            raise NotImplementedError(
                f"QuadrilinearMergeBlock: power={power} not implemented (only ±1 supported)."
            )
        new_T = _unmatricize(new_mat, shape, left_axes, right_axes)
        return (new_T,)

    def _eigenvalues_unscaled(self, state, use_cached):
        if use_cached:
            return state.cache["eigenvalues"]
        s_left, _ = kfac_utils.safe_psd_eigh(state.sigma_left.value)
        s_right, _ = kfac_utils.safe_psd_eigh(state.sigma_right.value)
        return jnp.einsum("p,q->pq", s_left, s_right).reshape(-1)

    def _update_cache(
        self, state, identity_weight, exact_powers, approx_powers, eigenvalues
    ):
        eps = self._MATPOWER_EPSILON_FLOOR
        return _update_two_kron_cache(
            state,
            state.sigma_left.value,
            state.sigma_right.value,
            identity_weight,
            exact_powers,
            approx_powers,
            eigenvalues,
            inverse_epsilon=eps,
        )

    def _to_dense_unscaled(self, state):
        return jnp.kron(state.sigma_left.value, state.sigma_right.value)

    def _norm_unscaled(self, state, norm_type):
        n_left = kfac_utils.psd_matrix_norm(state.sigma_left.value, norm_type=norm_type)
        n_right = kfac_utils.psd_matrix_norm(
            state.sigma_right.value, norm_type=norm_type
        )
        return n_left * n_right


@kfac_utils.register_state_class
class _StructuralQuadrilinearMergeState(_QuadrilinearMergeState):
    average_repeats: kfac_utils.WeightedMovingAverage


class StructuralQuadrilinearMergeBlock(_QuadrilinearMergeBlock):
    State = _StructuralQuadrilinearMergeState

    def _init(
        self, rng, exact_powers_to_cache, approx_powers_to_cache, cache_eigenvalues
    ):
        base = super()._init(
            rng, exact_powers_to_cache, approx_powers_to_cache, cache_eigenvalues
        )
        return self.State(
            **base.__dict__,
            average_repeats=kfac_utils.WeightedMovingAverage(
                value=jnp.ones((), dtype=self.dtype),
                weight=jnp.asarray(1.0, dtype=self.dtype),
            ),
        )

    def sync(self, state, pmap_axis_name):
        state = super().sync(state, pmap_axis_name)
        state.average_repeats.sync(pmap_axis_name)
        return state

    def state_dependent_scale(self, state):
        return 1.0 / jnp.where(
            state.average_repeats.value > 0, state.average_repeats.value, 1.0
        )

    def _update_cache(
        self, state, identity_weight, exact_powers, approx_powers, eigenvalues
    ):
        state = super()._update_cache(
            state, identity_weight, exact_powers, approx_powers, eigenvalues
        )
        scale = self.state_dependent_scale(state)
        if eigenvalues:
            state.cache["eigenvalues"] = scale * state.cache["eigenvalues"]
        if -1 in approx_powers:
            factor_scale = jnp.sqrt(scale)
            state.cache["-1"]["left_factor"] /= factor_scale
            state.cache["-1"]["right_factor"] /= factor_scale
        return state

    def update_curvature_matrix_estimate(
        self, state, estimation_data, ema_old, ema_new, identity_weight, batch_size
    ):
        del identity_weight
        from hamiltonzero.optim.blocks import (
            align_structural_mask_to_leading,
            structural_group_repeats,
        )

        state = state.copy()
        u_a, u_b, structural_mask = estimation_data.primals.inputs
        [dy] = estimation_data.tangents.outputs
        [T_param] = estimation_data.primals.params
        scan_shared = bool(self._layer_tag_eq.params["scan_shared"])
        repeat_ndim = int(self._layer_tag_eq.params["repeat_ndim"])
        structural_mask = align_structural_mask_to_leading(
            structural_mask, dy.shape[:-1], repeat_ndim=repeat_ndim
        )
        ua_g, mask_g, logical_batch, _ = structural_group_repeats(
            u_a,
            structural_mask,
            scan_shared=scan_shared,
            repeat_ndim=repeat_ndim,
            feature_ndim=1,
        )
        ub_g, _, ub_batch, _ = structural_group_repeats(
            u_b,
            structural_mask,
            scan_shared=scan_shared,
            repeat_ndim=repeat_ndim,
            feature_ndim=1,
        )
        dy_g, _, dy_batch, _ = structural_group_repeats(
            dy,
            structural_mask,
            scan_shared=scan_shared,
            repeat_ndim=repeat_ndim,
            feature_ndim=1,
        )
        if logical_batch != ub_batch or logical_batch != dy_batch:
            raise ValueError("quadrilinear structural logical batches differ")
        G, d_j, d_k, d_l = T_param.shape
        row_mask = mask_g.astype(dy_g.dtype)[..., None, None]
        dy_f = dy_g.reshape(-1, G, d_j) * row_mask.reshape(-1, 1, 1)
        uA_f = ua_g.reshape(-1, G, d_k) * row_mask.reshape(-1, 1, 1)
        uB_f = ub_g.reshape(-1, G, d_l) * row_mask.reshape(-1, 1, 1)
        sample_divisor = jnp.maximum(
            jnp.asarray(batch_size, dtype=dy_f.dtype),
            jnp.asarray(1.0, dtype=dy_f.dtype),
        )
        logical_divisor = jnp.asarray(logical_batch, dtype=dy_f.dtype)
        shape = tuple(T_param.shape)
        left_axes, right_axes = _balanced_axis_partition(shape)
        sigma_left = (
            _two_kron_marginal_from_merge(dy_f, uA_f, uB_f, left_axes) / sample_divisor
        )
        sigma_right = (
            _two_kron_marginal_from_merge(dy_f, uA_f, uB_f, right_axes) / sample_divisor
        )
        repeats = jnp.sum(mask_g) / logical_divisor
        trace_mass = _merge_gradient_trace(dy_f, uA_f, uB_f, sample_divisor)
        sigma_left, sigma_right = _trace_normalize_two_kron_marginals(
            sigma_left, sigma_right, trace_mass, repeat_mass=repeats
        )
        sigma_left = 0.5 * (sigma_left + sigma_left.T)
        sigma_right = 0.5 * (sigma_right + sigma_right.T)
        state.sigma_left.update(sigma_left, ema_old, ema_new)
        state.sigma_right.update(sigma_right, ema_old, ema_new)
        state.average_repeats.update(repeats, ema_old, ema_new)
        return state


kfac_jax.set_default_tag_to_block_ctor(
    STRUCTURAL_QUADRILINEAR_MERGE_TAG_VARIANT, StructuralQuadrilinearMergeBlock
)
SMALL_FULL_TAG_VARIANT = "small_full"
_SMALL_FULL_MAX_SIZE = 4096


@kfac_utils.register_state_class
class _SmallFullBlockState(kfac_jax.CurvatureBlock.State):
    matrix: kfac_utils.WeightedMovingAverage


class SmallFullBlock(kfac_jax.CurvatureBlock):
    State = _SmallFullBlockState

    @property
    def parameters_canonical_order(self) -> tuple[int, ...]:
        return (0,)

    def _param_size(self) -> int:
        shape = self.parameters_shapes[0]
        n = 1
        for s in shape:
            n *= int(s)
        return n

    @staticmethod
    def _safe_eigh(matrix):
        matrix = 0.5 * (matrix + matrix.T)
        diagonal_scale = jnp.max(jnp.abs(jnp.diagonal(matrix)))
        floor = jnp.maximum(
            jnp.asarray(1e-06, dtype=matrix.dtype),
            jnp.asarray(0.0001, dtype=matrix.dtype) * diagonal_scale,
        )
        matrix = matrix + floor * jnp.eye(matrix.shape[0], dtype=matrix.dtype)
        scale = jnp.maximum(
            jnp.max(jnp.abs(matrix)), jnp.asarray(1.0, dtype=matrix.dtype)
        )
        eigenvalues, eigenvectors = kfac_utils.safe_psd_eigh(matrix / scale)
        return (eigenvalues * scale, eigenvectors)

    def _init(
        self, rng, exact_powers_to_cache, approx_powers_to_cache, cache_eigenvalues
    ):
        del rng
        n = self._param_size()
        powers_to_cache = set(exact_powers_to_cache) | set(approx_powers_to_cache)
        unsupported = powers_to_cache - {-1}
        if unsupported:
            raise NotImplementedError(
                f"SmallFullBlock does not cache powers {sorted(unsupported)}."
            )
        cache = {}
        if -1 in powers_to_cache:
            cache["-1"] = jnp.eye(n, dtype=self.dtype)
        if cache_eigenvalues:
            cache["eigenvalues"] = jnp.zeros((n,), dtype=self.dtype)
        return self.State(
            cache=cache,
            matrix=kfac_utils.WeightedMovingAverage(
                value=jnp.eye(n, dtype=self.dtype),
                weight=jnp.asarray(1.0, dtype=self.dtype),
            ),
        )

    def sync(self, state, pmap_axis_name):
        state = state.copy()
        state.matrix.sync(pmap_axis_name)
        return state

    def update_curvature_matrix_estimate(
        self, state, estimation_data, ema_old, ema_new, identity_weight, batch_size
    ):
        del identity_weight
        state = state.copy()
        [dy] = estimation_data.tangents.outputs
        n = self._param_size()
        d2 = dy.reshape(-1, n)
        divisor = jnp.maximum(
            jnp.asarray(batch_size, dtype=d2.dtype), jnp.asarray(1.0, dtype=d2.dtype)
        )
        fisher = d2.T @ d2 / divisor
        fisher = 0.5 * (fisher + fisher.T)
        state.matrix.update(fisher, ema_old, ema_new)
        return state

    def _multiply_matpower_unscaled(
        self, state, vector, identity_weight, power, exact_power, use_cached
    ):
        del exact_power
        [v] = vector
        n = self._param_size()
        vf = v.reshape(n)
        if power == -1:
            if use_cached:
                out = state.cache["-1"] @ vf
            else:
                m = 0.5 * (state.matrix.value + state.matrix.value.T)
                w_eig, q_eig = self._safe_eigh(m)
                w_eig = w_eig + identity_weight
                out = q_eig @ (q_eig.T @ vf / w_eig)
        elif power == 1:
            m = 0.5 * (state.matrix.value + state.matrix.value.T)
            out = m @ vf + identity_weight * vf
        else:
            raise NotImplementedError(
                f"SmallFullBlock: power={power} not implemented (only ±1)."
            )
        return (out.reshape(v.shape),)

    def _eigenvalues_unscaled(self, state, use_cached):
        if use_cached:
            return state.cache["eigenvalues"]
        matrix = 0.5 * (state.matrix.value + state.matrix.value.T)
        eigenvalues, _ = self._safe_eigh(matrix)
        return eigenvalues

    def _update_cache(
        self, state, identity_weight, exact_powers, approx_powers, eigenvalues
    ):
        powers = set(exact_powers) | set(approx_powers)
        unsupported = powers - {-1}
        if unsupported:
            raise NotImplementedError(
                f"SmallFullBlock does not cache powers {sorted(unsupported)}."
            )
        state = state.copy()
        if eigenvalues or -1 in powers:
            m = 0.5 * (state.matrix.value + state.matrix.value.T)
            w_eig, q_eig = self._safe_eigh(m)
            eig_ok = jnp.all(jnp.isfinite(w_eig)) & jnp.all(jnp.isfinite(q_eig))
            if eigenvalues:
                state.cache["eigenvalues"] = jnp.where(
                    eig_ok, w_eig, state.cache["eigenvalues"]
                )
            if -1 in powers:
                inv_eig = 1.0 / (w_eig + identity_weight)
                candidate_inverse = q_eig * inv_eig[None, :] @ q_eig.T
                inverse_ok = eig_ok & jnp.all(jnp.isfinite(candidate_inverse))
                state.cache["-1"] = jnp.where(
                    inverse_ok, candidate_inverse, state.cache["-1"]
                )
        return state

    def _to_dense_unscaled(self, state):
        return state.matrix.value

    def _norm_unscaled(self, state, norm_type):
        del norm_type
        n = self._param_size()
        return jnp.trace(state.matrix.value) / n


kfac_jax.set_default_tag_to_block_ctor(SMALL_FULL_TAG_VARIANT, SmallFullBlock)


def register_small_full(param, *, tag_id: str = ""):
    if param.size > _SMALL_FULL_MAX_SIZE:
        raise ValueError(
            f"register_small_full: param size {param.size} exceeds {_SMALL_FULL_MAX_SIZE}; use a structured block instead."
        )
    return layer_tag.bind(
        param,
        meta=LayerMetaData(
            variant=SMALL_FULL_TAG_VARIANT,
            outputs_index=(0,),
            inputs_index=(),
            params_index=(0,),
        ),
    )
