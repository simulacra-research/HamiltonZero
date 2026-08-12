# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import kfac_jax
from kfac_jax._src import utils as kfac_utils
from kfac_jax._src.curvature_blocks import utils as cb_utils
from kfac_jax._src.layers_and_loss_tags import LayerMetaData, layer_tag


STRUCTURAL_DENSE_TAG_VARIANT = "structural_repeated_dense"
STRUCTURAL_SCALE_SHIFT_TAG_VARIANT = "structural_scale_and_shift"
STRUCTURAL_STACKED_DENSE_TAG_VARIANT = "structural_stacked_repeated_dense"
STRUCTURAL_STACKED_SCALE_SHIFT_TAG_VARIANT = "structural_stacked_scale_and_shift"
STRUCTURAL_TRAILING_STACKED_SCALE_SHIFT_TAG_VARIANT = (
    "structural_trailing_stacked_scale_and_shift"
)


def _optional_name_kw(name: str | None) -> dict[str, str]:
    return {} if name is None else {"name": name}


def _validate_structural_registration(
    x,
    structural_mask,
    *,
    repeat_ndim: int,
    feature_ndim: int,
) -> None:
    if int(repeat_ndim) < 0:
        raise ValueError(f"repeat_ndim must be non-negative, got {repeat_ndim}")
    expected_mask_shape = x.shape if feature_ndim == 0 else x.shape[:-feature_ndim]
    if tuple(structural_mask.shape) != tuple(expected_mask_shape):
        raise ValueError(
            "structural_mask must exactly cover the local repeat axes: "
            f"mask={structural_mask.shape}, expected={expected_mask_shape}, "
            f"x={x.shape}, feature_ndim={feature_ndim}"
        )


def register_structural_dense(
    y,
    x,
    structural_mask,
    weight,
    bias=None,
    *,
    scan_shared: bool,
    repeat_ndim: int,
    context_primal_reused_over_walkers: bool = False,
    name: str | None = None,
):

    _validate_structural_registration(
        x,
        structural_mask,
        repeat_ndim=repeat_ndim,
        feature_ndim=1,
    )
    args = (
        (y, x, structural_mask, weight)
        if bias is None
        else (y, x, structural_mask, weight, bias)
    )
    return layer_tag.bind(
        *args,
        meta=LayerMetaData(
            variant=STRUCTURAL_DENSE_TAG_VARIANT,
            outputs_index=(0,),
            inputs_index=(1, 2),
            params_index=tuple(range(3, len(args))),
        ),
        scan_shared=bool(scan_shared),
        repeat_ndim=int(repeat_ndim),
        context_primal_reused_over_walkers=bool(context_primal_reused_over_walkers),
        **_optional_name_kw(name),
    )


def register_structural_scale_and_shift(
    y,
    x,
    structural_mask,
    scale=None,
    shift=None,
    *,
    scan_shared: bool,
    repeat_ndim: int,
    context_primal_reused_over_walkers: bool = False,
    name: str | None = None,
):

    params = tuple(value for value in (scale, shift) if value is not None)
    if not params:
        raise ValueError("At least one of scale and shift must be provided")
    feature_ndim = params[0].ndim
    if any(tuple(param.shape) != tuple(params[0].shape) for param in params[1:]):
        raise ValueError("structural scale and shift shapes must match")
    _validate_structural_registration(
        x,
        structural_mask,
        repeat_ndim=repeat_ndim,
        feature_ndim=feature_ndim,
    )
    args = (y, x, structural_mask, *params)
    return layer_tag.bind(
        *args,
        meta=LayerMetaData(
            variant=STRUCTURAL_SCALE_SHIFT_TAG_VARIANT,
            outputs_index=(0,),
            inputs_index=(1, 2),
            params_index=tuple(range(3, len(args))),
        ),
        has_scale=scale is not None,
        has_shift=shift is not None,
        scan_shared=bool(scan_shared),
        repeat_ndim=int(repeat_ndim),
        context_primal_reused_over_walkers=bool(context_primal_reused_over_walkers),
        **_optional_name_kw(name),
    )


def register_structural_trailing_stacked_scale_and_shift(
    y,
    x,
    structural_mask,
    scale,
    *,
    repeat_ndim: int,
    context_primal_reused_over_walkers: bool = False,
    name: str | None = None,
):

    if scale.ndim != 2:
        raise ValueError(
            "trailing stacked scale/shift parameters must have shape [K,d]; "
            f"got {scale.shape}"
        )
    _validate_structural_registration(
        x,
        structural_mask,
        repeat_ndim=repeat_ndim,
        feature_ndim=2,
    )
    args = (y, x, structural_mask, scale)
    return layer_tag.bind(
        *args,
        meta=LayerMetaData(
            variant=STRUCTURAL_TRAILING_STACKED_SCALE_SHIFT_TAG_VARIANT,
            outputs_index=(0,),
            inputs_index=(1, 2),
            params_index=(3,),
        ),
        has_scale=True,
        has_shift=False,
        scan_shared=False,
        repeat_ndim=int(repeat_ndim),
        context_primal_reused_over_walkers=bool(context_primal_reused_over_walkers),
        **_optional_name_kw(name),
    )


def _structural_tag_contract(layer_tag_eq):
    params = layer_tag_eq.params
    return (
        bool(params["scan_shared"]),
        int(params["repeat_ndim"]),
        bool(params.get("context_primal_reused_over_walkers", False)),
    )


def _align_structural_primal_and_mask(
    x,
    dy,
    structural_mask,
    *,
    repeat_ndim: int,
    feature_ndim: int,
    context_primal_reused_over_walkers: bool,
):

    structural_mask = jnp.asarray(structural_mask, dtype=bool)
    x_leading = tuple(x.shape[:-feature_ndim]) if feature_ndim else tuple(x.shape)
    dy_leading = tuple(dy.shape[:-feature_ndim]) if feature_ndim else tuple(dy.shape)

    def _missing_walker_axis(source_leading, target_leading, *, what):
        if source_leading == target_leading:
            return None
        if len(target_leading) != len(source_leading) + 1:
            raise ValueError(
                f"{what} supports exactly one missing walker sample axis: "
                f"source={source_leading}, target={target_leading}"
            )
        insert_axis = len(source_leading) - int(repeat_ndim)
        if insert_axis < 0 or (
            source_leading[:insert_axis] != target_leading[:insert_axis]
            or source_leading[insert_axis:] != target_leading[insert_axis + 1 :]
        ):
            raise ValueError(
                f"{what} walker axis must be the final logical-sample axis "
                f"before the {repeat_ndim} repeat axes: "
                f"source={source_leading}, target={target_leading}"
            )
        return insert_axis

    x_insert_axis = _missing_walker_axis(
        x_leading,
        dy_leading,
        what="context primal reuse",
    )
    if x_insert_axis is not None:
        if not context_primal_reused_over_walkers:
            raise ValueError(
                "x/dy structural layouts differ without "
                "context_primal_reused_over_walkers: "
                f"x={x.shape}, dy={dy.shape}, mask={structural_mask.shape}"
            )
        x = jnp.expand_dims(x, axis=x_insert_axis)
        x = jnp.broadcast_to(
            x,
            (*dy_leading, *x.shape[-feature_ndim:]) if feature_ndim else dy_leading,
        )

    structural_mask = align_structural_mask_to_leading(
        structural_mask,
        dy_leading,
        repeat_ndim=repeat_ndim,
    )
    return x, dy, structural_mask


def align_structural_mask_to_leading(
    structural_mask,
    target_leading,
    *,
    repeat_ndim: int,
):

    structural_mask = jnp.asarray(structural_mask, dtype=bool)
    source_leading = tuple(structural_mask.shape)
    target_leading = tuple(target_leading)
    if source_leading == target_leading:
        return structural_mask
    if len(target_leading) != len(source_leading) + 1:
        raise ValueError(
            "structural mask supports exactly one missing walker sample "
            f"axis: source={source_leading}, target={target_leading}"
        )
    insert_axis = len(source_leading) - int(repeat_ndim)
    if insert_axis < 0 or (
        source_leading[:insert_axis] != target_leading[:insert_axis]
        or source_leading[insert_axis:] != target_leading[insert_axis + 1 :]
    ):
        raise ValueError(
            "structural mask walker axis must be the final logical-sample "
            f"axis before the {repeat_ndim} repeat axes: "
            f"source={source_leading}, target={target_leading}"
        )
    structural_mask = jnp.expand_dims(structural_mask, axis=insert_axis)
    return jnp.broadcast_to(structural_mask, target_leading)


def structural_group_repeats(
    value,
    structural_mask,
    *,
    scan_shared: bool,
    repeat_ndim: int,
    feature_ndim: int,
):

    feature_shape = tuple(value.shape[-feature_ndim:]) if feature_ndim else ()
    leading_shape = (
        tuple(value.shape[:-feature_ndim]) if feature_ndim else tuple(value.shape)
    )
    if tuple(structural_mask.shape) != leading_shape:
        raise ValueError(
            f"mask/value leading mismatch: {structural_mask.shape} vs {leading_shape}"
        )
    scan_ndim = 1 if scan_shared else 0
    if len(leading_shape) < scan_ndim + int(repeat_ndim):
        raise ValueError(
            "not enough leading axes for structural layout: "
            f"shape={value.shape}, scan_shared={scan_shared}, "
            f"repeat_ndim={repeat_ndim}"
        )
    sample_end = len(leading_shape) - int(repeat_ndim)
    sample_axes = tuple(range(scan_ndim, sample_end))
    repeat_axes = ((0,) if scan_shared else ()) + tuple(
        range(sample_end, len(leading_shape))
    )
    feature_axes = tuple(range(len(leading_shape), value.ndim))
    permutation = (*sample_axes, *repeat_axes, *feature_axes)
    mask_permutation = (*sample_axes, *repeat_axes)
    value = jnp.transpose(value, permutation) if permutation else value
    structural_mask = (
        jnp.transpose(structural_mask, mask_permutation)
        if mask_permutation
        else structural_mask
    )
    logical_batch = int(math.prod(leading_shape[i] for i in sample_axes)) or 1
    repeats = int(math.prod(leading_shape[i] for i in repeat_axes)) or 1
    return (
        value.reshape(logical_batch, repeats, *feature_shape),
        structural_mask.reshape(logical_batch, repeats),
        logical_batch,
        repeats,
    )


def _floor_matrix_avg_diag(mat, eps: float):

    d = mat.shape[-1]
    eps_arr = jnp.asarray(eps, dtype=mat.dtype)
    avg_diag = jnp.trace(mat) / d
    shift = jnp.maximum(eps_arr, eps_arr - avg_diag)
    return mat + shift * jnp.eye(d, dtype=mat.dtype)


def _floor_diag_avg(vec, eps: float):

    eps_arr = jnp.asarray(eps, dtype=vec.dtype)
    avg_diag = jnp.mean(vec)
    shift = jnp.maximum(eps_arr, eps_arr - avg_diag)
    return vec + shift


def _iter_factor_update(raw_update, n_iter: int, eps: float, dtype):

    del eps
    if n_iter == 1:
        return jnp.ones((1, 1), dtype=dtype)
    return raw_update


def _iter_factor_for_inverse(raw_update, n_iter: int, eps: float, dtype):

    return _floor_matrix_avg_diag(
        _iter_factor_update(raw_update, n_iter, eps, dtype),
        eps,
    )


def _validate_approx_inverse_cache_request(
    exact_powers_to_cache,
    approx_powers_to_cache,
):

    if exact_powers_to_cache:
        raise NotImplementedError(
            "Custom Kronecker blocks do not implement exact cached powers."
        )
    unsupported = set(approx_powers_to_cache) - {-1}
    if unsupported:
        raise NotImplementedError(
            f"Unsupported approximate cached powers: {sorted(unsupported)}."
        )


def _identity_factor(shape, dtype):

    shape = tuple(shape)
    if len(shape) == 1:
        return jnp.ones(shape, dtype=dtype)
    if len(shape) == 2 and shape[0] == shape[1]:
        return jnp.eye(shape[0], dtype=dtype)
    raise ValueError(f"Unsupported Kronecker factor shape: {shape}.")


def _init_factor_inverse_cache(
    factor_shapes,
    dtype,
    exact_powers_to_cache,
    approx_powers_to_cache,
    cache_eigenvalues,
    eigenvalue_count,
):

    _validate_approx_inverse_cache_request(
        exact_powers_to_cache,
        approx_powers_to_cache,
    )
    cache = {}
    if -1 in approx_powers_to_cache:
        cache["-1"] = {
            f"{i}_factor": _identity_factor(shape, dtype)
            for i, shape in enumerate(factor_shapes)
        }
    if cache_eigenvalues:
        cache["eigenvalues"] = jnp.zeros((eigenvalue_count,), dtype=dtype)
    return cache


@kfac_utils.register_state_class
class _StackedRepeatedDenseState(kfac_jax.CurvatureBlock.State):
    K_iter: kfac_utils.WeightedMovingAverage
    A: kfac_utils.WeightedMovingAverage
    G: kfac_utils.WeightedMovingAverage
    average_repeats: kfac_utils.WeightedMovingAverage


class _StackedRepeatedDense(kfac_jax.CurvatureBlock):
    State = _StackedRepeatedDenseState

    _MATPOWER_EPSILON_FLOOR: float = 1e-6

    @property
    def n_iter(self) -> int:

        return int(self.parameters_shapes[0][0])

    @property
    def in_dim(self) -> int:

        wshape = tuple(self.parameters_shapes[0][1:])
        if len(wshape) == 0:
            return 1
        if len(wshape) == 1:
            return 1
        return int(math.prod(wshape[:-1]))

    @property
    def out_dim(self) -> int:

        wshape = tuple(self.parameters_shapes[0][1:])
        if len(wshape) == 0:
            return 1
        return int(wshape[-1])

    @property
    def in_dim_aug(self) -> int:

        return self.in_dim + (1 if self.number_of_parameters == 2 else 0)

    def _init(
        self,
        rng,
        exact_powers_to_cache,
        approx_powers_to_cache,
        cache_eigenvalues,
    ):
        del rng
        K = self.n_iter
        cache = _init_factor_inverse_cache(
            (
                (K, K),
                (self.in_dim_aug, self.in_dim_aug),
                (self.out_dim, self.out_dim),
            ),
            self.dtype,
            exact_powers_to_cache,
            approx_powers_to_cache,
            cache_eigenvalues,
            self.dim,
        )
        return self.State(
            cache=cache,
            K_iter=kfac_utils.WeightedMovingAverage(
                value=jnp.eye(K, dtype=self.dtype),
                weight=jnp.asarray(1.0, dtype=self.dtype),
            ),
            A=kfac_utils.WeightedMovingAverage(
                value=jnp.eye(self.in_dim_aug, dtype=self.dtype),
                weight=jnp.asarray(1.0, dtype=self.dtype),
            ),
            G=kfac_utils.WeightedMovingAverage(
                value=jnp.eye(self.out_dim, dtype=self.dtype),
                weight=jnp.asarray(1.0, dtype=self.dtype),
            ),
            average_repeats=kfac_utils.WeightedMovingAverage(
                value=jnp.ones((K,), dtype=self.dtype),
                weight=jnp.asarray(1.0, dtype=self.dtype),
            ),
        )

    def sync(self, state, pmap_axis_name):
        state = state.copy()
        state.K_iter.sync(pmap_axis_name)
        state.A.sync(pmap_axis_name)
        state.G.sync(pmap_axis_name)
        state.average_repeats.sync(pmap_axis_name)
        return state

    def _locate_iter_axis(self, arr_shape) -> int:

        n_iter = self.n_iter
        candidates = [i for i, s in enumerate(arr_shape) if s == n_iter]
        if not candidates:
            raise ValueError(
                f"{type(self).__name__}: no axis of size n_iter={n_iter} "
                f"in shape {arr_shape}. Hoist contract drifted. "
                f"parameters_shapes={self.parameters_shapes!r}"
            )
        return 0 if 0 in candidates else candidates[0]

    def _iter_axis_tensors(self, x, dy):
        ax_x = self._locate_iter_axis(x.shape)
        ax_dy = self._locate_iter_axis(dy.shape)
        return jnp.moveaxis(x, ax_x, 0), jnp.moveaxis(dy, ax_dy, 0)

    def state_dependent_scale(self, state):

        repeats = jnp.mean(state.average_repeats.value)
        return 1.0 / jnp.where(repeats > 0, repeats, 1.0)

    def _multiply_matpower_unscaled(
        self,
        state,
        vector,
        identity_weight,
        power,
        exact_power,
        use_cached,
    ):
        if exact_power and power != 1:
            raise NotImplementedError(
                "StackedRepeatedDense implements approximate inverse powers only."
            )

        grad_aug = self._params_list_to_aug_array(vector)

        if power == 1:
            factors = (
                _iter_factor_update(
                    state.K_iter.value,
                    self.n_iter,
                    self._MATPOWER_EPSILON_FLOOR,
                    self.dtype,
                ),
                state.A.value,
                state.G.value,
            )
            scale = self.state_dependent_scale(state) if use_cached else 1.0
            new_grad_aug = kfac_utils.kronecker_product_axis_mul_v(
                factors,
                grad_aug,
                axis_groups=[(0,), (1,), (2,)],
            )
            new_grad_aug = scale * new_grad_aug + identity_weight * grad_aug
        elif power == -1:
            if use_cached:
                inv_factors = tuple(state.cache["-1"][f"{i}_factor"] for i in range(3))
            else:
                eps = self._MATPOWER_EPSILON_FLOOR
                inv_factors = kfac_utils.pi_adjusted_kronecker_inverse(
                    _iter_factor_for_inverse(
                        state.K_iter.value,
                        self.n_iter,
                        eps,
                        self.dtype,
                    ),
                    _floor_matrix_avg_diag(state.A.value, eps),
                    _floor_matrix_avg_diag(state.G.value, eps),
                    damping=identity_weight,
                )
            new_grad_aug = kfac_utils.kronecker_product_axis_mul_v(
                inv_factors,
                grad_aug,
                axis_groups=[(0,), (1,), (2,)],
            )
        else:
            raise NotImplementedError(
                f"StackedRepeatedDense: power={power} not implemented "
                f"(only ±1 supported)."
            )

        return self._aug_array_to_params_list(new_grad_aug)

    def _params_list_to_aug_array(self, parameters_list):

        W = parameters_list[0]
        W_arr = W.reshape(self.n_iter, self.in_dim, self.out_dim)
        if self.number_of_parameters == 2:
            b = parameters_list[1]
            b_aug = b.reshape(self.n_iter, 1, self.out_dim)
            return jnp.concatenate([W_arr, b_aug], axis=1)
        return W_arr

    def _aug_array_to_params_list(self, arr):

        W_shape = self.parameters_shapes[0]
        W = arr[:, : self.in_dim, :].reshape(W_shape)
        if self.number_of_parameters == 2:
            b_shape = self.parameters_shapes[1]
            b = arr[:, self.in_dim :, :].reshape(b_shape)
            return [W, b]
        return [W]

    def _eigenvalues_unscaled(self, state, use_cached):
        if use_cached:
            return state.cache["eigenvalues"]
        s_K, _ = kfac_utils.safe_psd_eigh(
            _iter_factor_update(
                state.K_iter.value,
                self.n_iter,
                self._MATPOWER_EPSILON_FLOOR,
                self.dtype,
            )
        )
        s_A, _ = kfac_utils.safe_psd_eigh(state.A.value)
        s_G, _ = kfac_utils.safe_psd_eigh(state.G.value)

        return jnp.einsum("k,a,o->kao", s_K, s_A, s_G).reshape(-1)

    def _update_cache(
        self,
        state,
        identity_weight,
        exact_powers,
        approx_powers,
        eigenvalues,
    ):
        _validate_approx_inverse_cache_request(exact_powers, approx_powers)
        state = state.copy()
        eps = self._MATPOWER_EPSILON_FLOOR
        factors = (
            _iter_factor_for_inverse(
                state.K_iter.value,
                self.n_iter,
                eps,
                self.dtype,
            ),
            _floor_matrix_avg_diag(state.A.value, eps),
            _floor_matrix_avg_diag(state.G.value, eps),
        )
        scale = self.state_dependent_scale(state)

        if eigenvalues:
            state.cache["eigenvalues"] = scale * self._eigenvalues_unscaled(
                state, use_cached=False
            )

        if -1 in approx_powers:
            inv_factors = kfac_utils.pi_adjusted_kronecker_inverse(
                *factors,
                damping=identity_weight,
            )
            factor_scale = jnp.power(scale, 1.0 / len(factors))
            for i, inv_factor in enumerate(inv_factors):
                state.cache["-1"][f"{i}_factor"] = inv_factor / factor_scale

        return state

    def _to_dense_unscaled(self, state):

        F_KA = jnp.kron(
            _iter_factor_update(
                state.K_iter.value,
                self.n_iter,
                self._MATPOWER_EPSILON_FLOOR,
                self.dtype,
            ),
            state.A.value,
        )
        return jnp.kron(F_KA, state.G.value)

    def _norm_unscaled(self, state, norm_type):
        n_K = kfac_utils.psd_matrix_norm(
            _iter_factor_update(
                state.K_iter.value,
                self.n_iter,
                self._MATPOWER_EPSILON_FLOOR,
                self.dtype,
            ),
            norm_type=norm_type,
        )
        n_A = kfac_utils.psd_matrix_norm(
            state.A.value,
            norm_type=norm_type,
        )
        n_G = kfac_utils.psd_matrix_norm(
            state.G.value,
            norm_type=norm_type,
        )
        return n_K * n_A * n_G


STACKED_SCALE_SHIFT_TAG_VARIANT = "stacked_scale_and_shift"


@kfac_utils.register_state_class
class _StackedScaleAndShiftState(kfac_jax.CurvatureBlock.State):
    K_iter_factors: tuple[kfac_utils.WeightedMovingAverage, ...]
    D_shared_factors: tuple[kfac_utils.WeightedMovingAverage, ...]


class _StackedScaleAndShiftDiagonal(kfac_jax.CurvatureBlock):
    State = _StackedScaleAndShiftState
    _MATPOWER_EPSILON_FLOOR: float = 1e-6

    @property
    def n_iter(self) -> int:
        return int(self.parameters_shapes[0][0])

    @property
    def _per_iter_shapes(self) -> tuple[tuple[int, ...], ...]:

        return (tuple(self.parameters_shapes[0][1:]),)

    @property
    def _per_iter_d_flats(self) -> tuple[int, ...]:

        shape = self._per_iter_shapes[0]
        return (int(math.prod(shape)) if shape else 1,)

    def _locate_iter_axis(self, arr_shape) -> int:
        n_iter = self.n_iter
        candidates = [i for i, s in enumerate(arr_shape) if s == n_iter]
        if not candidates:
            raise ValueError(
                f"{type(self).__name__}: no axis of size n_iter={n_iter} "
                f"in shape {arr_shape}."
            )

        return 0 if 0 in candidates else candidates[0]

    def _iter_axis_tensors(self, x, dy):
        ax_x = self._locate_iter_axis(x.shape)
        ax_dy = self._locate_iter_axis(dy.shape)
        return jnp.moveaxis(x, ax_x, 0), jnp.moveaxis(dy, ax_dy, 0)

    def _init(
        self,
        rng,
        exact_powers_to_cache,
        approx_powers_to_cache,
        cache_eigenvalues,
    ):
        del rng
        K = self.n_iter
        d = self._per_iter_d_flats[0]
        cache = _init_factor_inverse_cache(
            ((K, K), (d,)),
            self.dtype,
            exact_powers_to_cache,
            approx_powers_to_cache,
            cache_eigenvalues,
            self.dim,
        )
        return self.State(
            cache=cache,
            K_iter_factors=(
                kfac_utils.WeightedMovingAverage(
                    value=jnp.eye(K, dtype=self.dtype),
                    weight=jnp.asarray(1.0, dtype=self.dtype),
                ),
            ),
            D_shared_factors=(
                kfac_utils.WeightedMovingAverage(
                    value=jnp.ones((d,), dtype=self.dtype),
                    weight=jnp.asarray(1.0, dtype=self.dtype),
                ),
            ),
        )

    def sync(self, state, pmap_axis_name):
        state = state.copy()
        state.K_iter_factors[0].sync(pmap_axis_name)
        state.D_shared_factors[0].sync(pmap_axis_name)
        return state

    @kfac_utils.auto_scope_method
    def update_curvature_matrix_estimate(
        self,
        state,
        estimation_data,
        ema_old,
        ema_new,
        identity_weight,
        batch_size,
    ):
        del identity_weight, batch_size
        state = state.copy()
        [x] = estimation_data.primals.inputs
        [dy] = estimation_data.tangents.outputs
        x_iter, dy_iter = self._iter_axis_tensors(x, dy)
        mask_iter = 1.0 - jnp.all(
            dy_iter == 0.0,
            axis=-1,
            keepdims=True,
        )
        n_iter = self.n_iter
        per_iter_shape = self._per_iter_shapes[0]
        d_flat = self._per_iter_d_flats[0]

        def _per_iter(arr_i):
            return cb_utils.compatible_sum(
                arr_i,
                per_iter_shape,
                skip_axes=[0],
            )

        d_grad = jax.vmap(_per_iter)(x_iter * dy_iter).reshape(
            n_iter,
            -1,
            d_flat,
        )
        mask = jnp.any(
            mask_iter.reshape(n_iter, mask_iter.shape[1], -1) > 0,
            axis=-1,
        ).astype(self.dtype)
        d_grad = d_grad * mask[..., None]
        n_active = jnp.maximum(jnp.sum(mask), 1.0).astype(self.dtype)
        D_update = jnp.einsum("kbi,kbi->i", d_grad, d_grad) / n_active
        if n_iter == 1:
            K_update = jnp.ones((1, 1), dtype=self.dtype)
        else:
            weighted = d_grad * jnp.sqrt(jnp.maximum(D_update, 0.0))[None, None, :]
            numerator = jnp.einsum("kbi,lbi->kl", weighted, weighted)
            per_iter_active = jnp.sum(mask, axis=-1)
            active_norm = jnp.sqrt(
                jnp.maximum(
                    per_iter_active[:, None] * per_iter_active[None, :],
                    1.0,
                )
            ).astype(self.dtype)
            D_frob2 = jnp.maximum(jnp.sum(D_update * D_update), 1e-12)
            K_update = numerator / (active_norm * D_frob2)
            K_update = _iter_factor_update(
                0.5 * (K_update + K_update.T),
                n_iter,
                self._MATPOWER_EPSILON_FLOOR,
                self.dtype,
            )
        state.K_iter_factors[0].update(K_update, ema_old, ema_new)
        state.D_shared_factors[0].update(D_update, ema_old, ema_new)
        return state

    def _multiply_matpower_unscaled(
        self,
        state,
        vector,
        identity_weight,
        power,
        exact_power,
        use_cached,
    ):
        if exact_power and power != 1:
            raise NotImplementedError(
                "StackedScaleAndShiftDiagonal implements approximate "
                "inverse powers only."
            )
        n_iter = self.n_iter
        v = vector[0]
        v_flat = v.reshape(n_iter, -1)
        if power == -1 and use_cached:
            K_iter_inv = state.cache["-1"]["0_factor"]
            D_shared_inv = state.cache["-1"]["1_factor"]
            Kv = jnp.einsum("kl,li->ki", K_iter_inv, v_flat)
            result_flat = D_shared_inv[None, :] * Kv
        elif power == 1:
            K_factor = _iter_factor_update(
                state.K_iter_factors[0].value,
                n_iter,
                self._MATPOWER_EPSILON_FLOOR,
                self.dtype,
            )
            D_factor = state.D_shared_factors[0].value
            Kv = jnp.einsum("kl,li->ki", K_factor, v_flat)
            result_flat = D_factor[None, :] * Kv + identity_weight * v_flat
        elif power == -1:
            eps = self._MATPOWER_EPSILON_FLOOR
            K_floored = _iter_factor_for_inverse(
                state.K_iter_factors[0].value,
                n_iter,
                eps,
                self.dtype,
            )
            D_floored = _floor_diag_avg(
                state.D_shared_factors[0].value,
                eps,
            )
            shrink = jnp.maximum(
                1.0,
                jnp.mean(D_floored) / identity_weight,
            )
            D_floored = D_floored / shrink
            K_iter_inv, D_shared_inv = kfac_utils.pi_adjusted_kronecker_inverse(
                K_floored,
                D_floored,
                damping=identity_weight,
            )
            Kv = jnp.einsum("kl,li->ki", K_iter_inv, v_flat)
            result_flat = D_shared_inv[None, :] * Kv
        else:
            raise NotImplementedError(
                f"StackedScaleAndShiftDiagonal: power={power} not "
                f"implemented (only ±1 supported)."
            )
        return (result_flat.reshape(v.shape),)

    def _eigenvalues_unscaled(self, state, use_cached):
        if use_cached:
            return state.cache["eigenvalues"]
        s_K, _ = kfac_utils.safe_psd_eigh(
            _iter_factor_update(
                state.K_iter_factors[0].value,
                self.n_iter,
                self._MATPOWER_EPSILON_FLOOR,
                self.dtype,
            )
        )
        return jnp.einsum(
            "k,i->ki",
            s_K,
            state.D_shared_factors[0].value,
        ).reshape(-1)

    def _update_cache(
        self,
        state,
        identity_weight,
        exact_powers,
        approx_powers,
        eigenvalues,
    ):
        _validate_approx_inverse_cache_request(exact_powers, approx_powers)
        state = state.copy()

        if eigenvalues:
            state.cache["eigenvalues"] = self._eigenvalues_unscaled(
                state,
                use_cached=False,
            )

        if -1 in approx_powers:
            eps = self._MATPOWER_EPSILON_FLOOR
            K_floored = _iter_factor_for_inverse(
                state.K_iter_factors[0].value,
                self.n_iter,
                eps,
                self.dtype,
            )
            D_floored = _floor_diag_avg(
                state.D_shared_factors[0].value,
                eps,
            )
            shrink = jnp.maximum(
                1.0,
                jnp.mean(D_floored) / identity_weight,
            )
            D_floored = D_floored / shrink
            K_inv, D_inv = kfac_utils.pi_adjusted_kronecker_inverse(
                K_floored,
                D_floored,
                damping=identity_weight,
            )
            state.cache["-1"]["0_factor"] = K_inv
            state.cache["-1"]["1_factor"] = D_inv

        return state

    def _to_dense_unscaled(self, state):
        return jnp.kron(
            _iter_factor_update(
                state.K_iter_factors[0].value,
                self.n_iter,
                self._MATPOWER_EPSILON_FLOOR,
                self.dtype,
            ),
            jnp.diag(state.D_shared_factors[0].value),
        )

    def _norm_unscaled(self, state, norm_type):
        if norm_type in ("trace", "avg_diag"):
            component_norm = "trace"
        elif norm_type in ("fro", "avg_fro"):
            component_norm = "fro"
        else:
            component_norm = norm_type

        K = _iter_factor_update(
            state.K_iter_factors[0].value,
            self.n_iter,
            self._MATPOWER_EPSILON_FLOOR,
            self.dtype,
        )
        D = state.D_shared_factors[0].value
        if component_norm == "trace":
            norm = jnp.trace(K) * jnp.sum(D)
        elif component_norm == "fro":
            norm = jnp.linalg.norm(K) * jnp.linalg.norm(D)
        elif component_norm == "2_norm":
            norm = jnp.max(jnp.linalg.eigvalsh(K)) * jnp.max(D)
        elif component_norm == "1_norm":
            norm = jnp.max(jnp.sum(jnp.abs(K), axis=0)) * jnp.max(jnp.abs(D))
        elif component_norm == "one_over_dim":
            norm = jnp.asarray(1.0, dtype=self.dtype)
        else:
            raise NotImplementedError(
                f"Kronecker norm {norm_type!r} is not needed by KFAC stats"
            )
        total_dim = self.n_iter * self._per_iter_d_flats[0]
        if norm_type == "trace":
            return norm
        if norm_type == "avg_diag":
            return norm / total_dim
        if norm_type == "one_over_dim":
            return jnp.asarray(1.0 / total_dim, dtype=self.dtype)
        if norm_type in ("2_norm", "1_norm"):
            return norm
        if norm_type in ("fro", "avg_fro"):
            return norm if norm_type == "fro" else norm / jnp.sqrt(total_dim)
        raise NotImplementedError(
            f"direct-sum norm {norm_type!r} is not needed by KFAC stats"
        )


class _ScaleAndShiftDiagonal(kfac_jax.ScaleAndShiftDiagonal):
    @kfac_utils.auto_scope_method
    def update_curvature_matrix_estimate(
        self,
        state,
        estimation_data,
        ema_old,
        ema_new,
        identity_weight,
        batch_size,
    ):
        del identity_weight, batch_size
        state = state.copy()
        [x] = estimation_data.primals.inputs
        [dy] = estimation_data.tangents.outputs
        mask = 1.0 - jnp.all(dy == 0.0, axis=-1, keepdims=True)
        x_masked = x * mask
        n_active = jnp.maximum(jnp.sum(mask), 1.0).astype(x.dtype)
        if self.has_scale:
            scale_shape = estimation_data.primals.params[0].shape
            n_param_dims = len(scale_shape)
            x_flat = x_masked.reshape(
                (-1,) + tuple(x_masked.shape[-n_param_dims:]) if n_param_dims else (-1,)
            )
            dy_flat = dy.reshape(
                (-1,) + tuple(dy.shape[-n_param_dims:]) if n_param_dims else (-1,)
            )
            d_scale = cb_utils.compatible_sum(
                x_flat * dy_flat,
                scale_shape,
                skip_axes=[0],
            )
            scale_diag_update = (
                jnp.sum(
                    d_scale * d_scale,
                    axis=0,
                    keepdims=d_scale.ndim == len(scale_shape),
                )
                / n_active
            )
            state.diagonal_factors[0].update(
                scale_diag_update,
                ema_old,
                ema_new,
            )
        if self.has_shift:
            shift_shape = estimation_data.primals.params[-1].shape
            n_param_dims = len(shift_shape)
            dy_flat = dy.reshape(
                (-1,) + tuple(dy.shape[-n_param_dims:]) if n_param_dims else (-1,)
            )
            d_shift = cb_utils.compatible_sum(
                dy_flat,
                shift_shape,
                skip_axes=[0],
            )
            shift_diag_update = (
                jnp.sum(
                    d_shift * d_shift,
                    axis=0,
                    keepdims=d_shift.ndim == len(shift_shape),
                )
                / n_active
            )
            state.diagonal_factors[-1].update(
                shift_diag_update,
                ema_old,
                ema_new,
            )
        return state

    def _norm_unscaled(self, state, norm_type):
        diagonal = jnp.concatenate(
            [factor.value.flatten() for factor in state.diagonal_factors],
            axis=0,
        )
        return kfac_utils.psd_matrix_norm(
            diagonal,
            norm_type=norm_type,
        )

    def _multiply_matpower_unscaled(
        self,
        state,
        vector,
        identity_weight,
        power,
        exact_power,
        use_cached,
    ):
        scale = self.state_dependent_scale(state) if use_cached else 1.0
        factors = []
        for diagonal_factor in state.diagonal_factors:
            value = scale * diagonal_factor.value
            shrink = jnp.maximum(1.0, jnp.mean(value) / identity_weight)
            factors.append(value / shrink + identity_weight)
        assert len(factors) == len(vector)
        if power == 1:
            return tuple(factor * value for factor, value in zip(factors, vector))
        elif power == -1:
            return tuple(value / factor for factor, value in zip(factors, vector))
        return tuple(
            jnp.power(factor, power) * value for factor, value in zip(factors, vector)
        )


class StructuralRepeatedDenseKroneckerFactored(
    kfac_jax.RepeatedDenseKroneckerFactored,
):
    def state_dependent_scale(self, state):
        repeats = state.average_repeats.value
        return 1.0 / jnp.where(repeats > 0, repeats, 1.0)

    @kfac_utils.auto_scope_method
    def update_curvature_matrix_estimate(
        self,
        state,
        estimation_data,
        ema_old,
        ema_new,
        identity_weight,
        batch_size,
    ):
        del identity_weight
        state = state.copy()
        x, structural_mask = estimation_data.primals.inputs
        [dy] = estimation_data.tangents.outputs
        scan_shared, repeat_ndim, context_reuse = _structural_tag_contract(
            self._layer_tag_eq
        )
        try:
            x, dy, structural_mask = _align_structural_primal_and_mask(
                x,
                dy,
                structural_mask,
                repeat_ndim=repeat_ndim,
                feature_ndim=1,
                context_primal_reused_over_walkers=context_reuse,
            )
        except ValueError as error:
            meta = self._layer_tag_eq.params.get("meta")
            raise ValueError(
                f"{error}; structural dense tag="
                f"{getattr(meta, 'name', None)!r}, scan_shared={scan_shared}, "
                f"repeat_ndim={repeat_ndim}, context_reuse={context_reuse}, "
                f"x={x.shape}, dy={dy.shape}, mask={structural_mask.shape}"
            ) from error
        xg, mg, logical_batch, _ = structural_group_repeats(
            x,
            structural_mask,
            scan_shared=scan_shared,
            repeat_ndim=repeat_ndim,
            feature_ndim=1,
        )
        dyg, _, dy_batch, _ = structural_group_repeats(
            dy,
            structural_mask,
            scan_shared=scan_shared,
            repeat_ndim=repeat_ndim,
            feature_ndim=1,
        )
        if logical_batch != dy_batch:
            raise ValueError(
                f"x/dy logical batch mismatch: {logical_batch} vs {dy_batch}"
            )
        mask = mg.astype(xg.dtype)[..., None]
        xg = xg * mask
        dyg = dyg * mask.astype(dyg.dtype)
        x_flat = xg.reshape((-1, xg.shape[-1]))
        dy_flat = dyg.reshape((-1, dyg.shape[-1]))
        if self.number_of_parameters == 2:
            x_flat = jnp.concatenate(
                [x_flat, mask.reshape((-1, 1))],
                axis=-1,
            )
        logical_divisor = jnp.asarray(logical_batch, dtype=x_flat.dtype)
        global_divisor = jnp.asarray(batch_size, dtype=dy_flat.dtype)
        input_stats = jnp.einsum("ai,aj->ij", x_flat, x_flat) / logical_divisor
        output_stats = jnp.einsum("ao,ap->op", dy_flat, dy_flat) / global_divisor
        average_repeats = jnp.sum(mask) / logical_divisor
        state.factors[0].update(input_stats, ema_old, ema_new)
        state.factors[1].update(output_stats, ema_old, ema_new)

        state.average_repeats.update(
            average_repeats,
            ema_old,
            ema_new,
        )
        return state


class StructuralScaleAndShiftDiagonal(_ScaleAndShiftDiagonal):
    @kfac_utils.auto_scope_method
    def update_curvature_matrix_estimate(
        self,
        state,
        estimation_data,
        ema_old,
        ema_new,
        identity_weight,
        batch_size,
    ):
        del identity_weight
        state = state.copy()
        x, structural_mask = estimation_data.primals.inputs
        [dy] = estimation_data.tangents.outputs
        scan_shared, repeat_ndim, context_reuse = _structural_tag_contract(
            self._layer_tag_eq
        )
        reference_param = estimation_data.primals.params[0]
        feature_ndim = reference_param.ndim
        try:
            x, dy, structural_mask = _align_structural_primal_and_mask(
                x,
                dy,
                structural_mask,
                repeat_ndim=repeat_ndim,
                feature_ndim=feature_ndim,
                context_primal_reused_over_walkers=context_reuse,
            )
        except ValueError as error:
            meta = self._layer_tag_eq.params.get("meta")
            raise ValueError(
                f"{error}; structural scale/shift tag="
                f"{getattr(meta, 'name', None)!r}, scan_shared={scan_shared}, "
                f"repeat_ndim={repeat_ndim}, context_reuse={context_reuse}, "
                f"x={x.shape}, dy={dy.shape}, mask={structural_mask.shape}"
            ) from error
        xg, mg, logical_batch, _ = structural_group_repeats(
            x,
            structural_mask,
            scan_shared=scan_shared,
            repeat_ndim=repeat_ndim,
            feature_ndim=feature_ndim,
        )
        dyg, _, dy_batch, _ = structural_group_repeats(
            dy,
            structural_mask,
            scan_shared=scan_shared,
            repeat_ndim=repeat_ndim,
            feature_ndim=feature_ndim,
        )
        if logical_batch != dy_batch:
            raise ValueError(
                f"x/dy logical batch mismatch: {logical_batch} vs {dy_batch}"
            )
        mask = mg.astype(dyg.dtype)
        mask = mask.reshape((*mask.shape, *(1,) * feature_ndim))
        xg = xg * mask.astype(xg.dtype)
        dyg = dyg * mask
        divisor = jnp.asarray(batch_size, dtype=dyg.dtype)
        param_index = 0
        if self.has_scale:
            d_scale = jnp.sum(xg * dyg, axis=1)
            scale_update = jnp.sum(d_scale * d_scale, axis=0) / divisor
            state.diagonal_factors[param_index].update(
                scale_update,
                ema_old,
                ema_new,
            )
            param_index += 1
        if self.has_shift:
            d_shift = jnp.sum(dyg, axis=1)
            shift_update = jnp.sum(d_shift * d_shift, axis=0) / divisor
            state.diagonal_factors[param_index].update(
                shift_update,
                ema_old,
                ema_new,
            )
        return state


class StructuralStackedRepeatedDense(_StackedRepeatedDense):
    def state_dependent_scale(self, state):
        repeats = jnp.mean(state.average_repeats.value)
        return 1.0 / jnp.where(repeats > 0, repeats, 1.0)

    @kfac_utils.auto_scope_method
    def update_curvature_matrix_estimate(
        self,
        state,
        estimation_data,
        ema_old,
        ema_new,
        identity_weight,
        batch_size,
    ):
        del identity_weight
        state = state.copy()
        x, structural_mask = estimation_data.primals.inputs
        [dy] = estimation_data.tangents.outputs
        scan_shared, repeat_ndim, context_reuse = _structural_tag_contract(
            self._layer_tag_eq
        )
        x, dy, structural_mask = _align_structural_primal_and_mask(
            x,
            dy,
            structural_mask,
            repeat_ndim=repeat_ndim,
            feature_ndim=1,
            context_primal_reused_over_walkers=context_reuse,
        )
        ax_x = self._locate_iter_axis(x.shape)
        ax_dy = self._locate_iter_axis(dy.shape)
        ax_mask = self._locate_iter_axis(structural_mask.shape)
        x_iter = jnp.moveaxis(x, ax_x, 0)
        dy_iter = jnp.moveaxis(dy, ax_dy, 0)
        mask_iter = jnp.moveaxis(structural_mask, ax_mask, 0)

        x_groups = []
        dy_groups = []
        mask_groups = []
        logical_batch = None
        for k in range(self.n_iter):
            xg, mg, B, _ = structural_group_repeats(
                x_iter[k],
                mask_iter[k],
                scan_shared=scan_shared,
                repeat_ndim=repeat_ndim,
                feature_ndim=1,
            )
            dyg, _, B_dy, _ = structural_group_repeats(
                dy_iter[k],
                mask_iter[k],
                scan_shared=scan_shared,
                repeat_ndim=repeat_ndim,
                feature_ndim=1,
            )
            if B != B_dy or (logical_batch is not None and B != logical_batch):
                raise ValueError("stacked structural logical batches differ")
            logical_batch = B
            x_groups.append(xg)
            dy_groups.append(dyg)
            mask_groups.append(mg)
        x_group = jnp.stack(x_groups, axis=0)
        dy_group = jnp.stack(dy_groups, axis=0)
        mask_group = jnp.stack(mask_groups, axis=0).astype(x_group.dtype)
        x_group = x_group * mask_group[..., None]
        dy_group = dy_group * mask_group[..., None].astype(dy_group.dtype)
        if self.number_of_parameters == 2:
            x_group = jnp.concatenate(
                [x_group, mask_group[..., None]],
                axis=-1,
            )
        K = self.n_iter
        logical_batch = int(logical_batch or 1)
        logical_divisor = jnp.asarray(K * logical_batch, dtype=self.dtype)
        global_divisor = jnp.asarray(K * batch_size, dtype=self.dtype)
        x_flat = x_group.reshape(K, -1, x_group.shape[-1])
        dy_flat = dy_group.reshape(K, -1, dy_group.shape[-1])
        A_update = jnp.einsum("kbi,kbj->ij", x_flat, x_flat) / logical_divisor
        G_update = jnp.einsum("kbo,kbp->op", dy_flat, dy_flat) / global_divisor
        per_iter_active = jnp.sum(mask_group, axis=(1, 2))
        if K == 1:
            K_iter_update = jnp.ones((1, 1), dtype=self.dtype)
        else:
            if context_reuse:
                A_projection = state.A.value
                G_projection = state.G.value
            else:
                A_projection = A_update
                G_projection = G_update
            xA = jnp.einsum("kbi,ij->kbj", x_flat, A_projection)
            dyG = jnp.einsum("kbo,op->kbp", dy_flat, G_projection)
            numerator = jnp.einsum(
                "klb,klb->kl",
                jnp.einsum("kbi,lbi->klb", xA, x_flat),
                jnp.einsum("kbo,lbo->klb", dyG, dy_flat),
            )
            mean_repeats = jnp.mean(per_iter_active) / jnp.asarray(
                logical_batch, self.dtype
            )
            denominator = (
                jnp.asarray(batch_size, self.dtype)
                * jnp.maximum(
                    jnp.sum(A_projection * A_projection),
                    1e-12,
                )
                * jnp.maximum(
                    jnp.sum(G_projection * G_projection),
                    1e-12,
                )
            )
            K_iter_update = mean_repeats * numerator / denominator
            K_iter_update = _iter_factor_update(
                0.5 * (K_iter_update + K_iter_update.T),
                K,
                self._MATPOWER_EPSILON_FLOOR,
                self.dtype,
            )
        state.K_iter.update(K_iter_update, ema_old, ema_new)
        state.A.update(A_update, ema_old, ema_new)
        state.G.update(G_update, ema_old, ema_new)
        state.average_repeats.update(
            per_iter_active / jnp.asarray(logical_batch, self.dtype),
            ema_old,
            ema_new,
        )
        return state


class StructuralStackedScaleAndShiftDiagonal(_StackedScaleAndShiftDiagonal):
    def _structural_iter_axis(self, shape) -> int:
        if shape and int(shape[0]) == self.n_iter:
            return 0
        return self._locate_iter_axis(shape)

    @kfac_utils.auto_scope_method
    def update_curvature_matrix_estimate(
        self,
        state,
        estimation_data,
        ema_old,
        ema_new,
        identity_weight,
        batch_size,
    ):
        del identity_weight
        state = state.copy()
        x, structural_mask = estimation_data.primals.inputs
        [dy] = estimation_data.tangents.outputs
        scan_shared, repeat_ndim, context_reuse = _structural_tag_contract(
            self._layer_tag_eq
        )

        feature_ndim = len(self._per_iter_shapes[0])
        x, dy, structural_mask = _align_structural_primal_and_mask(
            x,
            dy,
            structural_mask,
            repeat_ndim=repeat_ndim,
            feature_ndim=feature_ndim,
            context_primal_reused_over_walkers=context_reuse,
        )
        x_iter = jnp.moveaxis(x, self._structural_iter_axis(x.shape), 0)
        dy_iter = jnp.moveaxis(dy, self._structural_iter_axis(dy.shape), 0)
        mask_iter = jnp.moveaxis(
            structural_mask,
            self._structural_iter_axis(structural_mask.shape),
            0,
        )
        self._update_structural_scale(
            state,
            x_iter,
            dy_iter,
            mask_iter,
            self._per_iter_shapes[0],
            scan_shared,
            repeat_ndim,
            context_reuse,
            batch_size,
            ema_old,
            ema_new,
        )
        return state

    def _update_structural_scale(
        self,
        state,
        x_iter,
        dy_iter,
        mask_iter,
        per_iter_shape,
        scan_shared,
        repeat_ndim,
        context_reuse,
        batch_size,
        ema_old,
        ema_new,
    ):
        K = self.n_iter
        feature_ndim = len(per_iter_shape)
        grads = []
        logical_batch = None
        for k in range(K):
            xg, mg, B, _ = structural_group_repeats(
                x_iter[k],
                mask_iter[k],
                scan_shared=scan_shared,
                repeat_ndim=repeat_ndim,
                feature_ndim=feature_ndim,
            )
            dyg, _, B_dy, _ = structural_group_repeats(
                dy_iter[k],
                mask_iter[k],
                scan_shared=scan_shared,
                repeat_ndim=repeat_ndim,
                feature_ndim=feature_ndim,
            )
            if B != B_dy or (logical_batch is not None and B != logical_batch):
                raise ValueError("stacked scale logical batches differ")
            logical_batch = B
            mask = mg.astype(dyg.dtype).reshape((*mg.shape, *(1,) * feature_ndim))
            row_grad = xg * dyg * mask
            grads.append(jnp.sum(row_grad, axis=1).reshape(B, -1))
        grad = jnp.stack(grads, axis=0)
        logical_batch = int(logical_batch or 1)
        D_update = jnp.einsum("kbi,kbi->i", grad, grad) / jnp.asarray(
            K * batch_size,
            self.dtype,
        )
        if K == 1:
            K_update = jnp.ones((1, 1), dtype=self.dtype)
        else:
            D_projection = (
                state.D_shared_factors[0].value if context_reuse else D_update
            )
            weighted = grad * jnp.sqrt(jnp.maximum(D_projection, 0.0))[None, None, :]
            numerator = jnp.einsum("kbi,lbi->kl", weighted, weighted)
            denom = jnp.asarray(batch_size, self.dtype) * jnp.maximum(
                jnp.sum(D_projection * D_projection),
                1e-12,
            )
            K_update = _iter_factor_update(
                0.5 * (numerator / denom + (numerator / denom).T),
                K,
                self._MATPOWER_EPSILON_FLOOR,
                self.dtype,
            )
        state.K_iter_factors[0].update(K_update, ema_old, ema_new)
        state.D_shared_factors[0].update(D_update, ema_old, ema_new)


class StructuralTrailingStackedScaleAndShiftDiagonal(
    StructuralStackedScaleAndShiftDiagonal,
):
    @kfac_utils.auto_scope_method
    def update_curvature_matrix_estimate(
        self,
        state,
        estimation_data,
        ema_old,
        ema_new,
        identity_weight,
        batch_size,
    ):
        del identity_weight
        state = state.copy()
        x, structural_mask = estimation_data.primals.inputs
        [dy] = estimation_data.tangents.outputs
        _, repeat_ndim, context_reuse = _structural_tag_contract(self._layer_tag_eq)
        x, dy, structural_mask = _align_structural_primal_and_mask(
            x,
            dy,
            structural_mask,
            repeat_ndim=repeat_ndim,
            feature_ndim=2,
            context_primal_reused_over_walkers=context_reuse,
        )
        K = self.n_iter
        if int(x.shape[-2]) != K or int(dy.shape[-2]) != K:
            raise ValueError(
                f"{type(self).__name__}: expected trailing K={K} axis, "
                f"got x={x.shape}, dy={dy.shape}"
            )
        x_iter = jnp.moveaxis(x, -2, 0)
        dy_iter = jnp.moveaxis(dy, -2, 0)
        mask_with_groups = jnp.broadcast_to(
            structural_mask[..., None],
            x.shape[:-1],
        )
        mask_iter = jnp.moveaxis(mask_with_groups, -1, 0)

        self._update_structural_scale(
            state,
            x_iter,
            dy_iter,
            mask_iter,
            self._per_iter_shapes[0],
            False,
            repeat_ndim,
            context_reuse,
            batch_size,
            ema_old,
            ema_new,
        )
        return state


class _DenseBlock(kfac_jax.DenseTwoKroneckerFactored):
    def update_curvature_matrix_estimate(
        self,
        state,
        estimation_data,
        ema_old,
        ema_new,
        identity_weight,
        batch_size,
    ):
        del identity_weight
        state = state.copy()
        [x] = estimation_data.primals.inputs
        [dy] = estimation_data.tangents.outputs
        if not kfac_jax.utils.first_dim_is_size(batch_size, x, dy):
            x, dy = (
                jnp.tile(a[None], (batch_size, *(1 for _ in a.shape))).reshape(
                    (-1, a.shape[-1])
                )
                for a in (x, dy)
            )
            batch_size = x.size // x.shape[-1]
        assert kfac_jax.utils.first_dim_is_size(batch_size, x, dy)
        mask = 1.0 - jnp.all(dy == 0.0, axis=-1, keepdims=True)
        x = x * mask
        n_active = jnp.maximum(jnp.sum(mask), 1.0).astype(x.dtype)
        x = x.reshape((-1, x.shape[-1]))
        dy = dy.reshape((-1, dy.shape[-1]))
        input_stats = jnp.einsum("ay,az->yz", x, x) / n_active
        output_stats = jnp.einsum("ay,az->yz", dy, dy) / n_active
        state.factors[0].update(input_stats, ema_old, ema_new)
        state.factors[1].update(output_stats, ema_old, ema_new)
        return state


kfac_jax.set_default_tag_to_block_ctor("dense", _DenseBlock)
kfac_jax.set_default_tag_to_block_ctor(
    "scale_and_shift",
    _ScaleAndShiftDiagonal,
)
kfac_jax.set_default_tag_to_block_ctor(
    STACKED_SCALE_SHIFT_TAG_VARIANT,
    _StackedScaleAndShiftDiagonal,
)
kfac_jax.set_default_tag_to_block_ctor(
    STRUCTURAL_DENSE_TAG_VARIANT,
    StructuralRepeatedDenseKroneckerFactored,
)
kfac_jax.set_default_tag_to_block_ctor(
    STRUCTURAL_SCALE_SHIFT_TAG_VARIANT,
    StructuralScaleAndShiftDiagonal,
)
kfac_jax.set_default_tag_to_block_ctor(
    STRUCTURAL_STACKED_DENSE_TAG_VARIANT,
    StructuralStackedRepeatedDense,
)
kfac_jax.set_default_tag_to_block_ctor(
    STRUCTURAL_STACKED_SCALE_SHIFT_TAG_VARIANT,
    StructuralStackedScaleAndShiftDiagonal,
)
kfac_jax.set_default_tag_to_block_ctor(
    STRUCTURAL_TRAILING_STACKED_SCALE_SHIFT_TAG_VARIANT,
    StructuralTrailingStackedScaleAndShiftDiagonal,
)


def make_graph_patterns():
    return ()


__all__ = [
    "STRUCTURAL_DENSE_TAG_VARIANT",
    "STRUCTURAL_SCALE_SHIFT_TAG_VARIANT",
    "STRUCTURAL_STACKED_DENSE_TAG_VARIANT",
    "STRUCTURAL_STACKED_SCALE_SHIFT_TAG_VARIANT",
    "STRUCTURAL_TRAILING_STACKED_SCALE_SHIFT_TAG_VARIANT",
    "StructuralRepeatedDenseKroneckerFactored",
    "StructuralScaleAndShiftDiagonal",
    "StructuralStackedRepeatedDense",
    "StructuralStackedScaleAndShiftDiagonal",
    "StructuralTrailingStackedScaleAndShiftDiagonal",
    "make_graph_patterns",
    "register_structural_dense",
    "register_structural_scale_and_shift",
    "register_structural_trailing_stacked_scale_and_shift",
    "structural_group_repeats",
]
