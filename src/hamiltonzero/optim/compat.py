# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations


def _scan_partition_sizes(eqn) -> tuple[int, int, int]:

    consts, carry, xs = eqn.params["ft_in"].unpack()
    return len(consts), len(carry), len(xs)


def _extend_scan_flat_trees(
    params: dict,
    *,
    extra_xs: int,
    extra_ys: int,
) -> None:

    from jax._src import flattree as _ft

    consts, carry, xs = params["ft_in"].unpack()
    carry_out, ys = params["ft_out"].unpack()
    if extra_xs:
        xs = _ft.pack((xs, _ft.nones(extra_xs)))
    if extra_ys:
        ys = _ft.pack((ys, _ft.nones(extra_ys)))
    params["ft_in"] = _ft.pack((consts, carry, xs))
    params["ft_out"] = _ft.pack((carry_out, ys))


def _patch() -> None:

    from jax._src import source_info_util as jex_source_info_util
    from kfac_jax._src import tag_graph_matcher as tgm

    if getattr(tgm.eval_jaxpr_eqn, "__hamiltonzero_patched__", False):
        return

    def eval_jaxpr_eqn(eqn, in_values):
        bind_params = eqn.primitive.get_bind_params(eqn.params)
        user_context = jex_source_info_util.user_context
        with user_context(eqn.source_info.traceback):
            output = eqn.primitive.bind(*in_values, **bind_params)
        return [output] if not isinstance(output, list) else output

    eval_jaxpr_eqn.__hamiltonzero_patched__ = True
    tgm.eval_jaxpr_eqn = eval_jaxpr_eqn


def _patch_allow_multiple_registrations() -> None:

    import threading

    from kfac_jax._src import tag_graph_matcher as tgm

    if getattr(tgm.auto_register_tags, "__hamiltonzero_allow_multi__", False):
        return

    _orig_auto = tgm.auto_register_tags
    _orig_check = tgm.TaggedFunction.check_multiple_registrations
    _state = threading.local()

    def auto_register_tags(
        func,
        func_args,
        *,
        allow_multiple_registrations: bool = False,
        **kwargs,
    ):
        prev = getattr(_state, "allow", False)
        _state.allow = bool(allow_multiple_registrations)
        try:
            return _orig_auto(func, func_args, **kwargs)
        finally:
            _state.allow = prev

    def check_multiple_registrations(self):
        if getattr(_state, "allow", False):
            return
        return _orig_check(self)

    auto_register_tags.__hamiltonzero_allow_multi__ = True
    tgm.auto_register_tags = auto_register_tags
    tgm.TaggedFunction.check_multiple_registrations = check_multiple_registrations


def _patch_orphan_registration_in_sub_graphs() -> None:

    from kfac_jax._src import tag_graph_matcher as tgm

    if getattr(tgm._auto_register_tags, "__hamiltonzero_orphan_subgraph__", False):
        return

    _orig = tgm._auto_register_tags

    def _patched(graph, *args, register_orphans=True, **kwargs):

        return _orig(graph, *args, register_orphans=True, **kwargs)

    _patched.__hamiltonzero_orphan_subgraph__ = True
    tgm._auto_register_tags = _patched


def _patch_manual_tag_outputs_that_are_graph_inputs() -> None:

    from kfac_jax._src import tag_graph_matcher as tgm
    import jax.extend as jex

    graph_cls = tgm.JaxprGraph
    if getattr(
        graph_cls.sub_graph_eqns,
        "__hamiltonzero_graph_input_tag_output__",
        False,
    ):
        return
    original = graph_cls.sub_graph_eqns

    def sub_graph_eqns(self, root_vars, leaf_vars):
        kept = []
        for value in leaf_vars:
            if (
                isinstance(value, jex.core.Literal)
                or value in self.params_vars
                or value in self.var_to_creation_op
            ):
                kept.append(value)
            elif value in self.jaxpr.invars:
                continue
            else:
                raise KeyError(value)
        return original(self, root_vars, tuple(kept))

    sub_graph_eqns.__hamiltonzero_graph_input_tag_output__ = True
    graph_cls.sub_graph_eqns = sub_graph_eqns


def _patch_hoist_layer_tags_from_scan() -> None:

    from kfac_jax._src import tag_graph_matcher as tgm
    from kfac_jax._src import layers_and_loss_tags as tags
    import jax
    from jax._src import core as _jcore
    from kfac_jax._src.tag_graph_matcher import (
        ClosedJaxpr,
        HIGHER_ORDER_NAMES,
        to_closed_jaxpr,
        to_jaxpr_or_closed_jaxpr,
    )

    from jax.extend.core import gensym, new_jaxpr_eqn

    if getattr(tgm.clean_layer_tags_jaxpr, "__hamiltonzero_hoist_tags__", False):
        return

    _orig_clean_layer = tgm.clean_layer_tags_jaxpr

    def _make_transpose_swap01_eqn(scan_outvar, make_var_func):

        ndim = len(scan_outvar.aval.shape)
        if ndim < 2:
            return None, scan_outvar
        from jax._src.lax import lax as _jlax

        permutation = (1, 0) + tuple(range(2, ndim))
        new_shape = tuple(scan_outvar.aval.shape[p] for p in permutation)
        new_aval = _jcore.ShapedArray(new_shape, scan_outvar.aval.dtype)
        new_outvar = make_var_func(new_aval)
        eqn = new_jaxpr_eqn(
            invars=[scan_outvar],
            outvars=[new_outvar],
            primitive=_jlax.transpose_p,
            params={"permutation": permutation},
            effects=frozenset(),
        )
        return eqn, new_outvar

    def _hoist_tags_recursive(closed_jaxpr, make_var_func):

        new_eqns = []
        hoisted_tags = []

        for eqn in closed_jaxpr.jaxpr.eqns:
            if eqn.primitive.name not in HIGHER_ORDER_NAMES:
                new_eqns.append(eqn)
                continue
            if eqn.primitive.name == "cond":
                new_eqns.append(eqn)
                continue
            if eqn.primitive.name == "while":
                body_jaxpr = eqn.params["body_jaxpr"]
                key = "body_jaxpr"
                supports_extension = False
            elif eqn.primitive.name == "scan":
                body_jaxpr = eqn.params["jaxpr"]
                key = "jaxpr"
                supports_extension = True
            elif eqn.primitive.name == "pjit":
                body_jaxpr = eqn.params["jaxpr"]
                key = "jaxpr"
                supports_extension = False
            elif eqn.primitive.name in ("xla_call", "xla_pmap"):
                body_jaxpr = eqn.params["call_jaxpr"]
                key = "call_jaxpr"
                supports_extension = False
            else:
                new_eqns.append(eqn)
                continue

            body_closed = to_closed_jaxpr(body_jaxpr)
            new_body_closed, nested_hoisted = _hoist_tags_recursive(
                body_closed, make_var_func
            )

            body_invars = body_jaxpr.jaxpr.invars
            body_eqns_no_tags = []
            tag_var_map = {}

            new_body_captures: list = []
            new_body_capture_id_to_idx: dict[int, int] = {}

            output_var_to_aux_xs: dict[int, tuple] = {}

            scan_length = eqn.params["length"] if eqn.primitive.name == "scan" else None

            deferred_specs: list = []

            import jax.extend as _jex_chain

            def _resolve_tag_chain(w):

                while not isinstance(w, _jex_chain.core.Literal) and w in tag_var_map:
                    w = tag_var_map[w]
                return w

            for body_eqn in new_body_closed.jaxpr.eqns:
                if not isinstance(body_eqn.primitive, tags.LayerTag):
                    body_eqns_no_tags.append(body_eqn)
                    continue

                meta = body_eqn.params["meta"]
                for ind1, ind2 in enumerate(meta.outputs_index):
                    tag_var_map[body_eqn.outvars[ind1]] = body_eqn.invars[ind2]

                params_index_set = set(meta.params_index)
                partial_invars: list = [None] * len(body_eqn.invars)
                deferred: list = []
                hoistable = True

                (
                    _scan_num_consts,
                    _scan_num_carry,
                    _,
                ) = _scan_partition_sizes(eqn)
                _xs_threshold = _scan_num_consts + _scan_num_carry

                tag_has_xs_iterated_params = False
                output_idx_set = set(meta.outputs_index)

                for _arg_idx_pre, _v_pre in enumerate(body_eqn.invars):
                    if _arg_idx_pre not in params_index_set:
                        continue
                    _v_pre_resolved = _resolve_tag_chain(_v_pre)
                    if _v_pre_resolved in body_invars:
                        _idx = body_invars.index(_v_pre_resolved)
                        if eqn.primitive.name == "scan" and _idx >= _xs_threshold:
                            tag_has_xs_iterated_params = True
                            break

                use_aux_xs_for_outputs = supports_extension and scan_length is not None

                use_accumulating_aux_base = False
                if (
                    use_aux_xs_for_outputs
                    and not tag_has_xs_iterated_params
                    and getattr(meta, "variant", None) == "dense"
                    and len(meta.inputs_index) == 1
                    and len(meta.outputs_index) == 1
                    and len(meta.params_index) >= 1
                ):
                    _const_indices = []
                    for _candidate_idx in (
                        *meta.inputs_index,
                        *meta.params_index,
                    ):
                        _candidate = _resolve_tag_chain(body_eqn.invars[_candidate_idx])
                        if _candidate not in body_invars:
                            _const_indices = []
                            break
                        _candidate_body_idx = body_invars.index(_candidate)
                        if _candidate_body_idx >= _scan_num_consts:
                            _const_indices = []
                            break
                        _const_indices.append(_candidate_body_idx)
                    _output_candidate = _resolve_tag_chain(
                        body_eqn.invars[meta.outputs_index[0]]
                    )
                    if (
                        len(_const_indices)
                        == len(meta.inputs_index) + len(meta.params_index)
                        and _output_candidate not in body_invars
                    ):
                        _outer_input = eqn.invars[_const_indices[0]]
                        use_accumulating_aux_base = (
                            _outer_input.aval.shape[:-1]
                            == _output_candidate.aval.shape[:-1]
                        )

                for arg_idx, v in enumerate(body_eqn.invars):
                    v_resolved = _resolve_tag_chain(v)
                    if v_resolved in body_invars:
                        idx = body_invars.index(v_resolved)
                        is_xs_iterated = (
                            eqn.primitive.name == "scan" and idx >= _xs_threshold
                        )
                        is_param = arg_idx in params_index_set
                        if is_xs_iterated and is_param:
                            tag_has_xs_iterated_params = True

                        _slot_is_scan_carry = (
                            eqn.primitive.name == "scan"
                            and idx >= _scan_num_consts
                            and idx < _xs_threshold
                        )
                        if (
                            (tag_has_xs_iterated_params or _slot_is_scan_carry)
                            and not is_param
                            and not is_xs_iterated
                            and supports_extension
                            and scan_length is not None
                        ):
                            cap_id = id(v_resolved)
                            if cap_id not in new_body_capture_id_to_idx:
                                new_body_capture_id_to_idx[cap_id] = len(
                                    new_body_captures,
                                )
                                new_body_captures.append(v_resolved)
                            deferred.append(
                                (arg_idx, new_body_capture_id_to_idx[cap_id]),
                            )
                            continue

                        partial_invars[arg_idx] = eqn.invars[idx]
                    elif arg_idx in params_index_set:
                        hoistable = False
                        break
                    elif (
                        arg_idx in output_idx_set
                        and use_aux_xs_for_outputs
                        and supports_extension
                        and scan_length is not None
                    ):
                        body_v_id = id(v_resolved)
                        if body_v_id not in output_var_to_aux_xs:
                            aux_body_invar = make_var_func(v_resolved.aval)
                            aux_outer_aval = _jcore.ShapedArray(
                                (scan_length, *v_resolved.aval.shape),
                                v_resolved.aval.dtype,
                            )
                            aux_outer_var = make_var_func(aux_outer_aval)
                            aux_tag_var = (
                                make_var_func(v_resolved.aval)
                                if use_accumulating_aux_base
                                else aux_outer_var
                            )
                            output_var_to_aux_xs[body_v_id] = (
                                v_resolved,
                                aux_body_invar,
                                aux_outer_var,
                                aux_tag_var,
                            )
                        (
                            _,
                            _,
                            aux_outer_var,
                            aux_tag_var,
                        ) = output_var_to_aux_xs[body_v_id]
                        if (
                            aux_tag_var is not aux_outer_var
                        ) != use_accumulating_aux_base:
                            raise ValueError(
                                "Conflicting scan accumulation contracts for "
                                "the same hoisted layer output."
                            )
                        partial_invars[arg_idx] = aux_tag_var
                    elif supports_extension:
                        cap_id = id(v_resolved)
                        if cap_id not in new_body_capture_id_to_idx:
                            new_body_capture_id_to_idx[cap_id] = len(
                                new_body_captures,
                            )
                            new_body_captures.append(v_resolved)
                        deferred.append(
                            (arg_idx, new_body_capture_id_to_idx[cap_id]),
                        )
                    else:
                        import jax.extend as _jex
                        import numpy as _np

                        zero_val = _np.zeros(
                            v_resolved.aval.shape,
                            dtype=v_resolved.aval.dtype,
                        )
                        partial_invars[arg_idx] = _jex.core.Literal(
                            zero_val,
                            v_resolved.aval,
                        )

                if not hoistable:
                    body_eqns_no_tags.append(body_eqn)
                    continue

                deferred_specs.append(
                    (
                        body_eqn,
                        partial_invars,
                        deferred,
                        tag_has_xs_iterated_params,
                        use_aux_xs_for_outputs,
                    )
                )

            import jax.extend as _jex

            def _remap_invars(eqns):

                out = []
                for e in eqns:
                    new_invars = [
                        _resolve_tag_chain(w)
                        if not isinstance(w, _jex.core.Literal)
                        else w
                        for w in e.invars
                    ]
                    out.append(e.replace(invars=new_invars))
                return out

            body_eqns_no_tags = _remap_invars(body_eqns_no_tags)
            new_body_outvars = [
                _resolve_tag_chain(v) if not isinstance(v, _jex.core.Literal) else v
                for v in new_body_closed.jaxpr.outvars
            ]

            if output_var_to_aux_xs:
                from jax._src.lax import lax as _jlax

                aug_for_id: dict[int, tuple] = {}
                for body_v_id, (
                    body_v,
                    aux_body_invar,
                    _,
                    _,
                ) in output_var_to_aux_xs.items():
                    aug_var = make_var_func(body_v.aval)
                    aug_for_id[body_v_id] = (aug_var, aux_body_invar)

                seen_ids: set[int] = set()

                def _retarget_to_aug(w):
                    if isinstance(w, _jex.core.Literal):
                        return w
                    wid = id(w)
                    if wid in aug_for_id and wid in seen_ids:
                        return aug_for_id[wid][0]
                    return w

                augmented_eqns = []
                for body_eqn_clean in body_eqns_no_tags:
                    augmented_eqns.append(
                        body_eqn_clean.replace(
                            invars=[_retarget_to_aug(w) for w in body_eqn_clean.invars]
                        )
                    )
                    for o in body_eqn_clean.outvars:
                        oid = id(o)
                        if oid in aug_for_id and oid not in seen_ids:
                            aug_var, aux_body_invar = aug_for_id[oid]
                            aug_eqn = new_jaxpr_eqn(
                                invars=[o, aux_body_invar],
                                outvars=[aug_var],
                                primitive=_jlax.add_p,
                                params={},
                                effects=frozenset(),
                            )
                            augmented_eqns.append(aug_eqn)
                            seen_ids.add(oid)

                assert seen_ids == set(aug_for_id.keys()), (
                    "aux-xs injection: some output Vars not encountered as "
                    "body-eqn outvars"
                )
                body_eqns_no_tags = augmented_eqns
                new_body_outvars = [
                    _retarget_to_aug(v) if not isinstance(v, _jex.core.Literal) else v
                    for v in new_body_outvars
                ]

            new_body_outvars = list(new_body_outvars) + list(new_body_captures)

            new_body_invars_list = list(new_body_closed.jaxpr.invars) + [
                aux_body_invar
                for _, aux_body_invar, _, _ in output_var_to_aux_xs.values()
            ]

            new_body_jaxpr = new_body_closed.jaxpr.replace(
                eqns=body_eqns_no_tags,
                outvars=new_body_outvars,
                invars=new_body_invars_list,
            )
            new_body_closed_clean = ClosedJaxpr(
                new_body_jaxpr,
                new_body_closed.consts,
            )

            params_dict = dict(**eqn.params)
            params_dict[key] = to_jaxpr_or_closed_jaxpr(
                new_body_closed_clean,
                body_jaxpr,
            )
            if eqn.primitive.name == "scan":
                _extend_scan_flat_trees(
                    params_dict,
                    extra_xs=len(output_var_to_aux_xs),
                    extra_ys=len(new_body_captures),
                )

            if output_var_to_aux_xs:
                from jax._src.lax import lax as _jlax
                import jax.extend as _jex
                import numpy as _np

                for (
                    body_v,
                    _aux_body_invar,
                    aux_outer_var,
                    aux_tag_var,
                ) in output_var_to_aux_xs.values():
                    zero_scalar_aval = _jcore.ShapedArray(
                        (),
                        body_v.aval.dtype,
                    )
                    zero_scalar_literal = _jex.core.Literal(
                        _np.array(0.0, dtype=body_v.aval.dtype),
                        zero_scalar_aval,
                    )
                    first_bcast_outvar = aux_tag_var
                    first_bcast_shape = aux_tag_var.aval.shape
                    bcast_eqn = new_jaxpr_eqn(
                        invars=[zero_scalar_literal],
                        outvars=[first_bcast_outvar],
                        primitive=_jlax.broadcast_in_dim_p,
                        params={
                            "shape": first_bcast_shape,
                            "broadcast_dimensions": (),
                            "sharding": None,
                        },
                        effects=frozenset(),
                    )
                    new_eqns.append(bcast_eqn)
                    if aux_tag_var is not aux_outer_var:
                        expand_eqn = new_jaxpr_eqn(
                            invars=[aux_tag_var],
                            outvars=[aux_outer_var],
                            primitive=_jlax.broadcast_in_dim_p,
                            params={
                                "shape": (
                                    scan_length,
                                    *body_v.aval.shape,
                                ),
                                "broadcast_dimensions": tuple(
                                    range(1, body_v.aval.ndim + 1)
                                ),
                                "sharding": None,
                            },
                            effects=frozenset(),
                        )
                        new_eqns.append(expand_eqn)

            new_capture_outvars = []
            for cap_v in new_body_captures:
                cap_aval = _jcore.ShapedArray(
                    (scan_length, *cap_v.aval.shape),
                    cap_v.aval.dtype,
                )
                new_capture_outvars.append(make_var_func(cap_aval))

            new_scan_invars = list(eqn.invars) + [
                aux_outer_var
                for _, _, aux_outer_var, _ in output_var_to_aux_xs.values()
            ]
            new_eqn = eqn.replace(
                params=params_dict,
                invars=new_scan_invars,
                outvars=list(eqn.outvars) + list(new_capture_outvars),
            )
            new_eqns.append(new_eqn)

            aux_xs_capture_ids: set[int] = set()
            for _be, _pi, _df, _has_xs, _use_aux in deferred_specs:
                if _use_aux:
                    for _aidx, _cidx in _df:
                        aux_xs_capture_ids.add(_cidx)
            transposed_outvars: list = []
            for _cap_idx, cap_outvar in enumerate(new_capture_outvars):
                if _cap_idx in aux_xs_capture_ids:
                    transposed_outvars.append(cap_outvar)
                    continue
                t_eqn, t_outvar = _make_transpose_swap01_eqn(
                    cap_outvar,
                    make_var_func,
                )
                if t_eqn is not None:
                    new_eqns.append(t_eqn)
                transposed_outvars.append(t_outvar)

            for (
                body_eqn,
                partial_invars,
                deferred,
                has_xs_iter_params,
                _use_aux_xs,
            ) in deferred_specs:
                final_invars = list(partial_invars)
                for arg_idx, cap_idx in deferred:
                    final_invars[arg_idx] = transposed_outvars[cap_idx]
                new_outvars = [make_var_func(v.aval) for v in body_eqn.outvars]

                hoisted_params = body_eqn.params
                if has_xs_iter_params:
                    import dataclasses as _dc

                    orig_meta = body_eqn.params["meta"]

                    _v = orig_meta.variant or ""
                    new_variant = None
                    if _v == "scale_and_shift":
                        new_variant = "stacked_scale_and_shift"
                    elif _v == "structural_repeated_dense":
                        new_variant = "structural_stacked_repeated_dense"
                    elif _v == "structural_scale_and_shift":
                        new_variant = "structural_stacked_scale_and_shift"
                    if new_variant is not None:
                        new_meta = _dc.replace(orig_meta, variant=new_variant)
                        hoisted_params = {**body_eqn.params, "meta": new_meta}
                hoisted_tags.append(
                    new_jaxpr_eqn(
                        invars=final_invars,
                        outvars=new_outvars,
                        primitive=body_eqn.primitive,
                        params=hoisted_params,
                        effects=body_eqn.effects,
                    )
                )

            for nh_eqn in nested_hoisted:
                outer_remapped = []
                ok = True
                for v in nh_eqn.invars:
                    if v in body_jaxpr.jaxpr.invars:
                        idx = body_jaxpr.jaxpr.invars.index(v)
                        outer_remapped.append(eqn.invars[idx])
                    else:
                        ok = False
                        break
                if ok:
                    new_outvars2 = [make_var_func(v.aval) for v in nh_eqn.outvars]
                    hoisted_tags.append(
                        new_jaxpr_eqn(
                            invars=outer_remapped,
                            outvars=new_outvars2,
                            primitive=nh_eqn.primitive,
                            params=nh_eqn.params,
                            effects=nh_eqn.effects,
                        )
                    )

        new_closed = ClosedJaxpr(
            closed_jaxpr.jaxpr.replace(eqns=new_eqns),
            closed_jaxpr.consts,
        )
        return new_closed, hoisted_tags

    def clean_layer_tags_jaxpr_patched(jaxpr, only_remove_auto_tags=False):

        closed = to_closed_jaxpr(jaxpr)
        make_var_func = gensym()
        closed, hoisted = _hoist_tags_recursive(closed, make_var_func)

        seen_param_keys = set()
        deduped = []
        for h in hoisted:
            meta = h.params["meta"]
            key = tuple(id(h.invars[i]) for i in meta.params_index)
            if key in seen_param_keys:
                continue
            seen_param_keys.add(key)
            deduped.append(h)
        hoisted = deduped

        if hoisted:
            new_eqns = list(closed.jaxpr.eqns) + list(hoisted)
            closed = ClosedJaxpr(
                closed.jaxpr.replace(eqns=new_eqns),
                closed.consts,
            )
        return _orig_clean_layer(
            to_jaxpr_or_closed_jaxpr(closed, jaxpr),
            only_remove_auto_tags=only_remove_auto_tags,
        )

    clean_layer_tags_jaxpr_patched.__hamiltonzero_hoist_tags__ = True
    tgm.clean_layer_tags_jaxpr = clean_layer_tags_jaxpr_patched


def _patch_kfactor_identity_init() -> None:

    import jax.numpy as _jnp

    from kfac_jax._src.curvature_blocks import (
        kronecker_factored as _kf,
    )
    from kfac_jax._src import utils as _kfac_utils

    if getattr(_kf.KroneckerFactored._init, "__hamiltonzero_kfactor_identity__", False):
        return

    _orig_kf_init = _kf.KroneckerFactored._init

    def _patched_kf_init(
        self, rng, exact_powers_to_cache, approx_powers_to_cache, cache_eigenvalues
    ):

        cache = {}
        factors = []
        for i, d in enumerate(self.array_shape):
            eye = _jnp.eye(d, dtype=self.dtype) * _jnp.asarray(1.0, dtype=self.dtype)

            wma = _kfac_utils.WeightedMovingAverage(
                value=eye,
                weight=_jnp.asarray(1.0, dtype=self.dtype),
            )
            factors.append(wma)
            if cache_eigenvalues or exact_powers_to_cache:
                cache[f"{i}_factor_eigenvalues"] = _jnp.ones((d,), dtype=self.dtype)
            if exact_powers_to_cache:
                cache[f"{i}_factor_eigen_vectors"] = _jnp.eye(d, dtype=self.dtype)
            for power in approx_powers_to_cache:
                if power != -1:
                    raise NotImplementedError(
                        f"Approximations for power {power} not implemented."
                    )
                if str(power) not in cache:
                    cache[str(power)] = {}
                cache[str(power)][f"{i}_factor"] = _jnp.eye(d, dtype=self.dtype)
        return _kf.KroneckerFactored.State(
            cache=cache,
            factors=tuple(factors),
        )

    _patched_kf_init.__hamiltonzero_kfactor_identity__ = True
    _kf.KroneckerFactored._init = _patched_kf_init

    if getattr(
        _kf.RepeatedDenseKroneckerFactored._init,
        "__hamiltonzero_avg_repeats_one__",
        False,
    ):
        return

    def _patched_rd_init(
        self, rng, exact_powers_to_cache, approx_powers_to_cache, cache_eigenvalues
    ):
        super_state = _kf.KroneckerFactored._init(
            self,
            rng,
            exact_powers_to_cache,
            approx_powers_to_cache,
            cache_eigenvalues,
        )
        avg = _kfac_utils.WeightedMovingAverage(
            value=_jnp.asarray(1.0, dtype=self.dtype),
            weight=_jnp.asarray(1.0, dtype=self.dtype),
        )
        return _kf.RepeatedDenseKroneckerFactored.State(
            average_repeats=avg,
            **super_state.__dict__,
        )

    _patched_rd_init.__hamiltonzero_avg_repeats_one__ = True
    _kf.RepeatedDenseKroneckerFactored._init = _patched_rd_init


def _patch_pi_adjusted_kronecker_factors_floor() -> None:

    import jax.numpy as _jnp
    from kfac_jax._src.utils import math as _kfac_math

    if getattr(
        _kfac_math.pi_adjusted_kronecker_factors,
        "__hamiltonzero_kron_floor__",
        False,
    ):
        return

    _orig = _kfac_math.pi_adjusted_kronecker_factors
    EPS_FLOOR = 1e-6
    EPS_REL = 1e-4

    def _shift_from_avg_diag(avg_diag, scale):
        eps_abs = _jnp.asarray(EPS_FLOOR, dtype=avg_diag.dtype)
        eps_rel = _jnp.asarray(EPS_REL, dtype=avg_diag.dtype)
        floor = _jnp.maximum(eps_abs, eps_rel * scale)
        return _jnp.maximum(floor, floor - avg_diag)

    def _floor_factor(f):
        if f.ndim == 0 or f.size == 1:
            return f + _shift_from_avg_diag(f, _jnp.abs(f))
        if f.ndim == 1:
            avg_diag = _jnp.mean(f)
            scale = _jnp.max(_jnp.abs(f))
            return f + _shift_from_avg_diag(avg_diag, scale)
        if f.ndim == 2:
            d = f.shape[-1]
            diag = _jnp.diagonal(f)
            avg_diag = _jnp.sum(diag) / d
            scale = _jnp.max(diag)
            shift = _shift_from_avg_diag(avg_diag, scale)
            return f + shift * _jnp.eye(d, dtype=f.dtype)

        if f.ndim >= 3 and f.shape[-1] == f.shape[-2]:
            d = f.shape[-1]
            eye = _jnp.eye(d, dtype=f.dtype)
            for _ in range(f.ndim - 2):
                eye = eye[None, ...]
            diag = _jnp.diagonal(f, axis1=-2, axis2=-1)
            avg_diag = _jnp.mean(diag, axis=-1)
            scale = _jnp.max(diag, axis=-1)
            shift = _shift_from_avg_diag(avg_diag, scale)
            return f + shift[..., None, None] * eye
        return f

    def patched(*factors, damping):
        floored = tuple(_floor_factor(f) for f in factors)
        return _orig(*floored, damping=damping)

    patched.__hamiltonzero_kron_floor__ = True
    _kfac_math.pi_adjusted_kronecker_factors = patched

    from kfac_jax._src import utils as _kfac_utils_pkg

    if hasattr(_kfac_utils_pkg, "pi_adjusted_kronecker_factors"):
        _kfac_utils_pkg.pi_adjusted_kronecker_factors = patched


def _patch_nested_scan_parent_walk() -> None:

    from kfac_jax._src import tag_graph_matcher as tgm

    _TagLocation = tgm.TagLocation
    if getattr(_TagLocation, "__hamiltonzero_nested_parent_walk__", False):
        return

    def _invars_of(eqn):
        nm = eqn.primitive.name
        if nm in ("scan", "pjit"):
            return eqn.params["jaxpr"].jaxpr.invars
        if nm == "while":
            return eqn.params["body_jaxpr"].jaxpr.invars
        if nm in ("xla_call", "xla_pmap"):
            return eqn.params["call_jaxpr"].invars
        raise NotImplementedError(f"higher-order primitive {nm!r}")

    def _walk(param_vars, eqns_in_order):
        for eqn, _ in eqns_in_order:
            invars = _invars_of(eqn)
            p_indexes = [invars.index(p) for p in param_vars]
            param_vars = tuple(eqn.invars[pi] for pi in p_indexes)
        return param_vars

    def _top_level_parameters(self):
        pv = self.bottom_level_parameters
        return _walk(pv, list(self.parent_equations))

    def _full_name_ordered(self, eqns_in_order):
        param_vars = self.bottom_level_parameters
        parts = []
        for eqn, n in eqns_in_order:
            nm = eqn.primitive.name
            invars = _invars_of(eqn)
            p_indexes = [invars.index(p) for p in param_vars]
            piece = f"{nm}_{n}/"
            if nm == "scan":
                num_consts, _, _ = _scan_partition_sizes(eqn)
                checks = [pi < num_consts for pi in p_indexes]
                if not (all(checks) or all(not ci for ci in checks)):
                    raise ValueError(
                        "Parameters inside scan of the same tag are not both "
                        "carry or const."
                    )
                piece = piece + ("const/" if all(checks) else "carry/")
            parts.append(piece)
            param_vars = [eqn.invars[pi] for pi in p_indexes]

        prefix = "".join(reversed(parts))
        return prefix + self.base_name

    def _full_name(self):
        return _full_name_ordered(self, list(self.parent_equations))

    _TagLocation.top_level_parameters = property(_top_level_parameters)
    _TagLocation.full_name = property(_full_name)
    _TagLocation.__hamiltonzero_nested_parent_walk__ = True


_patch()
_patch_allow_multiple_registrations()
_patch_orphan_registration_in_sub_graphs()
_patch_manual_tag_outputs_that_are_graph_inputs()
_patch_hoist_layer_tags_from_scan()
_patch_kfactor_identity_init()
_patch_pi_adjusted_kronecker_factors_floor()
_patch_nested_scan_parent_walk()


__all__: list[str] = []
