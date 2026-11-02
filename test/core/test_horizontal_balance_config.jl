#!/usr/bin/env julia
#
# Every preprocessing path that applies a Poisson balance uses the column
# balance and records `horizontal_balance = "column"`. GEOS closures that keep
# the native fluxes record `"none"`. The per-layer balance left no vertical mass
# flux (issue #11) and was removed with its `[numerics] balance_mode` key.
# `column_balance_weights` chooses how the correction is spread over the levels.

using Test
using Dates
using AtmosTransport
using .AtmosTransport.Preprocessing: HORIZONTAL_BALANCE_TAG, reject_removed_balance_mode,
    resolve_balance_settings, process_day, build_target_geometry,
    ERA5N320Settings, MERRA2Settings

const REPO = pkgdir(AtmosTransport)
