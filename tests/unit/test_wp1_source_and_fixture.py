"""WP1 solver fixes: source deposition band and fixture thermal phase.

Covers the two review findings P0-2 / P0-3 (AM-Benchmark review 2026-08-21):

* the legacy exponential volumetric source deposited most of the absorbed
  power below a thin powder layer into substrate/support; the new
  ``--source-depth-cutoff`` band confines it, optionally renormalized;
* substrate/support quadrature points kept solid-branch rho/cp/k at any
  temperature; ``--fixture-thermal-phase follow-temperature`` routes them
  through the mushy/liquid property branches.

Defaults must reproduce the historical behaviour exactly.
"""
import math
import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem_am.config.schema import build_parser
from jax_fem_am.domain.events import update_phase_reference_and_eqp
from jax_fem_am.materials.phases import (
    STATE_LIQUID,
    STATE_MUSHY,
    STATE_POWDER,
    STATE_SOLID,
    STATE_SUBSTRATE,
    STATE_SUPPORT,
    STATE_VOID,
    mechanics_material_quads,
    thermal_material_quads,
)
from jax_fem_am.physics.thermal import TransientThermal
from jax_fem_am.simulation import acceleration
from jax_fem_am.verification.thermal_ledger import integrate_volume_terms


ABSORBED_POWER_W = 120.9  # 0.62 * 195 W, the V1 CBM-B operating point
BEAM_RADIUS_M = 50.0e-6
SOURCE_DEPTH_M = 100.0e-6  # V1 optical penetration depth
POWDER_LAYER_M = 20.0e-6  # V1 powder layer = cutoff band
V1_DOMAIN_DEPTH_M = 300.0e-6
SOLIDUS_K = 1563.0
LIQUIDUS_K = 1623.0


def _legacy_problem(cutoff=0.0, renormalize=False, with_attrs=True):
    problem = object.__new__(TransientThermal)
    problem.plane_axis0_id = 0
    problem.plane_axis1_id = 1
    problem.build_axis_id = 2
    problem.build_sign = 1.0  # laser above at z=0, material below at z<0
    problem.front_surface_loss_h = 0.0
    problem.front_surface_loss_thickness = 0.0
    problem.front_surface_loss_radiation = False
    problem.ambient = 300.0
    problem.source_model = "legacy"
    if with_attrs:
        problem.source_depth_cutoff = cutoff
        problem.source_cutoff_renormalize = renormalize
    return problem


def _make_density(problem, center=(0.0, 0.0, 0.0)):
    """Volumetric deposition rate at a point (T == T_old kills storage)."""
    mass_map = problem.get_mass_map()
    temperature = jnp.asarray([problem.ambient])
    zero = jnp.asarray([0.0])
    center = jnp.asarray(center)

    def density(point):
        residual = mass_map(
            temperature,
            jnp.asarray(point),
            temperature,
            jnp.asarray([1.0]),
            center,
            jnp.asarray([ABSORBED_POWER_W]),
            jnp.asarray([BEAM_RADIUS_M]),
            jnp.asarray([SOURCE_DEPTH_M]),
            jnp.asarray([1.0]),
            jnp.asarray([1.0]),
            jnp.asarray([8000.0]),
            jnp.asarray([600.0]),
            jnp.asarray([20.0]),
            zero,
            zero,
            zero,
        )
        return -float(residual[0])

    return density


def _source_density(problem, point):
    return _make_density(problem)(point)


def _depth_profile_sum(problem, depths_m):
    density = _make_density(problem)
    return sum(density((0.0, 0.0, -float(z))) for z in depths_m)


class TestSourceDepthCutoff:
    def test_band_capture_matches_analytic_fractions(self):
        # Midpoint samples; the powder-layer boundary falls between samples.
        dz = 2.0e-6
        depths = np.arange(0.5 * dz, 0.8e-3, dz)
        full = _depth_profile_sum(_legacy_problem(), depths)
        banded = _depth_profile_sum(
            _legacy_problem(cutoff=POWDER_LAYER_M), depths
        )
        renormalized = _depth_profile_sum(
            _legacy_problem(cutoff=POWDER_LAYER_M, renormalize=True), depths
        )

        capture = 1.0 - math.exp(-POWDER_LAYER_M / SOURCE_DEPTH_M)
        assert banded / full == pytest.approx(capture, abs=5.0e-3)
        # Review P0-2 arithmetic: a 20 um powder layer captures ~18.1 % of
        # the half-space deposition; the 300 um V1 domain captures ~95.0 %.
        assert banded / full == pytest.approx(0.18127, abs=5.0e-3)
        in_domain = _depth_profile_sum(
            _legacy_problem(), depths[depths <= V1_DOMAIN_DEPTH_M]
        )
        assert in_domain / full == pytest.approx(0.95021, abs=5.0e-3)
        # Renormalization restores the full absorbed power inside the band.
        assert renormalized / full == pytest.approx(1.0, abs=5.0e-3)

    def test_cutoff_disabled_is_exactly_legacy(self):
        legacy = _legacy_problem(with_attrs=False)
        explicit_zero = _legacy_problem(cutoff=0.0)
        for z in (0.0, 10.0e-6, 150.0e-6, 400.0e-6):
            point = (5.0e-6, -3.0e-6, -z)
            assert _source_density(explicit_zero, point) == _source_density(
                legacy, point
            )

    def test_problem_validation(self):
        problem = object.__new__(TransientThermal)
        problem.location_fns = ()
        problem.boundary_inds_list = []
        with pytest.raises(ValueError, match="nonnegative"):
            TransientThermal.custom_init(
                problem,
                0.0, 300.0, 0.0, 5.67e-8, 2, 0, 1, -1.0, 0,
                0.0, 0.0, False,
                source_model="legacy",
                source_depth_cutoff=-1.0e-6,
            )
        with pytest.raises(ValueError, match="legacy"):
            TransientThermal.custom_init(
                problem,
                0.0, 300.0, 0.0, 5.67e-8, 2, 0, 1, -1.0, 0,
                0.0, 0.0, False,
                source_model="paper_hemispherical",
                source_depth_cutoff=1.0e-6,
            )
        with pytest.raises(ValueError, match="requires"):
            TransientThermal.custom_init(
                problem,
                0.0, 300.0, 0.0, 5.67e-8, 2, 0, 1, -1.0, 0,
                0.0, 0.0, False,
                source_model="legacy",
                source_depth_cutoff=0.0,
                source_cutoff_renormalize=True,
            )


class TestLedgerMirrorsKernel:
    def _ledger_laser_j(self, points, weights, cutoff, renormalize):
        shape = (1, len(points))
        constant = 300.0 * np.ones(shape)
        return integrate_volume_terms(
            jxw=np.asarray(weights, dtype=np.float64).reshape(shape),
            points=np.asarray(points, dtype=np.float64).reshape(
                shape + (3,)
            ),
            temperature_old=constant,
            temperature_new=constant,
            rho=8000.0 * np.ones(shape),
            cp=600.0 * np.ones(shape),
            latent_cp=np.zeros(shape),
            laser_center=np.zeros(3),
            effective_laser_power_w=ABSORBED_POWER_W,
            beam_radius_m=BEAM_RADIUS_M,
            source_depth_m=SOURCE_DEPTH_M,
            laser_switch=1.0,
            active=np.ones(shape),
            cooling_only=np.zeros(shape),
            old_layer_cooling_h=0.0,
            ambient_k=300.0,
            dt_s=1.0,
            build_axis=2,
            plane_axes=(0, 1),
            build_sign=1.0,
            front_loss_h=0.0,
            front_loss_thickness_m=0.0,
            front_loss_radiation=False,
            emissivity=0.0,
            stefan_boltzmann=5.67e-8,
            source_model="legacy",
            source_depth_cutoff_m=cutoff,
            source_cutoff_renormalize=renormalize,
        )["laser_deposited_j"]

    def test_ledger_reconstruction_matches_kernel(self):
        rng = np.random.default_rng(20260821)
        points = np.column_stack(
            [
                rng.uniform(-60e-6, 60e-6, 48),
                rng.uniform(-60e-6, 60e-6, 48),
                -rng.uniform(0.0, 120e-6, 48),
            ]
        )
        weights = rng.uniform(0.5, 2.0, 48) * 1.0e-15
        for cutoff, renormalize in (
            (0.0, False),
            (POWDER_LAYER_M, False),
            (POWDER_LAYER_M, True),
        ):
            problem = _legacy_problem(cutoff=cutoff, renormalize=renormalize)
            kernel = sum(
                w * _source_density(problem, tuple(p))
                for p, w in zip(points, weights)
            )
            ledger = self._ledger_laser_j(
                points, weights, cutoff, renormalize
            )
            assert ledger == pytest.approx(kernel, rel=1.0e-12)

    def test_tiny_cutoff_renormalization_stays_finite_and_mirrored(self):
        cutoff = 1.0e-20
        point = np.asarray([[0.0, 0.0, 0.0]])
        weight = np.asarray([1.0e-15])
        problem = _legacy_problem(cutoff=cutoff, renormalize=True)
        kernel = weight[0] * _source_density(problem, tuple(point[0]))
        ledger = self._ledger_laser_j(point, weight, cutoff, True)
        assert math.isfinite(kernel)
        assert math.isfinite(ledger)
        assert ledger == pytest.approx(kernel, rel=1.0e-12)

    def test_ledger_cutoff_validation(self):
        with pytest.raises(ValueError, match="nonnegative"):
            self._ledger_laser_j(
                np.zeros((1, 3)), np.ones(1), -1.0e-6, False
            )
        with pytest.raises(ValueError, match="requires"):
            self._ledger_laser_j(np.zeros((1, 3)), np.ones(1), 0.0, True)


def _thermal_args(fixture_thermal_phase="frozen-solid"):
    return SimpleNamespace(
        rho=8000.0,
        rho_solid=8000.0,
        rho_liquid=7500.0,
        rho_powder=4000.0,
        cp=500.0,
        cp_solid=500.0,
        cp_liquid=700.0,
        cp_powder=520.0,
        conductivity=20.0,
        conductivity_solid=20.0,
        conductivity_liquid=35.0,
        conductivity_powder=0.3,
        inactive_thermal_factor=1.0e-6,
        inactive_mass_factor=None,
        old_layer_thermal_factor=1.0,
        solidus_temperature=SOLIDUS_K,
        liquidus_temperature=LIQUIDUS_K,
        latent_heat=280_000.0,
        layer_activation_mode="front",
        future_layer_mode="void",
        powder_mode="powder",
        fixture_thermal_phase=fixture_thermal_phase,
    )


EMPTY_TABLES = {
    "cp_solid": None,
    "k_solid": None,
    "cp_powder": None,
    "k_powder": None,
    "cp_liquid": None,
    "k_liquid": None,
}


def _quad(values):
    return jnp.asarray(values, dtype=jnp.float64).reshape(-1, 1, 1)


class TestFixtureThermalPhase:
    # One quad point per row: support above liquidus, substrate in the
    # mushy interval, cold support, printed solid at liquid temperature
    # (identity check: non-fixture rows must never be touched), powder.
    PHASES = (
        STATE_SUPPORT,
        STATE_SUBSTRATE,
        STATE_SUPPORT,
        STATE_SOLID,
        STATE_POWDER,
    )
    TEMPERATURES = (1700.0, 1600.0, 300.0, 1700.0, 1700.0)

    def _run(self, mode):
        args = _thermal_args(mode)
        return thermal_material_quads(
            _quad(self.TEMPERATURES),
            jnp.ones((5, 1, 1)),
            _quad(self.PHASES),
            args,
            EMPTY_TABLES,
            printed_quad=jnp.ones((5, 1, 1)),
            cooling_only_quad=jnp.zeros((5, 1, 1)),
        )

    def test_frozen_solid_keeps_historical_fallthrough(self):
        rho, cp, k, latent = self._run("frozen-solid")
        assert float(k[0, 0, 0]) == 20.0  # molten support still solid k
        assert float(k[1, 0, 0]) == 20.0
        assert float(cp[0, 0, 0]) == 500.0
        assert float(rho[0, 0, 0]) == 8000.0

    def test_follow_temperature_routes_fixture_through_phase_branches(self):
        rho, cp, k, latent = self._run("follow-temperature")
        # Support above liquidus: liquid properties.
        assert float(k[0, 0, 0]) == pytest.approx(35.0)
        assert float(cp[0, 0, 0]) == pytest.approx(700.0)
        assert float(rho[0, 0, 0]) == pytest.approx(7500.0)
        # Substrate in the mushy interval: linear blend at fraction 37/60.
        frac = (1600.0 - SOLIDUS_K) / (LIQUIDUS_K - SOLIDUS_K)
        assert float(k[1, 0, 0]) == pytest.approx(20.0 + frac * 15.0)
        # Cold support: unchanged solid values.
        assert float(k[2, 0, 0]) == 20.0
        # Non-fixture rows identical to the frozen-solid run.
        frozen = self._run("frozen-solid")
        for field_new, field_old in zip((rho, cp, k, latent), frozen):
            np.testing.assert_array_equal(
                np.asarray(field_new[3:]), np.asarray(field_old[3:])
            )
        # Latent-heat support term is untouched by the flag.
        np.testing.assert_array_equal(
            np.asarray(latent), np.asarray(frozen[3])
        )

    def test_missing_flag_defaults_to_frozen_solid(self):
        args = _thermal_args()
        del args.fixture_thermal_phase
        rho, cp, k, latent = thermal_material_quads(
            _quad(self.TEMPERATURES),
            jnp.ones((5, 1, 1)),
            _quad(self.PHASES),
            args,
            EMPTY_TABLES,
            printed_quad=jnp.ones((5, 1, 1)),
            cooling_only_quad=jnp.zeros((5, 1, 1)),
        )
        assert float(k[0, 0, 0]) == 20.0


class TestJitPatchHonoursFixtureFlag:
    def _base_module(self):
        return SimpleNamespace(
            jax=jax,
            np=jnp,
            STATE_VOID=STATE_VOID,
            STATE_POWDER=STATE_POWDER,
            STATE_SOLID=STATE_SOLID,
            STATE_MUSHY=STATE_MUSHY,
            STATE_LIQUID=STATE_LIQUID,
            STATE_SUBSTRATE=STATE_SUBSTRATE,
            STATE_SUPPORT=STATE_SUPPORT,
            thermal_material_quads=thermal_material_quads,
            mechanics_material_quads=mechanics_material_quads,
            update_phase_reference_and_eqp=update_phase_reference_and_eqp,
        )

    def test_follow_temperature_bypasses_specialized_kernel(self):
        base = self._base_module()
        acceleration._LOOP_KERNEL_JIT_THERMAL_CACHE.clear()
        assert acceleration.install_loop_kernel_jit_patch(base, enabled=True)
        patched = base.thermal_material_quads

        molten_support = (
            _quad([1700.0]),
            jnp.ones((1, 1, 1)),
            _quad([STATE_SUPPORT]),
        )
        rho, cp, k, latent = patched(
            *molten_support,
            _thermal_args("follow-temperature"),
            EMPTY_TABLES,
            printed_quad=jnp.ones((1, 1, 1)),
            cooling_only_quad=jnp.zeros((1, 1, 1)),
        )
        # The specialized kernel would return solid k; the reference
        # implementation must be used and return liquid k.
        assert float(k[0, 0, 0]) == pytest.approx(35.0)

        rho, cp, k, latent = patched(
            *molten_support,
            _thermal_args("frozen-solid"),
            EMPTY_TABLES,
            printed_quad=jnp.ones((1, 1, 1)),
            cooling_only_quad=jnp.zeros((1, 1, 1)),
        )
        assert float(k[0, 0, 0]) == 20.0


class TestSchemaFlags:
    def test_defaults_preserve_historical_behaviour(self):
        args = build_parser().parse_args([])
        assert args.source_depth_cutoff == 0.0
        assert args.source_cutoff_renormalize is False
        assert args.fixture_thermal_phase == "frozen-solid"

    def test_explicit_values_parse(self):
        args = build_parser().parse_args(
            [
                "--source-depth-cutoff", "2.0e-5",
                "--source-cutoff-renormalize",
                "--fixture-thermal-phase", "follow-temperature",
            ]
        )
        assert args.source_depth_cutoff == 2.0e-5
        assert args.source_cutoff_renormalize is True
        assert args.fixture_thermal_phase == "follow-temperature"

    def test_invalid_choice_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["--fixture-thermal-phase", "molten"]
            )
