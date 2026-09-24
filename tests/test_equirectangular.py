# SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the equirectangular (360 panorama) camera model.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import math
import os

import pytest
import torch

import gsplat
from gsplat._helper import load_test_data
from gsplat.cuda._torch_cameras import _BaseCameraModel
from gsplat.cuda._wrapper import RollingShutterType

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
class TestEquirectangularMath:
    """Pure math tests for the Torch reference implementation
    (gsplat.cuda._torch_cameras._EquirectangularCameraModel). These don't
    depend on CameraWrappers, so they run regardless of BUILD_CAMERA_WRAPPERS.
    """

    def _make_camera(self, width=4096, height=2048, batch=1):
        pp = torch.zeros(batch, 2, device=device)
        return _BaseCameraModel.create(
            width=width,
            height=height,
            camera_model="equirectangular",
            principal_points=pp,
            rs_type=RollingShutterType.GLOBAL,
        )

    def test_known_directions(self):
        cam = self._make_camera()
        W, H = cam.width, cam.height
        cases = {
            "+Z (forward)": ([0.0, 0.0, 1.0], (W / 2, H / 2)),
            "+X (right)": ([1.0, 0.0, 0.0], (3 * W / 4, H / 2)),
            "-X (left)": ([-1.0, 0.0, 0.0], (W / 4, H / 2)),
            "-Y (up -> top row)": ([0.0, -1.0, 0.0], (W / 2, 0.0)),
        }
        for name, (ray, expected) in cases.items():
            r = torch.tensor([ray], device=device)
            ip, valid = cam.camera_ray_to_image_point(r, margin_factor=0.0)
            got = ip[0].tolist()
            assert valid.item(), name
            assert math.isclose(got[0], expected[0], abs_tol=1e-3), (name, got, expected)
            assert math.isclose(got[1], expected[1], abs_tol=1e-3), (name, got, expected)

    def test_round_trip(self):
        cam = self._make_camera(batch=3)
        torch.manual_seed(0)
        rays = torch.randn(3, 2000, 3, device=device)
        rays = rays / rays.norm(dim=-1, keepdim=True)

        image_points, valid = cam.camera_ray_to_image_point(rays, margin_factor=0.0)
        rays_back, valid_back = cam.image_point_to_camera_ray(image_points)

        err = (rays - rays_back).norm(dim=-1)
        # A handful of rays land exactly on the x=0/x=width seam or a pole
        # (margin_factor=0 boundary case); exclude those from this check.
        interior = valid & valid_back
        assert interior.float().mean() > 0.99
        assert err[interior].max().item() < 1e-4

    def test_seam_regression_via_ut_covariance(self):
        """A Gaussian centered exactly on the seam should not blow up its
        estimated screen-space x-variance (the has_periodic_image_x_axis /
        sigma-point-unwrap fix). Regression test for that fix specifically.
        """
        from gsplat.cuda._torch_impl_ut import _fully_fused_projection_with_ut

        W, H = 256, 128
        means = torch.tensor([[0.0, 0.0, -5.0]], device=device)  # yaw = pi: on the seam
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        scales = torch.ones(1, 3, device=device) * 0.3
        viewmats = torch.eye(4, device=device)[None]
        Ks = torch.eye(3, device=device)[None]

        radii, means2d, depths, conics, compensations = _fully_fused_projection_with_ut(
            means,
            quats,
            scales,
            None,  # opacities
            viewmats,
            Ks,
            W,
            H,
            eps2d=0.3,
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=0.0,
            calc_compensations=False,
            camera_model="equirectangular",
            global_z_order=False,
        )
        # Without the seam unwrap fix, this Gaussian's estimated x-variance
        # (and therefore radius) would be on the order of the image width (or
        # even culled to radii=0, since the "unwrapped" mean can land far
        # outside the image); with the fix it should be valid and small,
        # matching its actual small angular size.
        rx = radii[0, 0, 0].item()
        assert rx > 0, "Gaussian was unexpectedly culled (radii=0)"
        assert rx < W // 4, "seam unwrap fix appears not to be working (radius too large)"

    def test_pole_no_nan(self):
        cam = self._make_camera()
        for ray in ([0.0, 1.0, 0.0], [0.0, -1.0, 0.0]):
            r = torch.tensor([ray], device=device)
            ip, valid = cam.camera_ray_to_image_point(r, margin_factor=0.0)
            assert not torch.isnan(ip).any()
            assert not torch.isinf(ip).any()

            ray_back, valid_back = cam.image_point_to_camera_ray(ip)
            assert not torch.isnan(ray_back).any()


@pytest.fixture
def test_data():
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "viewmats": viewmats,
        "colors": colors,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_rasterization_smoke(test_data):
    """End-to-end smoke test: render a real scene's Gaussians through the
    equirectangular camera model and check for a sane, NaN-free output.
    Mirrors tests/test_ftheta.py's structure.
    """
    from gsplat.rendering import rasterization

    C = test_data["viewmats"].shape[0]
    colors = test_data["colors"].repeat(C, 1, 1)

    width, height = 512, 256  # 2:1 aspect, as real equirectangular panoramas use
    Ks = torch.eye(3, device=device)[None].expand(C, 3, 3).contiguous()

    renders, alphas, meta = rasterization(
        means=test_data["means"],
        quats=test_data["quats"],
        scales=test_data["scales"],
        opacities=test_data["opacities"],
        colors=colors,
        viewmats=test_data["viewmats"],
        Ks=Ks,
        width=width,
        height=height,
        render_mode="RGB",
        camera_model="equirectangular",
        packed=False,
        with_ut=True,
        with_eval3d=True,
        global_z_order=False,
    )

    assert renders.shape == (C, height, width, 3)
    assert not torch.isnan(renders).any()
    assert not torch.isinf(renders).any()
    assert not torch.isnan(alphas).any()
    # Some Gaussians should actually be visible somewhere in the panorama.
    assert renders.sum().item() > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_rasterization_gradients():
    """Backward pass sanity check: means/quats/scales/opacities/colors must
    all receive finite gradients (regression test for the with_eval3d
    requirement discovered while validating this feature: with_ut alone does
    not backprop into quats/scales for *any* camera model)."""
    from gsplat.rendering import rasterization

    means = torch.tensor([[0.0, 0.0, 5.0]], device=device, requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True)
    scales = (torch.ones(1, 3, device=device) * 0.3).detach().requires_grad_(True)
    opacities = torch.ones(1, device=device, requires_grad=True)
    colors = torch.ones(1, 3, device=device, requires_grad=True)
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.eye(3, device=device)[None]

    renders, _, _ = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks, 64, 32,
        camera_model="equirectangular", with_ut=True, with_eval3d=True,
        global_z_order=False, packed=False,
    )
    renders.sum().backward()

    for name, p in [
        ("means", means),
        ("quats", quats),
        ("scales", scales),
        ("opacities", opacities),
        ("colors", colors),
    ]:
        assert p.grad is not None, f"{name} received no gradient"
        assert not torch.isnan(p.grad).any(), f"{name} gradient has NaN"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_rasterization_requires_global_z_order_false():
    """A Gaussian behind the camera (z<=0) must still be visible with
    global_z_order=False - equirectangular is omnidirectional. With the
    default global_z_order=True it would be wrongly culled."""
    from gsplat.rendering import rasterization

    means = torch.tensor([[0.0, 0.0, -5.0]], device=device)  # behind the camera
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.ones(1, 3, device=device) * 0.3
    opacities = torch.ones(1, device=device)
    colors = torch.ones(1, 3, device=device)
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.eye(3, device=device)[None]

    renders, _, info = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks, 64, 32,
        camera_model="equirectangular", with_ut=True, with_eval3d=True,
        global_z_order=False, packed=False,
    )
    assert info["radii"].max().item() > 0, "behind-camera Gaussian was wrongly culled"
