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

"""Data-loading smoke test for the equirectangular (360 panorama) support
added to examples/datasets/colmap.py's Parser.

Builds a minimal synthetic COLMAP sparse/ directory (1 EQUIRECTANGULAR
camera, 1 image, 0 3D points) and checks that Parser.__init__ completes
without exception, producing empty distortion params, the right image
size, and a syntactically valid (if physically meaningless) 3x3 K.

This does not depend on CUDA, torch's CameraWrappers, or a GPU - only on
the vendored pycolmap package and Pillow.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import os
import struct
import sys

import pytest

_EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "../examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)


def _write_synthetic_equirect_colmap_dir(
    data_dir: str, width: int = 64, height: int = 32
) -> None:
    """Writes a minimal synthetic COLMAP dataset: sparse/{cameras,images,points3D}.bin
    plus a matching dummy image file, with a single EQUIRECTANGULAR (type 17) camera.
    """
    from PIL import Image as PILImage

    sparse_dir = os.path.join(data_dir, "sparse")
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # cameras.bin: 1 EQUIRECTANGULAR camera (type 17), 0 params.
    with open(os.path.join(sparse_dir, "cameras.bin"), "wb") as f:
        f.write(struct.pack("L", 1))
        f.write(struct.pack("IiLL", 1, 17, width, height))

    # images.bin: 1 image, identity pose, 0 2D points.
    with open(os.path.join(sparse_dir, "images.bin"), "wb") as f:
        f.write(struct.pack("L", 1))
        image_struct = struct.Struct("<I 4d 3d I")
        f.write(image_struct.pack(1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1))
        f.write(b"img0.jpg\x00")
        f.write(struct.pack("Q", 0))  # num_points2D = 0

    # points3D.bin: 0 points.
    with open(os.path.join(sparse_dir, "points3D.bin"), "wb") as f:
        f.write(struct.pack("Q", 0))

    # Parser.__init__ reads this to cross-check against the COLMAP-registered
    # camera resolution, so it must be a valid image at exactly (width, height).
    PILImage.new("RGB", (width, height)).save(os.path.join(images_dir, "img0.jpg"))


def test_parser_loads_equirectangular_camera(tmp_path):
    from datasets.colmap import Parser

    data_dir = str(tmp_path)
    _write_synthetic_equirect_colmap_dir(data_dir, width=64, height=32)

    parser = Parser(data_dir=data_dir, factor=1, normalize=False)

    camera_id = parser.camera_ids[0]
    params = parser.params_dict[camera_id]
    imsize = parser.imsize_dict[camera_id]
    K = parser.Ks_dict[camera_id]

    assert len(params) == 0, f"expected empty params for equirectangular, got {params}"
    assert imsize == (64, 32), f"expected (64, 32), got {imsize}"
    assert K.shape == (3, 3), f"expected 3x3 K, got shape {K.shape}"
    # No lens distortion exists for a spherical panorama, so this camera
    # should never enter the undistortion-map path.
    assert camera_id not in parser.mapx_dict
    assert camera_id not in parser.mapy_dict
