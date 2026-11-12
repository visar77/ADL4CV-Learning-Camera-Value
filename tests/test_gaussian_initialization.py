"""Check deterministic augmentation, base preservation and valid Gaussian fields."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaussian_initialization import augment_ply, prepare_initialization, sha256


def write_gaussians(path, count=8):
    fields = [*( (name, 'f4') for name in ('x', 'y', 'z', 'nx', 'ny', 'nz')),
              *((f'f_dc_{i}', 'f4') for i in range(3)),
              *((f'f_rest_{i}', 'f4') for i in range(45)), ('opacity', 'f4'),
              *((f'scale_{i}', 'f4') for i in range(3)), *((f'rot_{i}', 'f4') for i in range(4))]
    vertices = np.zeros(count, dtype=fields)
    vertices['x'] = np.linspace(-0.3, 0.3, count)
    vertices['rot_0'] = 1
    vertices['opacity'] = 0
    for i in range(3):
        vertices[f'scale_{i}'] = np.log(0.1)
        vertices[f'f_dc_{i}'] = 0.2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(str(path))
    return vertices


class InitializationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'base.ply'
        self.original = write_gaussians(self.source)

    def test_adds_requested_count_preserves_base_and_is_reproducible(self):
        before = self.source.read_bytes()
        a, b = self.root / 'a.ply', self.root / 'b.ply'
        augment_ply(self.source, a, 3000, seed=17)
        augment_ply(self.source, b, 3000, seed=17)
        vertices = PlyData.read(str(a))['vertex'].data
        self.assertEqual(len(vertices), 3008)
        np.testing.assert_array_equal(vertices[:8], self.original)
        self.assertEqual(a.read_bytes(), b.read_bytes())
        self.assertEqual(self.source.read_bytes(), before)
        np.testing.assert_allclose(1/(1+np.exp(-vertices['opacity'][8:])), 0.02, atol=1e-8)
        for name in vertices.dtype.names:
            self.assertTrue(np.isfinite(vertices[name]).all())
        self.assertGreater(np.std(vertices['y'][8:]), 0)
        np.testing.assert_allclose(vertices['scale_0'][8:], np.log(0.05))

    def test_rotation_is_applied_to_local_jitter(self):
        # Make the x-axis much larger than the other axes, then rotate x to y.
        vertices = self.original.copy()
        vertices['x'] = 0
        vertices['scale_0'] = np.log(0.2)
        vertices['scale_1'] = vertices['scale_2'] = np.log(0.001)
        vertices['rot_0'] = vertices['rot_3'] = np.sqrt(0.5)
        PlyData([PlyElement.describe(vertices, 'vertex')]).write(str(self.source))
        output = self.root / 'rotated.ply'
        augment_ply(self.source, output, 1000, jitter=1)
        added = PlyData.read(str(output))['vertex'].data[8:]
        self.assertGreater(np.std(added['y']), 50 * np.std(added['x']))

    def test_content_addressed_cache_reuses_and_detects_corruption(self):
        output, receipt = prepare_initialization(self.source, self.root / 'cache', 16)
        modified = output.stat().st_mtime_ns
        again, same = prepare_initialization(self.source, self.root / 'cache', 16)
        self.assertEqual(output, again)
        self.assertEqual(receipt, same)
        self.assertEqual(output.stat().st_mtime_ns, modified)
        output.write_bytes(b'corrupted')
        repaired, receipt = prepare_initialization(self.source, self.root / 'cache', 16)
        self.assertEqual(sha256(repaired), receipt['ply_sha256'])
        self.assertEqual(len(PlyData.read(str(repaired))['vertex'].data), 24)

    def test_invalid_options_and_overwrite_are_rejected(self):
        for kwargs in ({'count': -1}, {'count': 2, 'opacity': 0}, {'count': 2, 'scale': 0},
                       {'count': 2, 'jitter': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                augment_ply(self.source, self.root / 'invalid.ply', **kwargs)
        with self.assertRaises(ValueError):
            augment_ply(self.source, self.source, 2)


if __name__ == '__main__':
    unittest.main()
