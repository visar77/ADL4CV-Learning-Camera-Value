"""Golden numbers, direction conventions, and rejection of unmatched comparisons."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_reconstructions as comparison


def example(method=False):
    names = ['blindspot-1-1', 'next-1-1', 'random-1-1']
    delta = {'PSNR': [-1.4059, -0.0788, 0.3102], 'SSIM': [0.0459, 0.0214, 0.0590],
             'LPIPS': [0.0515, 0.0253, 0.0584]}
    base = {'PSNR': 20., 'SSIM': 0.5, 'LPIPS': 0.3}
    result = {'object_id': 'Wave Vase', 'experiment_id': 'furthest_8_seed_77', 'train_lattice_indices': [0, 1],
              'protocol': {'gain_reference': 'original_base', 'input_data_sha256': 'inputs'},
              'reconstructions': {}, 'metrics': {}, 'metrics_provenance': {}, 'initialization': None}
    for i, name in enumerate(['base'] + names):
        budget = 15000 if method and i else 30000
        index = i + 2 if i else None
        gain = {'PSNR': [1.4, 1.5, 1.5952][i-1] if i else 0.,
                'SSIM': 0.0087 if i else 0., 'LPIPS': 0.0129 if i else 0.}
        if method and i:
            gain = {key: value + delta[key][i-1] for key, value in gain.items()}
        values = {key: base[key] - gain[key] if key == 'LPIPS' else base[key] + gain[key] for key in base}
        seconds = ([80, 52.77, 53.60, 52.70] if method else [80, 101.844, 83.201, 81.2949])[i]
        result['reconstructions'][name] = {
            'status': 'complete', 'iterations': budget, 'reconstruction_path': f'/models/{method}/{name}',
            'train_lattice_indices': [0, 1] + ([index] if i else []),
            'timing': {'status': 'complete', 'wall_time_seconds': seconds}}
        if i:
            result['reconstructions'][name]['candidate_lattice_index'] = index
        result['metrics'][name] = {'averages': values, 'per_view': {'00000': values}}
        result['metrics_provenance'][name] = {'iteration': budget, 'checkpoint_sha256': 'same-base' if not i else name,
                                             'test_data_sha256': 'test-images', 'training_cameras_sha256': name}
    return result


class ComparisonTests(unittest.TestCase):
    def test_screenshot_values_and_metric_signs(self):
        context = comparison.compare_context(example(), example(True), 'fresh-30k', 'base-cloud-15k')
        s = context['summary']
        self.assertAlmostEqual(s['mean_delta_gain']['PSNR'], -0.3915)
        self.assertAlmostEqual(s['mean_delta_gain']['SSIM'], 0.0421)
        self.assertAlmostEqual(s['mean_delta_gain']['LPIPS'], (0.0515+0.0253+0.0584)/3)
        self.assertAlmostEqual(s['mean_reference_gain']['PSNR'], 1.4984)
        self.assertAlmostEqual(s['mean_method_gain']['PSNR'], 1.1069)
        self.assertAlmostEqual(s['speedup'], 266.3399/159.07)
        self.assertAlmostEqual(s['method_total_with_base_seconds'], 239.07)
        self.assertEqual(s['negative_psnr_gains_method'], 1)
        report = {'reference': 'fresh-30k', 'comparisons': [context], 'overall': {}, 'missing_contexts': []}
        with tempfile.TemporaryDirectory() as directory:
            text = comparison.write_report(report, directory)
            self.assertIn('-0.3915', text)
            self.assertIn('40.3% saved; 1.67x speedup', text)
            self.assertIn('Wave Vase', text)
            self.assertEqual(json.loads((Path(directory)/'comparison.json').read_text())['comparisons'][0]['summary'], s)
            self.assertTrue((Path(directory)/'comparison.csv').read_text().startswith('object_id,experiment_id'))

    def test_rejects_different_base_cameras_test_images_and_iteration(self):
        for kind in ('base', 'views', 'test', 'iteration', 'cameras', 'baseline-definition'):
            new = example(True)
            if kind == 'base':
                new['metrics_provenance']['base']['checkpoint_sha256'] = 'different-base'
            elif kind == 'views':
                new['train_lattice_indices'] = [0, 99]
            elif kind == 'test':
                new['metrics_provenance']['blindspot-1-1']['test_data_sha256'] = 'different-test'
            elif kind == 'iteration':
                new['metrics_provenance']['blindspot-1-1']['iteration'] = 30000
            elif kind == 'cameras':
                new['metrics_provenance']['blindspot-1-1']['training_cameras_sha256'] = 'different-cameras'
            else:
                new['protocol']['gain_reference'] = 'equal_budget_base_control'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                comparison.compare_context(example(), new, 'fresh', 'new')

    def test_redundancy_and_duplicate_aliases_are_not_extra_samples_or_time(self):
        old, new = example(), example(True)
        for record in (old, new):
            for name, source, status in [('redundancy-1-1', 'base', 'redundant'), ('alias', 'blindspot-1-1', 'complete')]:
                record['reconstructions'][name] = copy.deepcopy(record['reconstructions'][source])
                record['reconstructions'][name].update(status=status, candidate_lattice_index=0 if source == 'base' else 3)
                record['metrics'][name] = record['metrics'][source]
                record['metrics_provenance'][name] = record['metrics_provenance'][source]
        result = comparison.compare_context(old, new, 'fresh', 'new')
        self.assertEqual(result['summary']['paired_candidates'], 3)
        self.assertAlmostEqual(result['summary']['reference_candidate_seconds'], 266.3399)

    def test_missing_timing_is_unknown_and_missing_candidates_fail_by_default(self):
        new = example(True)
        new['reconstructions']['blindspot-1-1']['timing'] = {}
        context = comparison.compare_context(example(), new, 'fresh', 'new')
        self.assertIsNone(context['summary']['speedup'])
        new['reconstructions'].pop('next-1-1')
        with self.assertRaises(ValueError):
            comparison.compare_context(example(), new, 'fresh', 'new')
        context = comparison.compare_context(example(), new, 'fresh', 'new', allow_incomplete=True)
        self.assertEqual(len(context['rows']), 2)
        self.assertEqual(context['missing_from_method'], [4])

    def test_timing_environment_mismatch_is_rejected(self):
        old, new = example(), example(True)
        old['reconstructions']['blindspot-1-1']['timing']['environment'] = {'gpu_name': 'GPU A'}
        new['reconstructions']['blindspot-1-1']['timing']['environment'] = {'gpu_name': 'GPU B'}
        with self.assertRaisesRegex(ValueError, 'Timing environments differ'):
            comparison.compare_context(old, new, 'fresh', 'new')

    def test_ranking_ties_have_no_favorable_tiebreak(self):
        old, new = example(), example(True)
        for name in ('blindspot-1-1', 'next-1-1', 'random-1-1'):
            new['metrics'][name]['averages']['PSNR'] = 21
        result = comparison.compare_context(old, new, 'fresh', 'new')
        self.assertIsNone(result['summary']['psnr_spearman'])
        self.assertAlmostEqual(result['summary']['selected_view_regret_db'], 1.5952-1.4984)


if __name__ == '__main__':
    unittest.main()
