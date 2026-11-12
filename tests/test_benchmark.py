"""End-to-end CPU integration: real subprocess orchestration, PLYs and reports."""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_reconstructions as benchmark
import reconstruction.training as training
from reconstruction.io import load_json
from reconstruction.training import point_cloud_path
import evaluate_metrics as evaluation
from test_gaussian_initialization import write_gaussians


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'benchmark'
        self.object_dir = self.root / 'views' / 'Wave Vase'
        global_dir = self.object_dir / 'global_transforms'
        global_dir.mkdir(parents=True)
        frames = [{'file_path': f'images/train/{i}', 'transform_matrix': [[1.,0,0,i],[0,1.,0,0],[0,0,1.,0],[0,0,0,1.]]}
                  for i in range(100)]
        train = {'camera_angle_x': 0.8, 'frames': frames}
        test = {'camera_angle_x': 0.8, 'frames': [dict(frames[0], file_path='images/test/0')]}
        for name, manifest in [('train', train), ('test', test)]:
            (global_dir / f'transforms_{name}.json').write_text(json.dumps(manifest))
            for frame in manifest['frames']:
                image = self.object_dir / (frame['file_path'] + '.png')
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b'image input bytes')
        self.experiments = self.root / 'experiments'
        self.experiments.mkdir()
        (self.experiments/'context.json').write_text(json.dumps({
            'id': 'context', 'train': {'strategy': 'furthest', 'count': 8, 'seed': 77}}))
        self.fastgs = self.root / 'FastGS'
        self.fastgs.mkdir()
        seed = self.root / 'seed.ply'
        write_gaussians(seed)
        fake = '''
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--source_path'); p.add_argument('--model_path'); p.add_argument('--iterations',type=int)
p.add_argument('--init_ply'); p.add_argument('--test_iterations',nargs='*')
a,_=p.parse_known_args()
source, model=Path(a.source_path), Path(a.model_path)
manifest=json.loads((source/'transforms_train.json').read_text())
if a.init_ply:
    assert Path(a.init_ply).is_file()
with (Path(__file__).parent/'calls.jsonl').open('a') as f:
    f.write(json.dumps({'path': str(model), 'iterations': a.iterations,
                        'init': a.init_ply, 'frames':manifest['frames']})+'\\n')
output=model/'point_cloud'/f'iteration_{a.iterations}'/'point_cloud.ply'
output.parent.mkdir(parents=True,exist_ok=True)
output.write_bytes(Path(SEED_PLY).read_bytes())
'''.replace('SEED_PLY', repr(str(seed)))
        (self.fastgs/'train.py').write_text(fake)
        for file in ('scene/gaussian_model.py','scene/__init__.py','utils/fast_utils.py',
                     'arguments/__init__.py','gaussian_renderer/__init__.py'):
            path=self.fastgs/file
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('# test checkout fingerprint\n')
        self.config = self.root/'config.json'
        self.configuration = {
            'schema_version':1, 'reference':'fresh', 'base_iterations':12,
            'candidates':[{'strategy':'blindspot'},{'strategy':'next','count':1},{'strategy':'random','count':1}],
            'methods':[{'id':'fresh','initialization':'scratch','iterations':12},
                       {'id':'warm','initialization':'base','iterations':7},
                       {'id':'extra','initialization':'base','iterations':7,'extra_gaussians':16}]}
        self.config.write_text(json.dumps(self.configuration))
        self.argv = ['--config',str(self.config),'--output-root',str(self.output),
                     '--views-root',str(self.object_dir.parent),'--experiments-dir',str(self.experiments),
                     '--fastgs-dir',str(self.fastgs),'--skip-preflight']

    def calls(self):
        return [json.loads(line) for line in (self.fastgs/'calls.jsonl').read_text().splitlines()]

    def fake_metrics(self, path, dataset, indices, iterations, **kwargs):
        value = 20 if len(indices)==8 else (21 if 'fresh' in path.parts else 20.8)
        values = {'PSNR':value, 'SSIM':0.8 if 'fresh' in path.parts else 0.82, 'LPIPS':0.2}
        if len(indices)==8:
            values={'PSNR':20.,'SSIM':0.7,'LPIPS':0.3}
        return {'averages':values,'per_view':{'00000':values}}

    def run_benchmark(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(training, 'ADAPTER_SCRIPT', self.fastgs/'train.py'), \
                patch.object(evaluation, 'evaluate_reconstruction', side_effect=self.fake_metrics):
            return benchmark.main(self.argv + list(extra))

    def test_full_suite_has_shared_original_base_extra_points_and_matched_reports(self):
        report=self.run_benchmark()
        calls=self.calls()
        self.assertEqual(len(calls),10)  # one base, three candidates x three methods
        self.assertEqual(sum('shared_base' in call['path'] for call in calls),1)
        for method in ('fresh','warm','extra'):
            record=load_json(self.output/f'results/{method}/Wave Vase/context/results.json')
            self.assertEqual(record['protocol']['gain_reference'],'original_base')
            self.assertEqual(record['reconstructions']['base']['iterations'],12)
            self.assertNotIn('base_control',record['reconstructions'])
        warm=load_json(self.output/'results/warm/Wave Vase/context/results.json')
        extra=load_json(self.output/'results/extra/Wave Vase/context/results.json')
        self.assertEqual(extra['initialization']['original_base_sha256'],warm['initialization']['original_base_sha256'])
        self.assertEqual(extra['initialization']['preparation']['initial_gaussians'],24)
        self.assertEqual(report['overall']['warm']['paired_candidates'],3)
        self.assertAlmostEqual(report['overall']['warm']['mean_delta_gain']['PSNR'],-0.2)
        self.assertTrue((self.output/'report/comparison.csv').is_file())
        self.assertTrue((self.output/'report/comparison.json').is_file())
        self.assertFalse((self.object_dir/'transforms_train.json').exists())
        self.run_benchmark()
        self.assertEqual(len(self.calls()),10)

    def test_changed_augmentation_retrains_only_that_method_and_invalidates_metrics(self):
        self.run_benchmark()
        self.configuration['methods'][2]['extra_gaussian_jitter']=0.7
        self.config.write_text(json.dumps(self.configuration))
        self.run_benchmark()
        self.assertEqual(len(self.calls()),13)
        self.assertTrue(all('/extra/' in call['path'] for call in self.calls()[-3:]))

    def test_dry_run_is_non_mutating_and_report_only_needs_no_gpu_or_source_data(self):
        before=sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*'))
        output=io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(training, 'ADAPTER_SCRIPT', self.fastgs/'train.py'):
            benchmark.main(self.argv+['--dry-run'])
        self.assertEqual(output.getvalue().count('"command": ['),10)
        after=sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*'))
        self.assertEqual(before,after)
        self.run_benchmark()
        self.run_benchmark('--report-only','--views-root',str(self.root/'not-present'),
                           '--fastgs-dir',str(self.root/'no-gpu-checkout'))
        self.assertEqual(len(self.calls()),10)

    def test_separate_base_control_is_not_the_gain_reference(self):
        report=self.run_benchmark('--base-control')
        self.assertEqual(len(self.calls()),12)
        record=load_json(self.output/'results/warm/Wave Vase/context/results.json')
        self.assertIn('base_control',record['reconstructions'])
        self.assertNotEqual(record['reconstructions']['base']['reconstruction_path'],
                            record['reconstructions']['base_control']['reconstruction_path'])
        self.assertEqual(report['overall']['warm']['paired_candidates'],3)
        self.assertGreater(report['comparisons'][1]['summary']['base_control_seconds'],0)

    def test_metric_cache_is_invalidated_by_checkpoint_or_test_image_changes(self):
        self.run_benchmark()
        arguments=['--results-root',str(self.output/'results/warm')]
        with patch.object(evaluation,'evaluate_reconstruction',side_effect=self.fake_metrics) as evaluate:
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation.main(arguments)
            evaluate.assert_not_called()
            (self.object_dir/'images/test/0.png').write_bytes(b'new held-out pixels')
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation.main(arguments)
            self.assertEqual(evaluate.call_count,4)
            evaluate.reset_mock()
            record=load_json(self.output/'results/warm/Wave Vase/context/results.json')
            candidate=Path(record['reconstructions']['blindspot-1-1']['reconstruction_path'])
            checkpoint=point_cloud_path(candidate,7)
            checkpoint.write_bytes(checkpoint.read_bytes()+b'\n')
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation.main(arguments)
            self.assertEqual(evaluate.call_count,1)

    def test_evaluation_subprocesses_use_current_paths_and_restore_both_manifests(self):
        self.run_benchmark('--train-only')
        (self.object_dir/'transforms_train.json').write_bytes(b'original train manifest')
        (self.object_dir/'transforms_test.json').write_bytes(b'original test manifest')
        (self.fastgs/'render.py').write_text('''
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--model_path');p.add_argument('--source_path');p.add_argument('--iteration');p.add_argument('--skip_train',action='store_true')
a=p.parse_args()
source,model=Path(a.source_path),Path(a.model_path)
assert len(json.loads((source/'transforms_train.json').read_text())['frames'])==2
assert len(json.loads((source/'transforms_test.json').read_text())['frames'])==1
(model/'test'/f'ours_{a.iteration}').mkdir(parents=True,exist_ok=True)
''')
        (self.fastgs/'metrics.py').write_text('''
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model_paths');a=p.parse_args();root=Path(a.model_paths)
key=next((root/'test').iterdir()).name
values={'PSNR':22.,'SSIM':0.8,'LPIPS':0.2}
(root/'results.json').write_text(json.dumps({key:values}))
(root/'per_view.json').write_text(json.dumps({key:{m:{'00000.png':v} for m,v in values.items()}}))
''')
        reconstruction=self.output/'reconstructions/warm/Wave Vase/context/blindspot-1-1'
        metrics=evaluation.evaluate_reconstruction(reconstruction,self.object_dir,[0,1],7,
                                                  fastgs_dir=self.fastgs,python=sys.executable)
        self.assertEqual(metrics['averages']['PSNR'],22.)
        self.assertEqual(metrics['per_view']['00000']['LPIPS'],0.2)
        self.assertEqual((self.object_dir/'transforms_train.json').read_bytes(),b'original train manifest')
        self.assertEqual((self.object_dir/'transforms_test.json').read_bytes(),b'original test manifest')

    def test_existing_root_manifests_survive_the_complete_benchmark(self):
        train=self.object_dir/'transforms_train.json'
        test=self.object_dir/'transforms_test.json'
        train.write_bytes(b'preserved train')
        test.write_bytes(b'preserved test')
        self.run_benchmark()
        self.assertEqual(train.read_bytes(),b'preserved train')
        self.assertEqual(test.read_bytes(),b'preserved test')

    def test_config_errors_fail_before_any_work(self):
        self.configuration['methods'][1]['lr_sclae']=0.1
        self.config.write_text(json.dumps(self.configuration))
        with self.assertRaises(ValueError):
            benchmark.load_config(self.config)
        self.assertFalse(self.output.exists())

    def add_matrix_inputs(self):
        shutil.copytree(self.object_dir, self.object_dir.parent / 'Table')
        (self.experiments / 'second.json').write_text(json.dumps({
            'id': 'second', 'train': {'strategy': 'random', 'count': 8, 'seed': 88}}))

    def test_multiple_objects_and_experiments_run_the_complete_default_matrix_and_resume(self):
        self.add_matrix_inputs()
        self.config.write_text(json.dumps(benchmark.load_config(benchmark.DEFAULT_CONFIG)))
        selection = ['--object-ids', 'Wave Vase', 'Table', 'Table',
                     '--experiment-ids', 'context', 'second']
        report = self.run_benchmark(*selection)
        self.assertEqual(set(report['overall']), {'base-cloud-3k'})
        self.assertEqual(report['overall']['base-cloud-3k']['contexts'], 4)
        self.assertEqual(report['overall']['base-cloud-3k']['paired_candidates'], 12)
        self.assertEqual(len(self.calls()), 28)  # four bases + 4 contexts x 3 candidates x 2 methods
        self.assertEqual(sum('shared_base' in call['path'] for call in self.calls()), 4)
        for call in self.calls():
            self.assertEqual(call['iterations'], 3000 if call['init'] else 30000)
        self.run_benchmark(*selection)
        self.assertEqual(len(self.calls()), 28)
        saved = load_json(self.output / 'benchmark_manifest.json')
        self.assertEqual({(c['object_id'], c['experiment_id']) for c in saved['contexts']},
                         {(o, e) for o in ('Wave Vase', 'Table') for e in ('context', 'second')})

    def test_config_selection_and_cli_override(self):
        self.add_matrix_inputs()
        self.configuration.update(objects=['Table'], experiments=['second'])
        self.config.write_text(json.dumps(self.configuration))
        report = self.run_benchmark()
        self.assertEqual({(c['object_id'], c['experiment_id']) for c in report['comparisons']},
                         {('Table', 'second')})
        report = self.run_benchmark('--object-id', 'Wave Vase', '--experiment-id', 'context')
        self.assertEqual({(c['object_id'], c['experiment_id']) for c in report['comparisons']},
                         {('Wave Vase', 'context')})

    def test_evaluation_and_reports_replay_exact_saved_contexts_and_methods(self):
        self.add_matrix_inputs()
        self.run_benchmark('--object-ids', 'Wave Vase', 'Table',
                           '--experiment-ids', 'context', 'second',
                           '--max-contexts', '3', '--method', 'warm', '--train-only')
        manifest_before = (self.output / 'benchmark_manifest.json').read_bytes()
        expected = {('Wave Vase', 'context'), ('Wave Vase', 'second'), ('Table', 'context')}
        calls_before = len(self.calls())
        # Omit the custom config and input selectors: both come from the saved run.
        replay = ['--output-root', str(self.output), '--experiments-dir', str(self.experiments),
                  '--fastgs-dir', str(self.fastgs), '--skip-preflight']
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(evaluation, 'evaluate_reconstruction', side_effect=self.fake_metrics):
            report = benchmark.main(replay + ['--evaluate-only'])
        self.assertEqual({(c['object_id'], c['experiment_id']) for c in report['comparisons']}, expected)
        self.assertEqual(set(report['overall']), {'warm'})
        self.assertEqual(len(self.calls()), calls_before)
        report = self.run_benchmark('--report-only', '--views-root', str(self.root / 'absent'))
        self.assertEqual({(c['object_id'], c['experiment_id']) for c in report['comparisons']}, expected)
        self.assertEqual((self.output / 'benchmark_manifest.json').read_bytes(), manifest_before)

    def test_invalid_matrix_selection_fails_before_creating_outputs(self):
        for selection in (['--object-ids', 'missing'], ['--experiment-ids', 'missing']):
            with self.subTest(selection=selection), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.run_benchmark(*selection)
                self.assertFalse(self.output.exists())

    def test_duplicate_experiment_ids_are_rejected_before_training(self):
        shutil.copyfile(self.experiments / 'context.json', self.experiments / 'duplicate.json')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_benchmark()
        self.assertFalse(self.output.exists())

    def test_evaluation_only_rejects_changed_training_inputs(self):
        self.run_benchmark('--train-only')
        (self.object_dir / 'images/train/0.png').write_bytes(b'changed training pixels')
        before = len(self.calls())
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit):
            self.run_benchmark('--evaluate-only')
        self.assertIn('Inputs changed', errors.getvalue())
        self.assertEqual(len(self.calls()), before)

    def test_documented_validation_preset_plans_nine_contexts_with_controls_without_writing(self):
        shutil.copytree(self.object_dir, self.object_dir.parent / 'Table')
        shutil.copytree(self.object_dir, self.object_dir.parent / 'LEGO Miss Liberty (small)')
        for count in (3, 5, 10):
            identifier = f'random_{count}_seed_77'
            (self.experiments / f'{identifier}.json').write_text(json.dumps({
                'id': identifier, 'train': {'strategy': 'random', 'count': count, 'seed': 77}}))
        preset = Path(__file__).resolve().parents[1] / 'benchmark_configs/validation.json'
        self.config.write_text(json.dumps(benchmark.load_config(preset)))
        before = sorted(str(path.relative_to(self.root)) for path in self.root.rglob('*'))
        snapshot = self.run_benchmark('--base-control', '--dry-run')
        self.assertEqual(len(snapshot['contexts']), 9)
        self.assertEqual(snapshot['selected_methods'], ['fresh-30k', 'base-cloud-3k'])
        after = sorted(str(path.relative_to(self.root)) for path in self.root.rglob('*'))
        self.assertEqual(before, after)


if __name__=='__main__':
    unittest.main()
