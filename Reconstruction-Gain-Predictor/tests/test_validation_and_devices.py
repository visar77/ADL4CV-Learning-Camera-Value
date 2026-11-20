from pathlib import Path
from unittest.mock import patch
import contextlib
import io
import json
import tempfile
import unittest

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch

from gain_predictor.artifacts import atomic_save
from gain_predictor.checkpoints import load_run
from gain_predictor.data.splits import make_splits
from gain_predictor.devices import compute_precision, resolve_device
from gain_predictor.feature_cache import extract_features
from gain_predictor.scoring import predict_normalized, score_samples
from gain_predictor.training import RollingValidation, train
from tests.test_research_pipeline import TinyBackbone, example
from train import parse_args


class RollingValidationTests(unittest.TestCase):
    def test_cnn_training_and_validation_use_requested_db_target(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            # Each record gains 2 dB, or 0.1 in relative units. The normalized
            # target is zero in dB. A relative target here gives a huge loss.
            samples = [example(root, context, gain=0.1) for context in range(4)]
            manifest = root / "labels.jsonl"
            manifest.write_text("".join(json.dumps(s.to_json()) + "\n" for s in samples))
            args = parse_args([
                "--dataset", str(manifest), "--model", "toy", "--target", "delta-db",
                "--epochs", "1", "--device", "cpu", "--image-size", "8",
                "--output", str(root / "model.pt"), "--log-dir", str(root / "logs"),
            ])
            train(args)
            report = json.loads(args.output.with_suffix(".report.json").read_text())
            first_epoch = report["history"][0]
            self.assertLess(first_epoch["train_loss"], 1.0)
            self.assertLess(first_epoch["validation"]["loss"], 1.0)

    def test_weighted_window_budget_and_complete_cycle(self):
        values = torch.arange(5, dtype=torch.float32)[:, None]
        loader = DataLoader(TensorDataset(values, torch.zeros_like(values)), batch_size=2)
        visited = []
        def prediction(model, batch):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            visited.append(batch[0].flatten().tolist())
            return model(batch[0]), batch[1]
        monitor = RollingValidation(loader, prediction, window=2)
        model = nn.Identity().train()
        first, second, third, fourth = [monitor.update(model) for _ in range(4)]
        self.assertEqual(visited, [[0.,1.], [2.,3.], [4.], [0.,1.]])
        self.assertAlmostEqual(first['loss'], .5)
        self.assertAlmostEqual(second['loss'], 3.5)
        self.assertAlmostEqual(third['loss'], 29/3)
        self.assertAlmostEqual(fourth['loss'], 17/3)
        self.assertEqual(third['window_samples'], 3)
        self.assertEqual(third['completed_sweeps'], 1)
        self.assertTrue(all(v['batches'] == 1 and v['samples'] <= 2 for v in [first,second,third,fourth]))

    def test_validation_shuffle_does_not_consume_training_rng(self):
        values = torch.arange(9, dtype=torch.float32)[:, None]
        loader = DataLoader(TensorDataset(values, values), batch_size=2, shuffle=True,
                            generator=torch.Generator().manual_seed(91))
        monitor = RollingValidation(loader, lambda model,batch:(model(batch[0]),batch[1]), window=3)
        torch.manual_seed(42)
        before = torch.get_rng_state().clone()
        for _ in range(7):
            monitor.update(nn.Identity())
        torch.testing.assert_close(torch.get_rng_state(), before)

    def test_defaults_are_80_10_10_and_full_reports_only_at_end(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d)
            samples = [example(root, c, q, .1*q) for c in range(10) for q in range(3)]
            groups, _ = make_splits(samples)
            self.assertEqual([len(groups[k]) for k in ('train','validation','test')], [24,3,3])
            manifest = root/'data.jsonl'
            manifest.write_text(''.join(json.dumps(s.to_json())+'\n' for s in samples))
            cache = extract_features(samples, root/'cache', TinyBackbone, {'name':'tiny'}, image_size=8, device='cpu')
            args = parse_args(['--dataset',str(manifest),'--features',str(cache),'--model','cached',
                               '--epochs','5','--validation-batch-size','2','--validation-window','3',
                               '--scheduler','none','--early-stopping-patience','0','--device','cpu',
                               '--output',str(root/'model.pt'),'--log-dir',str(root/'logs')])
            self.assertEqual((args.validation_fraction,args.test_fraction),(.1,.1))
            with patch('gain_predictor.training.predict_normalized', wraps=predict_normalized) as full_evaluation:
                train(args)
                self.assertEqual(full_evaluation.call_count, 3)  # train/validation/test, once each at end
            report = json.loads(args.output.with_suffix('.report.json').read_text())
            self.assertEqual(report['monitor_kind'],'rolling_validation')
            for epoch in report['history']:
                self.assertLessEqual(epoch['validation']['samples'],2)
                self.assertLessEqual(epoch['validation']['window_batches'],3)
            self.assertEqual(report['metrics']['validation']['samples'],3)
            saved = torch.load(args.output, weights_only=True)  # no map_location needed for a portable artifact
            self.assertNotIn('device',saved)
            self.assertTrue(all(t.device.type=='cpu' for t in saved['model'].values()))


class DeviceTests(unittest.TestCase):
    def test_auto_selection_and_explicit_device_validation(self):
        with patch('torch.cuda.is_available',return_value=False):
            self.assertEqual(resolve_device(), 'cpu')
            with self.assertRaises(ValueError):
                resolve_device('cuda')
        with patch('torch.cuda.is_available',return_value=True), patch('torch.cuda.device_count',return_value=2):
            self.assertEqual(resolve_device(),'cuda')
            self.assertEqual(resolve_device('cuda:1'),'cuda:1')
            with self.assertRaises(ValueError):
                resolve_device('cuda:2')

    def test_bf16_is_not_bound_to_a_specific_gpu(self):
        self.assertEqual(compute_precision('bf16','cpu'),'bf16')
        notice = io.StringIO()
        with patch('torch.cuda.is_bf16_supported',return_value=False), contextlib.redirect_stderr(notice):
            self.assertEqual(compute_precision('bf16','cuda'),'fp32')
        self.assertIn('using FP32', notice.getvalue())
        with patch('torch.cuda.is_bf16_supported',return_value=True):
            self.assertEqual(compute_precision('bf16','cuda:1'),'bf16')

    def test_nested_saved_tensors_are_cpu_and_detached(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'artifact.pt'
            atomic_save({'a':torch.ones(2,requires_grad=True),'nested':[{'b':torch.zeros(1)}]},path)
            loaded = torch.load(path,weights_only=True)
            self.assertEqual(loaded['a'].device.type,'cpu')
            self.assertFalse(loaded['a'].requires_grad)
            self.assertEqual(loaded['nested'][0]['b'].device.type,'cpu')

    def test_complete_cli_cache_needs_no_backbone_weights(self):
        import cache_features
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d)
            samples = [example(root,0,q,.1*q) for q in range(3)]
            manifest = root/'data.jsonl'
            manifest.write_text(''.join(json.dumps(s.to_json())+'\n' for s in samples))
            weights = root/'weights.pt'
            weights.write_bytes(b'checkpoint identity only')
            from gain_predictor.artifacts import file_sha256
            cache = extract_features(samples, root/'cache', TinyBackbone,
                                     {'name':'tiny','checkpoint_sha256':file_sha256(weights),'latent_tokens':4},
                                     image_size=8, device='cpu', backbone_checkpoint=weights)
            weights.unlink()
            def forbidden(*args):
                raise AssertionError('A complete CLI cache must not initialize/download a backbone')
            with patch.dict(cache_features.BACKBONES, {'tiny':forbidden}):
                cache_features.main(['--dataset',str(manifest),'--output',str(root/'cache'),'--backbone','tiny',
                               '--lvsm-checkpoint',str(weights),'--image-size','8','--lvsm-latent-tokens','4',
                               '--device','cpu'])
            self.assertFalse(weights.exists())
            bundle = torch.load(cache,weights_only=True)
            self.assertEqual(bundle['timing']['reused_candidates'],3)
            self.assertEqual(bundle['timing']['encoded_contexts'],0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware is not available in this environment')
    def test_gpu_trained_head_loads_and_predicts_on_cpu(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d)
            samples = [example(root,0,q,.1*q) for q in range(3)]
            manifest = root/'data.jsonl'
            manifest.write_text(''.join(json.dumps(s.to_json())+'\n' for s in samples))
            cache = extract_features(samples,root/'cache',TinyBackbone,{'name':'tiny'},image_size=8,device='cpu')
            args = parse_args(['--dataset',str(manifest),'--features',str(cache),'--model','cached','--mode','overfit',
                               '--epochs','3','--device','cuda','--output',str(root/'head.pt'),'--log-dir',str(root/'logs')])
            train(args)
            raw = torch.load(args.output, weights_only=True)
            self.assertTrue(all(t.device.type=='cpu' for t in raw['model'].values()))
            cpu, cpu_model = load_run(args.output,'cpu')
            gpu, gpu_model = load_run(args.output,'cuda')
            torch.testing.assert_close(score_samples(cpu,cpu_model,samples,device='cpu'),
                                       score_samples(gpu,gpu_model,samples,device='cuda'), rtol=1e-4,atol=1e-5)


if __name__ == '__main__':
    unittest.main()
