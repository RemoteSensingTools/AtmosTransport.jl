#!/usr/bin/env python3
"""Run the prepared 46-tracer ensemble, with a three-day budget gate first.

Invoke from an immutable source snapshot. CUDA_VISIBLE_DEVICES must select the
desired physical GPU before launch. Logs and status survive terminal closure.
"""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root', type=Path, required=True)
    ap.add_argument('--julia', required=True)
    ap.add_argument('--reuse-smoke', action='store_true',
                    help='Revalidate existing completed smoke outputs instead of rerunning transport')
    args = ap.parse_args()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logs = root/'logs'
    logs.mkdir(exist_ok=True)
    status = {'controller_pid': os.getpid(), 'source_snapshot': str(Path.cwd()),
              'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}

    def save(phase, **kwargs):
        status.update(phase=phase, updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(), **kwargs)
        tmp = root/'status.partial.json'
        tmp.write_text(json.dumps(status, indent=2)+'\n')
        tmp.replace(root/'status.json')
        print(json.dumps(status), flush=True)

    def run(phase, cmd, logname):
        with (logs/logname).open('w') as log:
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            save(phase, child_pid=child.pid, log=str(logs/logname))
            code = child.wait()
        if code:
            raise RuntimeError(f'{phase} exited {code}; see {logs/logname}')

    try:
        save('preflight')
        manifest = json.loads((root/'fluxes/manifest.json').read_text())
        good = [v for v in manifest['models'].values() if v['status'] == 'complete']
        assert len(good) == 23, f'Expected 23 prepared models, found {len(good)}'
        assert all(Path(v['output']).is_file() for v in good)
        for subdir in (('output',) if args.reuse_smoke else ('smoke_output','output')):
            assert not any((root/subdir).glob('*.nc')), f'{subdir} already has output; use a fresh run directory'
        template = Path('config/runs/trendy_v14_s3_all_models_npp_rh_c30_2014_2024.toml').read_text()
        os.environ['TRENDY_RUN_ROOT'] = str(root)
        config = os.path.expandvars(template)
        assert '$' not in config, 'Unresolved environment variable in config'
        (root/'run.toml').write_text(config)
        (root/'smoke.toml').write_text(config.replace('end_date = "2024-12-31"', 'end_date = "2014-09-03"')
                                     .replace('/output/', '/smoke_output/'))
        julia = [args.julia, '--project=.', '--threads=4', 'scripts/run_transport.jl']
        checker = [sys.executable, 'scripts/diagnostics/check_trendy_npp_rh_run.py', '--run-root', str(root)]
        if not args.reuse_smoke:
            run('smoke', julia+[str(root/'smoke.toml')], 'smoke.log')
        run('smoke_validation', checker+['--output-dir', str(root/'smoke_output'),
            '--end-date', '2014-09-03', '--report', str(root/'smoke_validation.json')], 'smoke_validation.log')
        run('transport', julia+[str(root/'run.toml')], 'transport.log')
        run('final_validation', checker+['--output-dir', str(root/'output'),
            '--end-date', '2024-12-31', '--report', str(root/'validation.json')], 'validation.log')
        save('complete', child_pid=None)
    except Exception as exc:
        save('failed', error=str(exc), child_pid=None)
        raise


if __name__ == '__main__':
    main()
