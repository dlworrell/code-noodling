# SPDX-FileCopyrightText: 2026 Donovan Worrell
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:7cff44bf-2863-43b0-bfaa-3cad184db7e8

import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/ci/project_release.py'
PROJECTS = {'prime-number-suite': 'prime-number-suite', 'crypto-pipeline': 'crypto_pipeline',
            'election-analysis': 'election-analysis'}
def git(*args, cwd=ROOT, env=None):
    return subprocess.check_output(['git', *args], cwd=cwd, env=env, text=True).strip()



class ReleaseTest(unittest.TestCase):
    def call(self, *args, ok=True, cwd=ROOT):
        result = subprocess.run(['python3', str(SCRIPT), *args], cwd=cwd,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, ok, result.stderr)
        return result.stdout

    def test_namespace_guards_and_prerelease(self):
        for project in PROJECTS:
            selection = json.loads(self.call('select', project, '1.2.3-rc.1', 'refs/heads/main'))
            self.assertEqual(selection['tag'], f'{project}/v1.2.3-rc.1')
            self.assertEqual(selection['root'], PROJECTS[project])
            self.assertTrue(selection['prerelease'])
        for project, version, ref in [('crypto_pipeline', '1.2.3', 'refs/heads/main'),
                                      ('crypto-pipeline', '1.2.3-', 'refs/heads/main'),
                                      ('crypto-pipeline', '1.2.3', 'refs/heads/dev-daily')]:
            self.call('select', project, version, ref, ok=False)

    def test_ruleset_requires_complete_immutable_evidence(self):
        rule = {'target': 'tag', 'enforcement': 'active', 'bypass_actors': [],
                'conditions': {'ref_name': {'include': ['refs/tags/crypto-pipeline/v*'], 'exclude': []}},
                'rules': [{'type': x} for x in ('update', 'deletion', 'non_fast_forward')]}
        self.call('rules', 'crypto-pipeline', json.dumps(rule))
        for change in [{'enforcement': 'evaluate'}, {'bypass_actors': [{}]},
                       {'conditions': {'ref_name': {'include': ['refs/tags/v*'], 'exclude': []}}},
                       {'conditions': {'ref_name': {'include': ['refs/tags/crypto-pipeline/v*'],
                                                   'exclude': ['refs/tags/crypto-pipeline/v3*']}}},
                       {'rules': [{'type': 'deletion'}]}]:
            self.call('rules', 'crypto-pipeline', json.dumps(rule | change), ok=False)
        rule.pop('bypass_actors')
        self.call('rules', 'crypto-pipeline', json.dumps(rule), ok=False)

    def test_project_archives_are_isolated_deterministic_and_licensed(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / 'repo'
            git('-c', 'advice.detachedHead=false', 'clone', '--quiet', '--shared', str(ROOT), str(repo))
            for name in ('Apache-2.0.txt', 'MIT.txt'):
                (repo / 'LICENSES' / name).write_bytes((ROOT / 'LICENSES' / name).read_bytes())
            git('add', '--', 'LICENSES', cwd=repo)
            git('-c', 'user.name=Test Fixture', '-c', 'user.email=fixture@example.invalid',
                '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-qm', 'license fixture', cwd=repo)
            sha = git('rev-parse', 'HEAD', cwd=repo)
            self.check_archives(repo, sha)

    def check_archives(self, repo, sha):
        for project, scope in PROJECTS.items():
            with self.subTest(project=project), tempfile.TemporaryDirectory() as directory:
                wrapper = ROOT / scope / 'ci/package-release.sh'
                # Run the real packaging helper in the isolated fixture checkout,
                # without publishing or making a commit in the working checkout.
                for name in ('package-project-source.sh', 'project_release.py'):
                    target = repo / 'scripts/ci' / name
                    target.write_bytes((ROOT / 'scripts/ci' / name).read_bytes())
                fixture_wrapper = repo / wrapper.relative_to(ROOT)
                fixture_wrapper.parent.mkdir(parents=True, exist_ok=True)
                fixture_wrapper.write_bytes(wrapper.read_bytes())
                paths = [Path(directory) / name for name in ('first.tar.gz', 'second.tar.gz')]
                for path in paths:
                    result = subprocess.run(['bash', str(repo / wrapper.relative_to(ROOT)), sha, '1.2.3', str(path)],
                                            cwd=repo, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
                with tarfile.open(paths[0]) as archive:
                    prefix = f'{project}-1.2.3/'
                    names = [member.name.rstrip('/') + '/' if member.isdir() else member.name
                             for member in archive.getmembers()]
                    self.assertTrue(all(name.startswith((prefix + scope + '/', prefix + 'LICENSES/'))
                                        or name == prefix for name in names))
                    self.assertIn(prefix + 'LICENSES/Apache-2.0.txt', names)
                    self.assertIn(prefix + 'LICENSES/MIT.txt', names)
                    if project == 'election-analysis':
                        excluded = ('data/crosswalks/', 'data/elections/', 'reports/', 'tests/fixtures/')
                        self.assertFalse(any(name.startswith(prefix + scope + '/' + entry)
                                             for name in names for entry in excluded))
                        self.assertIn(prefix + scope + '/data/watch.toml', names)

    def test_notes_use_the_latest_reachable_project_tag_or_scoped_history(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            git('init', '--quiet', cwd=repo)

            def commit(day, file, subject):
                path = repo / file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(subject + '\n', encoding='utf-8')
                git('add', '--', file, cwd=repo)
                date = f'2026-01-{day:02d}T12:00:00Z'
                git('-c', 'user.name=Test Fixture', '-c', 'user.email=fixture@example.invalid',
                    '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null',
                    'commit', '-qm', subject, cwd=repo,
                    env=os.environ | {'GIT_AUTHOR_DATE': date, 'GIT_COMMITTER_DATE': date})
                return git('rev-parse', 'HEAD', cwd=repo)

            commit(1, 'crypto_pipeline/before.txt', 'initial crypto change')
            older = commit(2, 'crypto_pipeline/older.txt', 'older crypto release')
            baseline = commit(3, 'crypto_pipeline/baseline.txt', 'latest crypto release')
            commit(4, 'crypto_pipeline/after.txt', 'new crypto change')
            commit(5, 'LICENSES/shared.txt', 'new shared license change')
            tested = commit(6, 'election-analysis/other.txt', 'unrelated election change')

            # A newer repository-wide tag must never become a project baseline.
            git('-c', 'tag.gpgSign=false', 'tag', 'v0.1.0', tested, cwd=repo)
            command = ('notes', 'crypto-pipeline', '3.0.0', tested)
            first_release = self.call(*command, cwd=repo)
            self.assertIn('No prior reachable project tag.', first_release)
            self.assertNotIn('v0.1.0', first_release)
            self.assertIn('initial crypto change', first_release)
            self.assertIn('new shared license change', first_release)
            self.assertNotIn('unrelated election change', first_release)

            git('-c', 'tag.gpgSign=false', 'tag', 'crypto-pipeline/v2.0.0', older, cwd=repo)
            git('-c', 'tag.gpgSign=false', 'tag', 'crypto-pipeline/v1.0.0', baseline, cwd=repo)
            git('-c', 'tag.gpgSign=false', 'tag', 'crypto-pipeline/vinvalid', tested, cwd=repo)
            git('-c', 'advice.detachedHead=false', 'checkout', '--quiet', '--detach', baseline, cwd=repo)
            unreachable = commit(7, 'crypto_pipeline/off-branch.txt', 'unreachable crypto change')
            git('-c', 'tag.gpgSign=false', 'tag', 'crypto-pipeline/v2.5.0', unreachable, cwd=repo)

            notes = self.call(*command, cwd=repo)
            self.assertIn('Changes since `crypto-pipeline/v1.0.0`', notes)
            self.assertIn('new crypto change', notes)
            self.assertIn('new shared license change', notes)
            for subject in ('initial crypto change', 'older crypto release', 'latest crypto release',
                            'unrelated election change', 'unreachable crypto change'):
                self.assertNotIn(subject, notes)

    def test_head_check_refuses_a_different_commit(self):
        sha = git('rev-parse', 'HEAD')
        self.call('main', sha, sha)
        self.call('main', sha, '0' * 40, ok=False)

    def test_tag_lookup_refuses_existing_and_unreachable_remote(self):
        sha = git('rev-parse', 'HEAD')
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / 'repo'
            git('-c', 'advice.detachedHead=false', 'clone', '--quiet', '--shared', str(ROOT), str(repo))
            target = 'prime-number-suite/v91.0.0'
            command = ['python3', str(SCRIPT), 'tag-absent', target]
            git('remote', 'set-url', 'origin', str(repo), cwd=repo)
            self.assertEqual(subprocess.run(command, cwd=repo, capture_output=True).returncode, 0)
            git('-c', 'tag.gpgSign=false', 'tag', target, sha, cwd=repo)  # fixture only, never the actual checkout
            self.assertNotEqual(subprocess.run(command, cwd=repo, capture_output=True).returncode, 0)
            git('remote', 'set-url', 'origin', str(repo / 'missing'), cwd=repo)
            self.assertNotEqual(subprocess.run(command, cwd=repo, capture_output=True).returncode, 0)
