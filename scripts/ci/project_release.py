#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Donovan Worrell
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:914eb7b0-0d7c-425b-a0d5-2d1135d49812
"""Implement the release selector and fail-closed, read-only policy gate."""
import json
import re
import subprocess
import sys

PROJECTS = {
    'prime-number-suite': 'prime-number-suite',
    'crypto-pipeline': 'crypto_pipeline',
    'election-analysis': 'election-analysis',
}
NUMBER = r'(?:0|[1-9][0-9]*)'
IDENTIFIER = rf'(?:{NUMBER}|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)'
VERSION = re.compile(rf'{NUMBER}\.{NUMBER}\.{NUMBER}(?:-{IDENTIFIER}(?:\.{IDENTIFIER})*)?')


def select(project, version, ref):
    if project not in PROJECTS or ref != 'refs/heads/main' or not VERSION.fullmatch(version):
        raise ValueError('Expected a known project, a version such as 1.2.3 or 1.2.3-rc.1, and main')
    return {'project': project, 'root': PROJECTS[project], 'tag': f'{project}/v{version}',
            'asset': f'{project}-{version}.tar.gz', 'prerelease': '-' in version}


def validate_rules(project, rule):
    required = {'update', 'deletion', 'non_fast_forward'}
    refs = rule.get('conditions', {}).get('ref_name', {})
    if not (
        project in PROJECTS
        and rule.get('target') == 'tag'
        and rule.get('enforcement') == 'active'
        and rule.get('bypass_actors') == []
        and f'refs/tags/{project}/v*' in refs.get('include', [])
        and refs.get('exclude') == []
        and required.issubset({entry['type'] for entry in rule.get('rules', [])})
    ):
        raise ValueError(f'Cannot prove immutable no-bypass protection for {project} tags')


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def check_tag(tag):
    code = subprocess.run(['git', 'ls-remote', '--exit-code', '--tags', 'origin',
                           f'refs/tags/{tag}'], stdout=subprocess.DEVNULL).returncode
    if code != 2:
        raise ValueError(f'Tag lookup failed (exit {code}) or {tag} already exists')


def notes(project, version, sha):
    selection = select(project, version, 'refs/heads/main')
    tags = git('for-each-ref', '--merged', sha, '--sort=-creatordate',
               '--format=%(refname:strip=2)', f'refs/tags/{project}/').splitlines()
    previous = next((tag for tag in tags if tag.startswith(f'{project}/v') and
                     VERSION.fullmatch(tag.removeprefix(f'{project}/v'))), None)
    rev = f'{previous}..{sha}' if previous else sha
    history = git('log', '--no-merges', '--format=- %h %s', rev, '--', selection['root'], 'LICENSES')
    print(f'# {selection["tag"]}\n')
    print(f'Tested commit: `{sha}`.\n')
    if previous:
        print(f'Changes since `{previous}` for the project and shared license texts.\n')
    else:
        print('No prior reachable project tag. Showing project and shared license history to the tested commit.\n')
    print(history or 'No scoped changes in this range.')


def check_main(sha, actual):
    if not re.fullmatch(r'[a-f0-9]{40}', sha) or sha != actual:
        raise ValueError('main changed since verification; dispatch again for the new head')


def main(argv):
    command, *args = argv
    if command == 'select':
        print(json.dumps(select(*args)))
    elif command == 'rules':
        project, doc = args
        validate_rules(project, json.loads(sys.stdin.read() if doc == '-' else doc))
    elif command == 'notes':
        notes(*args)
    elif command == 'tag-absent':
        check_tag(*args)
    elif command == 'main':
        check_main(*args)
    else:
        raise ValueError('Unknown project release operation')


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except (ValueError, TypeError, KeyError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
