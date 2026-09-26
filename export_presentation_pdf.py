"""Convert the actual PPTX to searchable PDF with a task-local Korean font config."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from xml.sax.saxutils import escape
from zipfile import ZipFile

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
RUNTIME = Path('/Users/hamseongho/.cache/codex-runtimes/codex-primary-runtime/dependencies')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pptx', type=Path)
    args = parser.parse_args()
    pptx = args.pptx.resolve()
    if not pptx.is_relative_to(ROOT / 'output/presentation') or pptx.suffix != '.pptx':
        parser.error('Use a validated project output/presentation/*.pptx')
    output = ROOT / 'output/pdf' / (pptx.stem + '.pdf')
    if output.exists():
        parser.error('PDF exists; use a new presentation revision')
    private = ROOT / 'tmp/presentations' / (pptx.stem + '-pdf')
    private.mkdir(parents=True, exist_ok=True)
    font_dir = Path('/Users/hamseongho/Library/Fonts')
    if not (font_dir / 'NanumGothic.ttf').is_file():
        parser.error('Install the NanumGothic font used by this deck before exporting')
    bundled_fonts = RUNTIME / 'native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype'
    config = private / 'fonts.conf'
    config.write_text('<?xml version="1.0"?>\n<fontconfig>\n'
                      f'  <dir>{escape(str(font_dir))}</dir>\n'
                      f'  <dir>{escape(str(bundled_fonts))}</dir>\n'
                      f'  <cachedir>{escape(str(private / "font-cache"))}</cachedir>\n'
                      '</fontconfig>\n')
    environment = os.environ.copy()
    environment['FONTCONFIG_FILE'] = str(config)
    command = [str(RUNTIME / 'bin/override/soffice'), '--headless', '--convert-to', 'pdf',
               '--outdir', str(private), str(pptx)]
    process = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
    receipt = {'command': command, 'exit_code': process.returncode,
               'stdout': process.stdout, 'stderr': process.stderr,
               'font_config': str(config.relative_to(ROOT)),
               'pptx_sha256': hashlib.sha256(pptx.read_bytes()).hexdigest()}
    (private / 'conversion.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    process.check_returncode()
    candidate = private / output.name
    reader = PdfReader(candidate)
    with ZipFile(pptx) as archive:
        slide_count = sum(name.startswith('ppt/slides/slide') and name.endswith('.xml') for name in archive.namelist())
    if len(reader.pages) != slide_count or '사출성형' not in reader.pages[0].extract_text():
        raise ValueError('PDF export failed slide count or Korean text preservation check')
    output.parent.mkdir(parents=True, exist_ok=True)
    # No overwrite: only publish after the actual conversion passed text checks.
    with output.open('xb') as handle:
        handle.write(candidate.read_bytes())
    print(json.dumps({'pdf': str(output), 'pages': slide_count,
                      'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                      'visual_review_required': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
