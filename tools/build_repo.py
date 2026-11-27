"""Build md_tree_free.zip and plugins.xml (QGIS custom plugin repository).

Run from the repository root:  python tools/build_repo.py [user] [repo] [branch]
"""
import configparser, datetime, os, sys, zipfile
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN = 'md_tree_free'
user = sys.argv[1] if len(sys.argv) > 1 else 'modidev-varkala'
repo = sys.argv[2] if len(sys.argv) > 2 else 'md_tree_free'
branch = sys.argv[3] if len(sys.argv) > 3 else 'main'

cp = configparser.ConfigParser(interpolation=None)
cp.read(os.path.join(ROOT, PLUGIN, 'metadata.txt'), encoding='utf-8')
g = cp['general']

zip_path = os.path.join(ROOT, PLUGIN + '.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for d, dirs, files in os.walk(os.path.join(ROOT, PLUGIN)):
        dirs[:] = [x for x in dirs if x != '__pycache__']
        for f in sorted(files):
            if f.endswith('.pyc'):
                continue
            full = os.path.join(d, f)
            z.write(full, os.path.relpath(full, ROOT))

raw = f'https://raw.githubusercontent.com/{user}/{repo}/{branch}'
today = datetime.date.today().isoformat()
e = lambda k, d='': escape(g.get(k, d).strip())
xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<plugins>
  <pyqgis_plugin name="{e('name')}" version="{e('version')}">
    <description>{e('description')}</description>
    <about>{e('about')}</about>
    <version>{e('version')}</version>
    <qgis_minimum_version>{e('qgisMinimumVersion', '3.22')}</qgis_minimum_version>
    <homepage>{e('homepage')}</homepage>
    <file_name>{PLUGIN}.zip</file_name>
    <icon>icon.svg</icon>
    <author_name>{e('author')}</author_name>
    <download_url>{raw}/{PLUGIN}.zip</download_url>
    <uploaded_by>{e('author')}</uploaded_by>
    <create_date>{today}</create_date>
    <update_date>{today}</update_date>
    <experimental>{e('experimental', 'False')}</experimental>
    <deprecated>{e('deprecated', 'False')}</deprecated>
    <tracker>{e('tracker')}</tracker>
    <repository>{e('repository')}</repository>
    <tags>{e('tags')}</tags>
    <category>{e('category', 'Vector')}</category>
  </pyqgis_plugin>
</plugins>
'''
with open(os.path.join(ROOT, 'plugins.xml'), 'w', encoding='utf-8') as f:
    f.write(xml)
print('built', zip_path, 'and plugins.xml for', raw)
