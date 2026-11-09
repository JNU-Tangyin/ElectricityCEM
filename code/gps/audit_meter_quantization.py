from pathlib import Path
from zipfile import ZipFile
from xml.etree.ElementTree import iterparse
import csv,json,hashlib
import argparse
parser=argparse.ArgumentParser(description='Audit 15-minute PV/load quantization for the panel in every worksheet.')
parser.add_argument('--source',type=Path,required=True)
args=parser.parse_args()
package=Path(__file__).resolve().parents[2];root=package.parent;source=args.source.resolve()
ids={r['id'] for r in csv.DictReader((package/'data/gps/coordinates_71.csv').open())};stats={};ns='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
with ZipFile(source) as z:
 for name in z.namelist():
  if not (name.startswith('xl/worksheets/sheet') and name.endswith('.xml')):continue
  with z.open(name) as stream:
   for _,row in iterparse(stream,events=['end']):
    if row.tag!=ns+'row':continue
    cells={c.attrib['r'].rstrip('0123456789'):c.findtext(ns+'v') for c in row.findall(ns+'c')}
    try:h=str(int(float(cells.get('A'))))
    except (ValueError,TypeError):row.clear();continue
    if h in ids:
     s=stats.setdefault(h,dict(household=h,n=0,zero_pv=0,zero_load=0,pv_off_grid=0,load_off_grid=0,pv_min_positive=float('inf'),load_min_positive=float('inf'),non_15min=0))
     pv=float(cells['B']);load=float(cells['C']);s['n']+=1
     for label,value in [('pv',pv),('load',load)]:
      s['zero_'+label]+=value==0;s[label+'_off_grid']+=abs(value*10-round(value*10))>1e-8
      if value>0:s[label+'_min_positive']=min(s[label+'_min_positive'],value)
     if cells.get('D') and cells.get('E'):s['non_15min']+=abs((float(cells['E'])-float(cells['D']))*1440-15)>1e-4
    row.clear()
rows=list(stats.values());out=package/'results/gps/comparison';p=out/'meter_quantization_file1_audit.csv'
with p.open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metadata={'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'panel_households_found':len(stats),'rows':sum(s['n'] for s in rows),'load_zero_fraction':sum(s['zero_load'] for s in rows)/sum(s['n'] for s in rows),'pv_zero_fraction':sum(s['zero_pv'] for s in rows)/sum(s['n'] for s in rows),'combined_zero_fraction':sum(s['zero_load']+s['zero_pv'] for s in rows)/(2*sum(s['n'] for s in rows)),'panel_households_not_found':71-len(stats)}
(out/'meter_quantization_file1_audit.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata))
