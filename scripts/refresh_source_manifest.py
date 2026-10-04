from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]/'source_data'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for p in sorted(root.iterdir(),key=lambda p:p.name.lower()) if p.is_file() and p.name not in {'manifest.json','MANIFEST_SHA256.txt'}]
manifest=[{'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in files]
(root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
check_files=[p for p in sorted(root.iterdir(),key=lambda p:p.name.lower()) if p.is_file() and p.name!='MANIFEST_SHA256.txt']
(root/'MANIFEST_SHA256.txt').write_text('\n'.join(f'{sha(p)}  {p.name}  {p.stat().st_size}' for p in check_files)+'\n',encoding='utf-8')
print(len(manifest),len(check_files))
