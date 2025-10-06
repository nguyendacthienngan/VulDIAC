import os, glob, argparse, subprocess
from multiprocessing import Pool
from functools import partial

JOERN_HOME = os.environ.get("JOERN_HOME", "/kaggle/working/joern-2.0.72")
JOERN_PARSE = os.path.join(JOERN_HOME, "joern-parse")
JOERN_EXPORT = os.path.join(JOERN_HOME, "joern-export")
JOERN = os.path.join(JOERN_HOME, "joern")

def parse_options():
    p = argparse.ArgumentParser(description='Extracting CPGs.')
    p.add_argument('-i', '--input', help='The dir path of input', type=str, default='/kaggle/working/c_src/')
    p.add_argument('-o', '--output', help='The dir path of output', type=str, default='/kaggle/working/exports/')
    p.add_argument('-t', '--type', help='The type of procedures: parse or export', type=str, default='export', choices=['parse','export'])
    p.add_argument('-r', '--repr', help='The type of representation: pdg or lineinfo_json', type=str, default='lineinfo_json',
                   choices=['pdg','cpg','cpg14','cfg','lineinfo_json'])
    p.add_argument('--language', type=str, default='c')   # cho joern-parse
    p.add_argument('--pool', type=int, default=4)         # Kaggle nên <=4
    return p.parse_args()

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def joern_parse(src_file, outdir, language='c'):
    ensure_dir(outdir)
    rec = os.path.join(outdir, "parse_res.txt")
    open(rec, 'a').close()
    name = os.path.splitext(os.path.basename(src_file))[0]
    with open(rec,'r') as f: done = set(x.strip() for x in f)
    if name in done: return
    outbin = os.path.join(outdir, name + '.bin')
    if os.path.exists(outbin): return
    print(f"[parse] {name}")
    subprocess.run([JOERN_PARSE, src_file, '--language', language, '-o', outbin],
                   check=True)
    with open(rec,'a') as f: f.write(name+'\n')

def _move_first_child_with_prefix(src_dir, prefix, out_dot):
    try:
        for entry in os.listdir(src_dir):
            if entry.startswith(prefix):
                src = os.path.join(src_dir, entry)
                subprocess.run(['mv', src, out_dot], check=True)
                subprocess.run(['rm','-rf', src_dir], check=True)
                return
    except Exception:
        pass

def joern_export(bin_file, outdir, repr_):
    ensure_dir(outdir)
    rec = os.path.join(outdir, "export_res.txt")
    open(rec,'a').close()
    name = os.path.basename(bin_file).replace('.bin','')
    with open(rec,'r') as f: done = set(x.strip() for x in f)
    if name in done: return
    print(f"[export:{repr_}] {name}")
    outbase = os.path.join(outdir, name)

    if repr_ in ['pdg','cpg','cpg14','cfg']:
        # joern-export path
        args = [JOERN_EXPORT, bin_file, '--repr', repr_, '-o', outbase]
        # cfg đôi khi cần --out thay vì -o tùy version, nhưng 2.0.72 chấp nhận -o
        subprocess.run(args, check=True)

        # Gom dot ra một file .dot như script gốc làm
        if repr_ == 'pdg':
            _move_first_child_with_prefix(outbase, '0-pdg', outbase+'.dot')
        elif repr_ == 'cpg':
            # tìm *_global_.dot như script gốc
            cfile_dir = os.path.join(outbase, name + '.c')
            candidate = os.path.join(cfile_dir, '_global_.dot')
            if os.path.exists(candidate):
                subprocess.run(['mv', candidate, outbase+'.dot'], check=True)
                subprocess.run(['rm','-rf', outbase], check=True)
        elif repr_ == 'cpg14':
            _move_first_child_with_prefix(outbase, '1-cpg', outbase+'.dot')
        elif repr_ == 'cfg':
            _move_first_child_with_prefix(outbase, '0-cfg', outbase+'.dot')

    else:
        # lineinfo_json: chạy joern non-interactive và gọi script Scala
        outjson = outbase + '.json'
        script_path = os.path.join(JOERN_HOME, 'scripts', 'graph', 'graph-for-funcs.sc')
        # Một số version đổi API; nếu thiếu script này bạn có thể dùng cơ chế --script (interpreter)
        cmd = f'importCpg("{bin_file}")\n' \
              f'cpg.runScript("{script_path}").toString() |> "{outjson}"\n'
        # chạy joern và feed lệnh
        proc = subprocess.Popen([JOERN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"joern script failed: {stderr}\n{stdout}")

    with open(rec,'a') as f: f.write(name+'\n')

def main():
    args = parse_options()
    inp = args.input if args.input.endswith('/') else args.input + '/'
    out = args.output if args.output.endswith('/') else args.output + '/'
    ensure_dir(out)

    if args.type == 'parse':
        files = glob.glob(inp + '*.c')
        with Pool(args.pool) as pool:
            pool.map(partial(joern_parse, outdir=os.path.dirname(out), language=args.language), files)
    elif args.type == 'export':
        bins = glob.glob(inp + '*.bin')
        with Pool(args.pool) as pool:
            pool.map(partial(joern_export, outdir=out, repr_=args.repr), bins)
    else:
        raise ValueError("type must be parse or export")

if __name__ == "__main__":
    main()
