#!/usr/bin/env python3
"""One guarded saved-pencil m0 search. Default CLI performs metadata-only check.

Execution is separate root-owned compute admission. No native solver can be
imported. A computed candidate passing saved-pencil checks is not a rigorously
certified eigenvalue, completeness statement or physical-mode qualification.
"""
from __future__ import annotations
import argparse,hashlib,importlib.abc,io,json,os,re,stat,sys,time,traceback,types
from pathlib import Path
sys.dont_write_bytecode=True

class NativeDeny(importlib.abc.MetaPathFinder):
    def __init__(self):self.attempts=[]
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='pyoomph' or fullname.startswith(('pyoomph.','_pyoomph')):
            self.attempts.append(fullname);raise ImportError('native solver import denied: '+fullname)
        return None

DENY=NativeDeny();sys.meta_path.insert(0,DENY)

def no_native():
    bad=[n for n in sys.modules if n=='pyoomph' or n.startswith(('pyoomph.','_pyoomph'))]
    if bad or DENY.attempts:raise RuntimeError('native-import invariant failed: '+repr((bad,DENY.attempts)))

def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def unique_pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise RuntimeError('duplicate packet key '+key)
        result[key]=value
    return result
def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def regular(path):
    p=Path(os.path.abspath(path))
    for q in [p,*p.parents]:
        if q.is_symlink():raise RuntimeError('symlink in packet authority path '+str(q))
    if not stat.S_ISREG(p.stat().st_mode):raise RuntimeError('packet authority is not regular')
    return p

def boot_read(path):
    p=regular(path);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):raise RuntimeError('nonregular source authority')
        blocks=[]
        while True:
            b=os.read(fd,1024**2)
            if not b:break
            blocks.append(b)
        after=os.fstat(fd)
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):raise RuntimeError('source authority changed during read')
        return b''.join(blocks)
    finally:os.close(fd)

def load_closure(base,packet):
    expected={'saved_search.py','admission.py','numerics.py','geometry_volume.py'}
    if set(packet['source_closure'])!=expected:raise RuntimeError('source closure differs')
    contents={}
    for name,record in packet['source_closure'].items():
        if set(record)!= {'sha256','bytes'} or re.fullmatch('[0-9a-f]{64}',record['sha256']) is None:raise RuntimeError('closure record schema differs')
        path=regular(base/name);data=boot_read(path)
        if len(data)!=record['bytes'] or sha_bytes(data)!=record['sha256']:raise RuntimeError('changed runtime source '+name)
        contents[name]=data
    if boot_read(__file__)!=contents['saved_search.py']:raise RuntimeError('entrypoint is outside bound closure')
    modules={}
    for name in ('admission.py','numerics.py','geometry_volume.py'):
        module=types.ModuleType('_bound_'+name[:-3]);module.__file__=str(base/name)
        exec(compile(contents[name],str(base/name),'exec'),module.__dict__);modules[name]=module
    return modules

class BudgetStop(RuntimeError):pass

class Store:
    """Exclusive regular output files; fsync each receipt before advancing."""
    def __init__(self,path,cap=2*1024**3):
        self.path=Path(path);self.cap=cap;self.bytes=0;self.closed=False;self.files={};self.last_stage='new'
        parent=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.mkdir(self.path.name,mode=0o700,dir_fd=parent);os.fsync(parent)
        finally:os.close(parent)
        self.fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        self.progress=os.open('progress.jsonl',os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_WRONLY,0o600,dir_fd=self.fd)
    def write(self,name,payload):
        if self.closed or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',name) is None:raise RuntimeError('closed/invalid output file')
        terminal=name in ('failure.json','postcheck-failure.json','result.json','output-hashes.json','completion.json')
        limit=self.cap if terminal else self.cap-1024**2
        if self.bytes+len(payload)>limit:raise BudgetStop('output storage cap (terminal receipt reserve retained)')
        fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=self.fd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):raise RuntimeError('output not regular')
            view=memoryview(payload)
            while view:
                count=os.write(fd,view);view=view[count:]
            os.fsync(fd)
        finally:os.close(fd)
        os.fsync(self.fd);self.bytes+=len(payload);self.files[name]={'sha256':sha_bytes(payload),'bytes':len(payload)}
    def json(self,name,value):self.write(name,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
    def npz(self,name,np,**arrays):
        stream=io.BytesIO();np.savez_compressed(stream,**arrays);self.write(name,stream.getvalue())
    def event(self,stage,**values):
        if self.closed:raise RuntimeError('event after completion is prohibited')
        self.last_stage=stage;data=(json.dumps({'event':stage,'wall_clock_utc_ns':time.time_ns(),**values},allow_nan=False)+'\n').encode()
        limit=self.cap if stage=='final' else self.cap-1024**2
        if self.bytes+len(data)>limit:raise BudgetStop('progress exceeds storage cap')
        view=memoryview(data)
        while view:
            count=os.write(self.progress,view);view=view[count:]
        os.fsync(self.progress);self.bytes+=len(data)
    def finish(self,status,record):
        self.json('result.json',{'status':status,**record})
        self.event('final',status=status,result_sha256=self.files['result.json']['sha256'])
        os.close(self.progress);self.progress=None
        self.files['progress.jsonl']={'sha256':sha_file(self.path/'progress.jsonl'),'bytes':(self.path/'progress.jsonl').stat().st_size}
        # Preserve a partial file if a write itself failed; never overwrite it.
        for path in self.path.iterdir():
            if path.is_symlink() or not path.is_file():raise RuntimeError('unexpected nonregular output')
            if path.name not in self.files:self.files[path.name]={'sha256':sha_file(path),'bytes':path.stat().st_size,'partial_unsealed_file':True}
        self.json('output-hashes.json',dict(self.files))
        self.json('completion.json',{'schema':'stone-paper-saved-spectrum-output-v1','status':status,'files':dict(self.files),
            'scientific_acceptance':False,'finite_spectrum_complete':False,'global_dominance_certified':False,'native_imports':False})
        self.closed=True;os.close(self.fd)

def elapsed_check(start,cap=10800):
    if time.monotonic()-start>=cap:raise BudgetStop('runtime cap elapsed')

def saved_geometry_diagnostics(math,prepared,helper,vector):
    """Keep unavailable matrix-only geometry separate from pencil diagnostics."""
    if prepared['geometry'] is None:
        return {'status':'explicitly_unavailable_matrix_only','native_physical_mode_qualification':False}
    return math.geometry_diagnostics(vector,prepared['geometry'],helper,prepared['j'],
                                     prepared['layout']['globals']['gas_volume/pressure'][0])

def run(prepared,packet,store,modules,stop_after_block,start):
    np=modules['numerics.py'].np;math=modules['numerics.py'];admit=modules['admission.py'];helper=modules['geometry_volume.py'].__dict__
    j,m,R=prepared['j'],prepared['m'],prepared['rows'];layout=prepared['layout'];limits=packet['gates'];algorithm=packet['algorithm'];n=j.shape[0];q=len(R)
    caps=admit.external_execution_caps();store.json('execution-caps.json',caps)
    k=(j-algorithm['shift']*m).tocsr();admit.require(k.dtype==np.float64,'shift promotion is forbidden')
    k_signature={name:{'shape':list(getattr(k,name).shape),'dtype':str(getattr(k,name).dtype),'sha256':sha_bytes(getattr(k,name).tobytes())} for name in ('indptr','indices','data')}
    store.json('stage-0.json',{'input_witness':prepared['witness'],'metadata':prepared['metadata'],'K_nnz':k.nnz,'K_fields':k_signature,
        'algorithm':algorithm,'external_caps':caps,'source_closure':packet['source_closure'],'native_imports':False})
    store.event('factor_started');elapsed_check(start)
    tic=time.monotonic();inverse=math.RealInverse(k);factor_seconds=time.monotonic()-tic
    memory=admit.resource_measurement();P1=math.p1(memory['VmHWM_bytes'])
    store.json('stage-1.json',{'factor_wall_seconds':factor_seconds,'lu_nnz':inverse.nnz,
        'factor_storage_12nnz_plus_8n_proxy_bytes':12*inverse.nnz+8*n,'factor_proxy_is_not_measured_workspace':True,
        'measurements':memory,'P1':P1,'L_or_U_materialised':False,'factorisations':1})
    store.event('factor_finished',P1=P1)
    if not P1:raise BudgetStop('P1 process HWM plus1.5GiB exceeds60GiB')
    b=m[R,:];g=np.full((q,q),np.nan,dtype=np.float64);block_times=[];block_records=[];inverse_proxy=None
    b1=float(np.max(np.asarray(abs(b).sum(axis=0)).ravel()));max_resid_1=0.
    for block,first in enumerate(range(0,q,16),start=1):
        elapsed_check(start);last=min(first+16,q);tic=time.monotonic()
        values,record=math.reduced_block(inverse,k,b,R,first,last);seconds=time.monotonic()-tic
        backward=record['solve_refinements'][-1]['backward_residual_fro']
        if not np.isfinite(backward) or backward>limits['backward']:
            store.json(f'block-{block:04d}-failure.json',record);raise math.NumericalFailure('original block backward gate')
        g[:,first:last]=values;record.update(block=block,wall_seconds=seconds,measurements=admit.resource_measurement())
        block_times.append(seconds);block_records.append(record);max_resid_1=max(max_resid_1,record['solve_refinements'][-1]['max_column_residual_1norm'])
        store.json(f'block-{block:04d}.json',record)
        if block==1:
            P2=math.p2(time.monotonic()-start,seconds,algorithm['blocks']-1)
            store.npz('G-first-block.npz',np,G=g[:,:last],mass_rows=R,completed_columns=np.array(last,dtype=np.int64))
            store.json('stage-2.json',{'first_block_seconds':seconds,'elapsed_seconds':time.monotonic()-start,
                'remaining_blocks':algorithm['blocks']-1,'P2':P2,'forecast_formula':f"elapsed+{algorithm['blocks']-1}*first_block_seconds*1.5+900<=10800",'record':record})
            store.event('first_block_finished',P2=P2)
            if not P2:raise BudgetStop('P2 measured forecast exceeds10800s')
        if block%10==0 or last==q:
            store.npz(f'G-partial-{block:04d}.npz',np,G=g[:,:last],mass_rows=R,completed_columns=np.array(last,dtype=np.int64))
        if stop_after_block is not None and block==stop_after_block:
            if block%10 and block!=1 and last!=q:store.npz(f'G-stop-{block:04d}.npz',np,G=g[:,:last],mass_rows=R,completed_columns=np.array(last,dtype=np.int64))
            raise BudgetStop('requested stop-after-block '+str(block))
        if inverse_proxy is None:
            elapsed_check(start);inverse_proxy=math.inverse_1norm_proxy(inverse,k);store.json('inverse-1norm-proxy.json',inverse_proxy)
        if not math.p1(admit.self_memory()['VmHWM_bytes']):raise BudgetStop('process HWM resource predicate between blocks')
    no_native();elapsed_check(start)
    store.npz('G-full.npz',np,G=g,mass_rows=R)
    solve_scale_1=b1*inverse_proxy['K_inverse_1norm_estimate']*max_resid_1
    g1=float(np.linalg.norm(g,1));fp_scale_1=float(np.finfo(float).eps*q*g1)
    proxy=solve_scale_1+fp_scale_1
    store.json('stage-3.json',{'blocks':len(block_records),'block_times_seconds':block_times,'B_1norm':b1,
        'max_full_residual_column_1norm':max_resid_1,'K_inverse_1norm_estimate':inverse_proxy,
        'solve_contribution_1norm_proxy':solve_scale_1,'floating_scale_1norm_proxy':fp_scale_1,'dG_1norm_proxy':proxy,
        'proxy_is_not_certified_upper_bound':True,'norm_consistency':'All solve-contribution factors use induced matrix1-norm/maximum residual-column1-norm.',
        'limitations':'Inverse estimate and floating residual/product/eig rounding are not verified error bounds; refinement changes are sensitivity observations.'})
    store.event('small_spectrum_started');tic=time.monotonic()
    mu,left,right=math.linalg.eig(g,left=True,right=True,check_finite=True)
    eig_seconds=time.monotonic()-tic;records,classification=math.classify_roots(g,mu,left,right,algorithm['shift'],proxy)
    store.npz('small-spectrum.npz',np,mu=mu,left=left,right=right,mass_rows=R)
    returned=prepared['eigen']['eigen_plus']['values']
    sanity=math.sanity_match(records,returned,limits['two_shift_complex'])
    selection=math.select_lifts(records,sanity,returned,packet['reference_rectangles'],prepared['metadata']['time_factor'],packet['model']['physics']['ca_in'])
    store.json('stage-4.json',{'dense_eig_wall_seconds':eig_seconds,'classification':classification,'sanity':sanity,'selection':selection,'measurements':admit.resource_measurement()})
    if not sanity['passed']:
        store.json('roots.json',records);return {'status':'numerical_sanity_failed_inconclusive','sanity':sanity,'selection':selection,'lifts':[],
            'interpretation':'Numerical sanity failure; no proven code/input defect and no spectrum absence statement.'}
    lifts=[];lookup={r['index']:r for r in records}
    for ordinal,index in enumerate(selection['indices']):
        elapsed_check(start);row=lookup[index];value=math.mapped(row)
        rhs=math.injected(n,R,right[:,index]);v,refinement=math.refined_complex(inverse,k,rhs)
        vn=float(np.linalg.norm(v));math.finite(v,'full lift');math.finite(vn,'full lift norm');admit.require(vn>0,'zero lift');v/=vn
        diagnostic=math.diagnose(value,v,j,m,layout,prepared['metadata']['time_factor'],limits)
        geometric=saved_geometry_diagnostics(math,prepared,helper,v)
        row['lifted']=True;record={'ordinal':ordinal,'reduced_index':index,'reciprocal_classification':row['classification'],
            'selection_tags':{'control':index in selection['control_indices'],'exact_rectangle_priority':index in selection['rectangle_priority_indices'],
                'additional_challenger':index in selection['additional_challenger_indices']},'diagnostic':diagnostic,'refinement':refinement,
            'saved_data_geometry':geometric,'native_physical_mode_qualification':False}
        store.npz(f'lift-{ordinal:02d}.npz',np,vector=v,lambda_value=np.array(value),mu_value=np.array(mu[index]),reduced_index=np.array(index,dtype=np.int64))
        store.json(f'lift-{ordinal:02d}.json',record);lifts.append(record)
        store.event('lift_finished',ordinal=ordinal,reduced_index=index,passes_saved_pencil=diagnostic['computed_candidate_passes_original_pencil_gates'])
    store.json('roots.json',records)
    accepted=[r for r in lifts if not r['selection_tags']['control'] and lookup[r['reduced_index']]['additional_challenger_numerical_mapping'] and r['diagnostic']['computed_candidate_passes_original_pencil_gates']]
    status='computed_additional_candidate_passes_saved_pencil_gates' if accepted else ('lift_cap_truncated_inconclusive' if selection['cap_truncated_any_candidate_set'] else 'no_additional_challenger_found_among_numerically_classified_checked_roots')
    return {'status':status,'sanity':sanity,'selection':selection,'lifts':lifts,'dense_eig_wall_seconds':eig_seconds,
        'cap_truncated_any_candidate_set':selection['cap_truncated_any_candidate_set'],
        'inverse_rhs_counts':{'forward':inverse.forward_rhs,'transpose':inverse.transpose_rhs},
        'accepted_candidate_indices':[r['reduced_index'] for r in accepted],
        'absence_claim_allowed':False,'exact_root_exclusion_certified':False,'finite_spectrum_complete':False,
        'global_dominance_certified':False,'physical_mode_qualification':False,'second_shift_run':False,
        'interpretation':'Only computed numerical candidates/checks at one saved pencil; unresolved/poor roots and all unlifted candidates remain retained. No exact-root, completeness, dominance or model conclusion.'}

def main(argv=None):
    start=time.monotonic();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--packet-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--check',action='store_true');parser.add_argument('--shift',type=float,default=.001)
    parser.add_argument('--stop-after-block',type=int)
    args=parser.parse_args(argv)
    if args.execute and args.check:parser.error('choose metadata-only check or execution')
    if args.shift!=.001:parser.error('only the one bound real+.001 shift is admitted')
    if args.stop_after_block is not None and not 1<=args.stop_after_block<=93:parser.error('stop-after-block must be1..93')
    packet_path=regular(args.packet);raw=boot_read(packet_path)
    if re.fullmatch('[0-9a-f]{64}',args.packet_sha256) is None or sha_bytes(raw)!=args.packet_sha256:raise RuntimeError('packet authority hash differs')
    packet=json.loads(raw,object_pairs_hook=unique_pairs,parse_constant=lambda x:(_ for _ in ()).throw(RuntimeError('nonfinite packet '+x)))
    base=packet_path.parent;modules=load_closure(base,packet);admit=modules['admission.py'];math=modules['numerics.py']
    packet=admit.json_bytes(raw);prepared=admit.preflight(packet,base,args.output,math.__dict__);no_native()
    if not args.execute:
        print(json.dumps({'status':'metadata_checked_no_execution','packet_sha256':args.packet_sha256,**prepared['metadata'],
            'input_witness':prepared['witness'],'source_closure':packet['source_closure'],'output_absent':True,
            'external_caps_not_exercised':True,'runtime_seconds':time.monotonic()-start}));return 0
    store=Store(prepared['out'],packet['resource_caps']['storage_bytes']);record=None;exit_code=0
    try:
        store.event('prepared',packet_sha256=args.packet_sha256)
        record=run(prepared,packet,store,modules,args.stop_after_block,start)
    except BudgetStop as exc:
        record={'status':'budget_or_requested_stop','reason':str(exc),'partial_stage':store.last_stage,'scientific_statement':None}
    except BaseException as exc:
        exit_code=2;error={'error_class':type(exc).__name__,'error_message':str(exc),'original_traceback':traceback.format_exc(),'partial_stage':store.last_stage}
        store.json('failure.json',error);record={'status':'failed_preserved_inconclusive',**error,'scientific_statement':None}
    try:
        no_native();_,post=admit.verify_inputs(base,packet['inputs']);admit.require(post==prepared['witness'],'post input witness differs')
        for name,item in packet['source_closure'].items():admit.read_regular(base/name,item['sha256'],item['bytes'])
        record.update(post_input_hashes_verified=True,no_native_import=True,total_wall_seconds=time.monotonic()-start,
            resource_measurements=admit.resource_measurement(),second_shift_automatic=False)
    except BaseException as exc:
        exit_code=2;error={'error_class':type(exc).__name__,'error_message':str(exc),'original_traceback':traceback.format_exc()}
        store.json('postcheck-failure.json',error);record['postcheck_failure']=error;record['status']='postcheck_failed_inconclusive'
    store.finish(record.pop('status'),record)
    print(json.dumps({'completion':str(store.path/'completion.json'),'completion_sha256':sha_file(store.path/'completion.json'),'exit_code':exit_code}));return exit_code

if __name__=='__main__':raise SystemExit(main())
