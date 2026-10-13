"""Strict regular-file, metadata, CSR and external-cap admission.

No solver package is imported. Original input files remain read-only. A check
may read arrays and distribution metadata but never factors or eigensolves.
"""
from __future__ import annotations
import hashlib,importlib.metadata,io,json,os,re,stat,sys
from pathlib import Path
import numpy as np
from scipy import sparse

SCHEMA='stone-paper-saved-spectrum-packet-v1'
MATRIX_ONLY_SCHEMA='stone-paper-saved-spectrum-coarse-matrix-only-v1'
SOURCE='bf704cc07c964daa800abcc09b1d6de611ae80fb'
COARSE_SOURCE='83e94bd4d9294fa8bf1dd9339958b0858aee27c4'
NUMERICS_ORIGIN={'sha256':'c9cbfd252c26af9017e248ed34965a8e9200c5cc4c5577dbb9258294cdb977cf','bytes':13167}
PYOOMPH='a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618'
CORE='af7334e81f38453d6852581b6c208e93a0648c6887a6766b0fe63acd2ab51c0d'
STRATEGY='98a00ce698eee6f3ef4a2069f673377df09843e41feb160b37e90bc30b6d8601'
FACTOR=.6716793862139855
LIMITS={'backward':1e-7,'mass_action':1e-8,'forced_coefficient':1e-12,'two_shift_complex':1e-7,'normal_overlap':.99,'epsilon':1e-4,'halving':1e-4,'normal_rank_relative':1e-10}
CAPS={'memory_bytes':64*1024**3,'swap_bytes':0,'runtime_seconds':10800,'storage_bytes':2*1024**3,'whole_cores':1,'logical_cpus':2}
ALGORITHM={'shift':.001,'rhs_columns':16,'blocks':93,'refinements':2,'max_lifts':24,
    'P1_hwm_limit_bytes':60*1024**3,'P1_margin_bytes':int(1.5*1024**3),'P2_safety_multiplier':1.5,'P2_tail_seconds':900,
    'poor_multiplier':1000,'onenormest_t':2,'onenormest_itmax':5,'second_shift_automatic':False}
GEOMETRY_ROLES={'gas_geometry','liquid_geometry','volume_w4','volume_identity','volume_provenance','volume_restoration'}
BASE_ROLES={'j','m','layout','eigen_plus','eigen_minus','case','native_manifest','native_completion','preparation','reference','strategy','core','source_modes','source_model','source_bubble','source_spectra'}|{f'pair_plus_{i:02d}' for i in range(8)}

class AdmissionFailure(RuntimeError):pass

def require(condition,message):
    if not condition:raise AdmissionFailure(message)

def packet_profile(packet):
    """Select one of two closed producer types; other anchors are refused."""
    schema=packet.get('schema')
    require(schema in (SCHEMA,MATRIX_ONLY_SCHEMA),'packet schema differs')
    matrix_only=schema==MATRIX_ONLY_SCHEMA
    if matrix_only:
        require(packet['source_closure'].get('numerics.py')==NUMERICS_ORIGIN,
                'matrix-only numerical kernel origin differs')
    return {'matrix_only':matrix_only,'source':COARSE_SOURCE if matrix_only else SOURCE,
            'ca':-.0038 if matrix_only else -.00315,
            'j_nnz':10490006 if matrix_only else 10488602,
            'roles':BASE_ROLES if matrix_only else BASE_ROLES|GEOMETRY_ROLES}

def matrix_only_reference(reference,rectangles):
    require(reference=={'schema':'saved-paper-matrix-only-reference-v1','ca_in':-.0038,
                       'reference_rectangles':[],'purpose':'pencil-level-challenger-search'}
            and rectangles==[],'matrix-only reference policy differs')

def packet_policy(packet):
    """Validate the closed case/policy metadata before opening original inputs."""
    keys(packet,('schema','authority','environment','inputs','source_closure','model','target','gates','resource_caps','algorithm','reference_rectangles','protected_roots'),'packet')
    profile=packet_profile(packet)
    strategy=packet['authority'].get('strategy_sha256') if profile['matrix_only'] else STRATEGY
    require(type(strategy) is str and re.fullmatch('[0-9a-f]{64}',strategy),'strategy authority differs')
    require(packet['authority']=={'source_commit':profile['source'],'pyoomph_commit':PYOOMPH,'native_core_sha256':CORE,'strategy_sha256':strategy,'pencil_sign':'J=-dR/dU; Jv=lambda Mv'},'source/sign authority differs')
    require(same_types(packet['gates'],LIMITS) and same_types(packet['resource_caps'],CAPS) and same_types(packet['algorithm'],ALGORITHM),'gate/cap/algorithm policy differs')
    require(packet['model']['physics']=={'problem':'finite_gas_bubble','reynolds':0.,'bond':1.56,'ca_in':profile['ca'],'r0':1.,'r_tube':FACTOR,'vhat':3.3,'domain_over_r':15.,'upper_tip_over_r':10.,'gas_density_ratio':.000949,'gas_viscosity_ratio':1.8e-5,'seed_radius_over_tube':.8},'physical model packet differs')
    require(set(packet['inputs'])==profile['roles'],'bounded input roles differ')
    if profile['matrix_only']:require(packet['reference_rectangles']==[],'matrix-only reference policy differs')
    t=packet['target'];keys(t,('n','j_nnz','m_nnz','mass_rows','mass_columns','mass_rows_sha256','globals','forced','m','time_factor'),'target')
    require(all(type(t[k]) is int for k in ('n','j_nnz','m_nnz','mass_rows','mass_columns','m')),'target integer field type differs')
    require(t['n']==316989 and t['j_nnz']==profile['j_nnz'] and t['m_nnz']==11836 and t['mass_rows']==1481 and t['mass_columns']==2960 and t['m']==0 and type(t['time_factor']) is float and t['time_factor']==FACTOR,'target counts/control differ')
    return profile,strategy

def keys(value,expected,label):
    require(type(value) is dict and set(value)==set(expected),label+' schema keys differ')

def same_types(actual,expected):
    if type(actual) is not type(expected):return False
    if isinstance(expected,dict):return set(actual)==set(expected) and all(same_types(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):return len(actual)==len(expected) and all(same_types(a,b) for a,b in zip(actual,expected))
    return actual==expected

def no_duplicates(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise AdmissionFailure('duplicate JSON key '+k)
        result[k]=v
    return result

def json_bytes(data):
    try:return json.loads(data,object_pairs_hook=no_duplicates,parse_constant=lambda x:(_ for _ in ()).throw(AdmissionFailure('nonfinite JSON '+x)))
    except (UnicodeError,json.JSONDecodeError) as exc:raise AdmissionFailure('invalid JSON') from exc

def nonsymlink_components(path,allow_missing_final=False):
    p=Path(os.path.abspath(path))
    for q in reversed([p,*p.parents]):
        if not q.exists() and not q.is_symlink():
            require(allow_missing_final and q==p,'missing path component '+str(q));continue
        require(not q.is_symlink(),'symlink path component '+str(q))
    return p

def read_regular(path,expected=None,expected_bytes=None):
    p=nonsymlink_components(path)
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd);require(stat.S_ISREG(before.st_mode),'nonregular file '+str(p))
        if expected_bytes is not None:require(before.st_size==expected_bytes,'file size differs '+str(p))
        blocks=[];h=hashlib.sha256()
        while True:
            b=os.read(fd,4*1024*1024)
            if not b:break
            blocks.append(b);h.update(b)
        after=os.fstat(fd);require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'input changed during read '+str(p))
        value=h.hexdigest()
        if expected is not None:require(value==expected,'hash differs '+str(p))
        return b''.join(blocks),value
    finally:os.close(fd)

def record_path(base,record):
    keys(record,('path','sha256','bytes','kind'),'input record')
    require(type(record['path']) is str and type(record['sha256']) is str and re.fullmatch('[0-9a-f]{64}',record['sha256']),'invalid path/hash record')
    require(type(record['bytes']) is int and record['bytes']>=0,'invalid byte count')
    require(record['kind'] in ('npz','json','source','binary','text'),'unknown input kind')
    p=Path(record['path']);return p if p.is_absolute() else base/p

def verify_inputs(base,records):
    paths={};witness={}
    for name,record in records.items():
        require(re.fullmatch('[a-z][a-z0-9_]*',name) is not None,'invalid input role')
        p=record_path(base,record);_,h=read_regular(p,record['sha256'],record['bytes'])
        paths[name]=Path(os.path.abspath(p));witness[name]={'path':str(paths[name]),'sha256':h,'bytes':record['bytes']}
    return paths,witness

def npz(path,expected=None):
    # Parse exactly the regular-file bytes whose hash was admitted, without a
    # second pathname open that could follow a replaced symlink.
    raw,_=read_regular(path,expected)
    with np.load(io.BytesIO(raw),allow_pickle=False) as s:return {k:s[k].copy() for k in s.files}

def csr(path,n,nnz,expected=None):
    a=npz(path,expected);keys(a,('data','indices','indptr','format','shape'),'CSR archive')
    require(a['data'].dtype.str=='<f8' and a['data'].shape==(nnz,) and np.isfinite(a['data']).all(),'CSR data dtype/shape/finite')
    require(a['indices'].dtype.str=='<i4' and a['indices'].shape==(nnz,),'CSR indices dtype/shape')
    require(a['indptr'].dtype.str=='<i4' and a['indptr'].shape==(n+1,),'CSR indptr dtype/shape')
    require(a['shape'].dtype.str=='<i8' and a['shape'].tolist()==[n,n],'CSR square shape')
    require(a['format'].shape==() and a['format'].dtype.str=='|S3' and a['format'].item()==b'csr','CSR format')
    ptr=a['indptr'];ix=a['indices'];require(ptr[0]==0 and ptr[-1]==nnz and np.all(np.diff(ptr)>=0),'CSR pointer ranges')
    require(np.all((ix>=0)&(ix<n)),'CSR column range')
    # Original M columns are unsorted. Preserve those bytes/order; only reject
    # duplicates. The producer's ordinary J-sigma*M arithmetic defines K.
    for r in np.flatnonzero(np.diff(ptr)>0):
        columns=ix[ptr[r]:ptr[r+1]]
        require(len(np.unique(columns))==len(columns),'CSR duplicate columns')
    return sparse.csr_matrix((a['data'],ix,ptr),shape=(n,n),copy=False)

def jread(path):return json_bytes(read_regular(path)[0])

def validate_output(out,base,protected,paths):
    out=nonsymlink_components(out,allow_missing_final=True)
    require(not out.exists(),'output already exists')
    require(out.parent.is_dir(),'output parent must already exist')
    for root in [base,*map(Path,protected)]:
        root=Path(os.path.abspath(root));require(not out.is_relative_to(root) and not root.is_relative_to(out),'output overlaps protected original '+str(root))
    for p in paths.values():require(out!=p and not p.is_relative_to(out),'output contains an original input')
    return out

def environment(expected,paths):
    keys(expected,('interpreter','resolved_interpreter','python_version','numpy_version','scipy_version','pyoomph_version','pyoomph_commit'),'environment')
    require(sys.flags.isolated==1,'canonical interpreter must be invoked with -I')
    core=paths['core'];venv=core.parents[4]
    require(venv.name=='.venv' and core.parent.name=='pyoomph'
            and core.parents[1].name=='site-packages','native core custody differs')
    canonical=venv/'bin/python'
    require(sys.executable==expected['interpreter']==str(canonical),'noncanonical interpreter invocation')
    require(str(Path(sys.executable).resolve())==expected['resolved_interpreter'],'resolved interpreter differs')
    require('.'.join(map(str,sys.version_info[:3]))==expected['python_version']=='3.13.14','Python version differs')
    require(np.__version__==expected['numpy_version']=='2.5.3','NumPy version differs')
    require(importlib.metadata.version('scipy')==expected['scipy_version']=='1.18.1','SciPy version differs')
    require(importlib.metadata.version('pyoomph')==expected['pyoomph_version']=='0.2.1','pyoomph metadata version differs')
    dist=importlib.metadata.distribution('pyoomph');direct=json.loads(dist.read_text('direct_url.json'))
    require(direct['vcs_info']['commit_id']==expected['pyoomph_commit']==PYOOMPH,'pyoomph installed metadata commit differs')

def preflight(packet,base,out,kernels):
    profile,strategy=packet_policy(packet)
    paths,witness=verify_inputs(base,packet['inputs']);environment(packet['environment'],paths)
    require(witness['strategy']['sha256']==strategy and witness['core']['sha256']==CORE,'strategy/core witness differs')
    require(same_types(jread(paths['case']),packet['model']),'case bytes/model differ')
    manifest=jread(paths['native_manifest']);require(manifest['component_commit']==profile['source'] and manifest['case_sha256']==witness['case']['sha256'],'producer case/source differs')
    require(witness['source_modes']['sha256']==manifest['implementation']['runners/run_paper_modes.py']['sha256'],'modes producer source differs')
    for role,relative in [('source_model','problems/paper_bubble.py'),('source_bubble','problems/bretherton_bubble.py')]:
        require(witness[role]['sha256']==manifest['component']['sources'][relative]['sha256'],'component source witness differs')
    request=manifest['mode_request'];require(request['paper_time_factor']==FACTOR and request['shifts']==[-.001,.001] and request['limits']==LIMITS,'actual producer sign/time/gates differ')
    if profile['matrix_only']:
        require(request['schema']=='paper-returned-III-coarse-m0-request-v1'
                and request['modes']==[0] and request['ca_in']==-.0038
                and request['neigen']==8 and request['eigenproblem_calls']==2,
                'typed coarse m0 producer request differs')
    require(manifest['api_source']['pencil_sign']=='J=-dR/dU; Jv=lambda Mv','actual producer pencil sign differs')
    require(manifest['pyoomph']['commit']==PYOOMPH and manifest['pyoomph']['native_core']['sha256']==CORE,'producer environment differs')
    complete=jread(paths['native_completion']);require(complete['status']=='complete','parent not complete')
    for role,relative in [('j','m0-shift-1/J.npz'),('m','m0-shift-1/M.npz'),('layout','m0-shift-1/layout.json'),('eigen_plus','m0-shift-1/eigenpairs.npz'),('eigen_minus','m0-shift-0/eigenpairs.npz')]:
        require(complete['files'][relative]==witness[role]['sha256'],'original completion does not bind '+role)
    t=packet['target']
    j=csr(paths['j'],t['n'],t['j_nnz'],witness['j']['sha256']);m=csr(paths['m'],t['n'],t['m_nnz'],witness['m']['sha256']);require(np.all(m.data!=0),'explicit mass zero differs')
    rows=kernels['mass_rows'](m);require(len(rows)==1481 and hashlib.sha256(rows.tobytes()).hexdigest()==t['mass_rows_sha256'],'exact mass row order differs')
    require(len(np.unique(m.indices))==2960,'mass supported columns differ')
    layout=jread(paths['layout']);keys(layout,('types','names','incidence','forced','selectors','globals','field_counts'),'layout')
    require(len(layout['names'])==117 and all(type(v) is str and v for v in layout['names']),'layout name inventory invalid')
    require(all(type(v) is int for v in layout['types']),'layout type IDs are not exact integers')
    ty=np.asarray(layout['types']);require(ty.dtype.kind in 'iu' and ty.shape==(t['n'],) and np.all((ty>=0)&(ty<len(layout['names']))),'typed layout invalid')
    require(layout['globals']==t['globals']=={'globals/U':[316987],'gas_volume/pressure':[316988]} and layout['forced']==t['forced']==[],'globals/forced incidence differs')
    require(all(type(i) is int and 0<=i<t['n'] for ids in layout['incidence'].values() for i in ids),'incidence index differs')
    for name,ids in layout['globals'].items():require(layout['incidence'][name]==ids,'global exact address mapping differs')
    labels=[layout['names'][int(ty[r])] for r in rows]
    require(labels.count('liquid/interface/_kin_bc')==1479 and labels.count('liquid/interface/axis_front/_kin_bc')==1 and labels.count('liquid/interface/axis_rear/_kin_bc')==1,'mass row types differ')
    for ids in layout['globals'].values():
        for i in ids:require(m.indptr[i]==m.indptr[i+1] and not np.any(m.indices==i),'global mass row/column not zero')
    eig={}
    for role in ('eigen_plus','eigen_minus'):
        a=npz(paths[role],witness[role]['sha256']);keys(a,('values','vectors','modes','original_values','original_vectors','types','names','forced'),'eigenbank')
        require(a['values'].dtype.str=='<c16' and a['values'].shape==(8,) and a['vectors'].dtype.str=='<c16' and a['vectors'].shape==(8,t['n']),'eigenbank dtype/shape differs')
        require(a['original_values'].dtype==a['values'].dtype and a['original_values'].shape==a['values'].shape and a['original_vectors'].dtype==a['vectors'].dtype and a['original_vectors'].shape==a['vectors'].shape,'original eigenbank dtype/shape differs')
        require(np.isfinite(a['values']).all() and np.isfinite(a['vectors']).all() and a['values'].tobytes()==a['original_values'].tobytes() and a['vectors'].tobytes()==a['original_vectors'].tobytes(),'eigenbank phase/order/finite differs')
        require(a['modes'].dtype.str=='<i4' and a['modes'].shape==(8,) and np.all(a['modes']==0),'mode bank differs')
        require(a['types'].dtype.str=='<i8' and a['types'].tolist()==layout['types'] and a['names'].tolist()==layout['names'] and a['forced'].dtype.str=='<i8' and a['forced'].size==0,'eigenbank typed address mapping differs')
        eig[role]=a
    for i in range(8):
        pair=jread(paths[f'pair_plus_{i:02d}']);value=eig['eigen_plus']['values'][i]
        require(pair['pencil_qualified'] and pair['physical_qualified'] and pair['lambda_real']==value.real and pair['lambda_imag']==value.imag,'returned control qualification/value differs')
    reference=jread(paths['reference'])
    if profile['matrix_only']:
        matrix_only_reference(reference,packet['reference_rectangles'])
    else:
        rectangles=[{'source_CSV_line':r['CSV_line'],'ca_bounds':r['ca_bounds'],'growth_bounds':r['growth_bounds']} for r in reference['reference']['0']['eligible_rectangles']]
        require(packet['reference_rectangles']==rectangles and len(rectangles)==2,'exact reference rectangle bytes differ')
        for r in rectangles:require(r['ca_bounds'][0]<=-.00315<=r['ca_bounds'][1] and r['growth_bounds'][0]<r['growth_bounds'][1],'rectangle ranges invalid')
    geometry=None if profile['matrix_only'] else {}
    for phase in (() if profile['matrix_only'] else ('gas','liquid')):
        a=npz(paths[phase+'_geometry'],witness[phase+'_geometry']['sha256']);g={k:a[k] for k in ('coordinates','connectivity','node_indices','position_equations','position_pins')}
        require(g['coordinates'].dtype.str=='<f8' and g['coordinates'].shape==(1481,2) and np.isfinite(g['coordinates']).all(),'geometry coordinates invalid')
        require(g['connectivity'].dtype.str=='<i8' and g['connectivity'].shape==(740,3) and np.all((g['connectivity']>=0)&(g['connectivity']<1481)),'geometry connectivity invalid')
        require(g['node_indices'].dtype.str=='<i8' and np.array_equal(g['node_indices'],np.arange(1481)),'first-seen interface node inventory differs')
        c=g['connectivity'];require(np.array_equal(c[1:,0],c[:-1,2]) and len(np.unique(c))==1481 and len(np.unique(c[:,1]))==740,'P2 chain incidence differs')
        require(not set(c[:,1].tolist())&set(c[:,[0,2]].ravel().tolist()),'P2 midside/endpoint incidence overlaps')
        eq=g['position_equations'];pins=g['position_pins'];require(eq.dtype.str=='<i8' and eq.shape==(1481,2) and pins.dtype.kind=='b' and pins.shape==eq.shape and np.array_equal(pins,eq<0),'geometry position incidence invalid')
        require(np.all(eq>=-1) and np.all(eq<t['n']) and len(np.unique(eq[eq>=0]))==2960,'geometry equation ranges/duplicates differ')
        for component,suffix in ((0,'/mesh_y'),(1,'/mesh_x')):
            for i in eq[:,component][eq[:,component]>=0]:require(layout['names'][int(ty[i])].startswith(phase+'/') and layout['names'][int(ty[i])].endswith(suffix),'geometry coordinate/address type differs')
        geometry[phase]=g
    derived_protected=[paths['native_manifest'].parent,paths['case'].parent,paths['source_model'].parent.parent,Path(packet['environment']['interpreter']).parent.parent]
    if not profile['matrix_only']:derived_protected.append(paths['gas_geometry'].parent)
    out=validate_output(out,base,[*packet['protected_roots'],*derived_protected],paths)
    metadata={'n':t['n'],'J_nnz':j.nnz,'M_nnz':m.nnz,'mass_rows':len(rows),'mass_columns':2960,'explicit_mass_zeros':0,'globals_preserved':layout['globals'],'forced':[],
            'model_hash':witness['case']['sha256'],'time_factor':FACTOR,'pencil_sign':packet['authority']['pencil_sign'],'native_imports':False,'factorisations':0,'eigensolves':0}
    if profile['matrix_only']:
        metadata.update(matrix_only=True,geometry_diagnostics_available=False,
                        unavailable_geometry_roles=sorted(GEOMETRY_ROLES),reference_rectangles=[])
    return {'j':j,'m':m,'rows':rows,'layout':layout,'eigen':eig,'geometry':geometry,'paths':paths,'witness':witness,'out':out,'metadata':metadata}

def self_memory():
    result={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmHWM:','VmRSS:')):
            key,value,unit=line.split();require(unit=='kB','unexpected self memory unit');result[key[:-1]+'_bytes']=int(value)*1024
    require(set(result)=={'VmHWM_bytes','VmRSS_bytes'},'self memory unavailable')
    return result

def own_cgroup():
    lines=Path('/proc/self/cgroup').read_text().splitlines();relative=next((x.split(':',2)[2] for x in lines if x.startswith('0::')),None)
    require(relative is not None,'unified execution cgroup unavailable')
    root=Path('/sys/fs/cgroup');p=(root/relative.lstrip('/')).resolve()
    require(p.is_relative_to(root),'execution cgroup escaped root');return p

def external_execution_caps():
    cg=own_cgroup();require((cg/'memory.max').read_text().strip()==str(CAPS['memory_bytes']),'external64GiB memory cap absent')
    require((cg/'memory.swap.max').read_text().strip()=='0','external zero-swap cap absent')
    affinity=sorted(os.sched_getaffinity(0));require(len(affinity)==2,'execution must use two logical CPUs of one whole core')
    topology=[]
    for cpu in affinity:
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology');topology.append((int((p/'physical_package_id').read_text()),int((p/'core_id').read_text())))
    require(topology[0]==topology[1],'execution CPUs are not one whole core')
    return {'cgroup_path':str(cg),'memory_max_bytes':CAPS['memory_bytes'],'memory_swap_max_bytes':0,'affinity':affinity,
        'runtime_limit_seconds_external_required':10800,'storage_reservation_bytes_external_required':2*1024**3,
        'limit_interpretation':'Root external caps; not measurements of need. Worker also checks elapsed time/output bytes.'}

def resource_measurement():
    result=self_memory()
    try:
        cg=own_cgroup();result['cgroup_memory_peak_bytes']=int((cg/'memory.peak').read_text().strip())
    except (OSError,ValueError,AdmissionFailure):result['cgroup_memory_peak_bytes']=None
    return result
