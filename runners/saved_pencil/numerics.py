"""Saved-pencil numerical kernels; no native solver dependency.

Classification thresholds, inverse-norm estimates and condition proxies are
numerical bookkeeping. They are never exact-root exclusion or error bounds.
"""
from __future__ import annotations
import math,time
import numpy as np
from scipy import linalg,sparse
from scipy.sparse.linalg import LinearOperator,onenormest,splu

class NumericalFailure(RuntimeError):pass

def finite(value,label):
    if not np.isfinite(value).all():raise NumericalFailure('nonfinite '+label)

def c_record(value):
    z=complex(value)
    return {'real':float(z.real) if math.isfinite(z.real) else None,
            'imag':float(z.imag) if math.isfinite(z.imag) else None}

def norm_fro(matrix):return float(np.linalg.norm(matrix.data if sparse.issparse(matrix) else matrix))

def mass_rows(mass):
    rows=[]
    for i in np.flatnonzero(np.diff(mass.indptr)>0):
        lo,hi=mass.indptr[i:i+2]
        if np.any(mass.data[lo:hi]!=0):rows.append(int(i))
    return np.asarray(rows,dtype=np.int64)

def injected(n,rows,values):
    value=np.asarray(values)
    if value.shape[0]!=len(rows):raise NumericalFailure('injection shape differs')
    rhs=np.zeros((n,)+value.shape[1:],dtype=value.dtype);rhs[rows]=value;return rhs

class RealInverse:
    """Exactly the real SuperLU/default-options producer path; never inspect L/U."""
    def __init__(self,k,factor=splu):
        if k.dtype!=np.float64:raise NumericalFailure('shifted matrix must remain real float64')
        self.forward_rhs=0;self.transpose_rhs=0
        self.lu=factor(k.tocsc())
        self.nnz=int(self.lu.nnz)
    def solve(self,rhs,trans='N'):
        value=np.asarray(rhs)
        if value.dtype.kind=='c':raise NumericalFailure('complex RHS handed to real factor')
        count=1 if value.ndim==1 else value.shape[1]
        if trans=='N':self.forward_rhs+=count
        elif trans=='T':self.transpose_rhs+=count
        else:raise NumericalFailure('unsupported transpose request')
        return self.lu.solve(value,trans=trans)

def refined_real(inverse,k,rhs,steps=2):
    if steps!=2:raise NumericalFailure('fixed refinement count differs')
    x=inverse.solve(rhs);initial=x.copy()
    records=[]
    kf=norm_fro(k)
    for iteration in range(3):
        residual=rhs-k@x;finite(residual,'solve residual')
        absolute=float(np.linalg.norm(residual));xnorm=float(np.linalg.norm(x));rhsnorm=float(np.linalg.norm(rhs))
        column1=float(np.max(np.sum(abs(residual),axis=0))) if residual.ndim==2 else float(np.sum(abs(residual)))
        item={'iteration':iteration,'absolute_residual_fro':absolute,'max_column_residual_1norm':column1,
              'rhs_fro':rhsnorm,'solution_fro':xnorm,'K_fro':kf,
              'backward_residual_fro':absolute/max(kf*xnorm+rhsnorm,1e-300)}
        records.append(item)
        if iteration<2:
            correction=inverse.solve(residual);finite(correction,'refinement correction');x+=correction
            item['correction_fro']=float(np.linalg.norm(correction))
    return x,initial,records

def refined_complex(inverse,k,rhs):
    value=np.asarray(rhs,dtype=np.complex128)
    xr,_,rr=refined_real(inverse,k,value.real)
    xi,_,ri=refined_real(inverse,k,value.imag)
    return xr+1j*xi,{'real':rr,'imag':ri}

def reduced_block(inverse,k,b,rows,first,last):
    n=k.shape[0];rhs=np.zeros((n,last-first),dtype=np.float64)
    rhs[rows[first:last],np.arange(last-first)]=1.
    x,initial,records=refined_real(inverse,k,rhs)
    result=b@x;unrefined=b@initial;finite(result,'reduction block')
    return result,{'first_column':first,'last_column_exclusive':last,'solve_refinements':records,
        'refinement_G_change_fro':float(np.linalg.norm(result-unrefined)),
        'refinement_G_change_max_column_1norm':float(np.max(np.sum(abs(result-unrefined),axis=0))),
        'product_absolute_column_sum_1norm_proxy':float(np.max(np.sum(abs(b)@abs(x),axis=0)))}

def inverse_1norm_proxy(inverse,k):
    before=(inverse.forward_rhs,inverse.transpose_rhs)
    op=LinearOperator(k.shape,matvec=lambda x:inverse.solve(x),rmatvec=lambda x:inverse.solve(x,trans='T'),
        matmat=lambda x:inverse.solve(x),rmatmat=lambda x:inverse.solve(x,trans='T'),dtype=np.float64)
    estimate=float(onenormest(op,t=2,itmax=5))
    finite(estimate,'inverse norm estimate')
    return {'K_inverse_1norm_estimate':estimate,'forward_rhs_count':inverse.forward_rhs-before[0],
        'transpose_rhs_count':inverse.transpose_rhs-before[1],
        'interpretation':'Hager/Higham numerical estimate, not a certified upper bound.'}

def classify_roots(g,mu,left,right,shift,solve_proxy_1):
    finite(g,'G');threshold=float(np.finfo(float).eps*len(mu)*norm_fro(g))
    records=[]
    for i,z in enumerate(mu):
        mag=float(abs(z));lam=None
        if np.isfinite(z) and z!=0:
            with np.errstate(over='ignore',divide='ignore',invalid='ignore'):mapped=shift+1/z
            if np.isfinite(mapped):lam=complex(mapped)
        classification='unresolved' if not np.isfinite(z) or mag<=threshold else 'poorly_resolved' if mag<=1000*threshold else 'numerically_resolved'
        denominator=float(abs(np.vdot(left[:,i],right[:,i])))
        condition=(float(np.linalg.norm(right[:,i],1))*float(np.linalg.norm(left[:,i],np.inf))/denominator) if denominator else None
        sensitivity=None
        if condition is not None and mag and math.isfinite(condition):
            with np.errstate(over='ignore',divide='ignore',invalid='ignore'):sensitivity=condition*solve_proxy_1/(mag*mag)
            if not math.isfinite(sensitivity):sensitivity=None
        records.append({'index':i,'mu':c_record(z),'mu_abs':mag if math.isfinite(mag) else None,
            'classification':classification,'mapped_lambda':c_record(lam) if lam is not None else None,
            'condition_1norm_proxy':condition if condition is not None and math.isfinite(condition) else None,
            'first_order_lambda_sensitivity_1norm_proxy':sensitivity,
            'selected':False,'lifted':False})
    return records,{'empirical_reciprocal_threshold':threshold,'poorly_resolved_multiplier':1000,
        'cutoff_is_not_an_error_bound':True,'exact_root_exclusion_certified':False,
        'condition_norm_definition':'||right||_1 ||left||_infinity / |left^H right|; first-order simple-root proxy only.'}

def mapped(record):
    x=record['mapped_lambda'];return complex(x['real'],x['imag']) if x is not None else None

def sanity_match(records,returned,tolerance):
    """Numerical comparison, not an exact-eigenvalue or input-defect proof."""
    eligible=[r for r in records if r['classification']=='numerically_resolved' and mapped(r) is not None]
    used=set();matches=[]
    for i,value in enumerate(returned):
        choices=[(abs(mapped(r)-value),r['index']) for r in eligible if r['index'] not in used]
        distance,index=min(choices,default=(math.inf,None))
        if distance<=tolerance:used.add(index)
        matches.append({'returned_index':i,'reduced_index':index,'distance':float(distance) if math.isfinite(distance) else None,'matched':bool(distance<=tolerance)})
    return {'passed':all(r['matched'] for r in matches),'matches':matches,
            'failure_interpretation':'Numerical sanity failure/inconclusive; not proof of code or input defect.'}

def select_lifts(records,sanity,returned,rectangles,factor,ca,cap=24):
    if cap!=24:raise NumericalFailure('lift cap changed')
    if not sanity['passed']:return {'indices':[],'sanity_failed':True,'absence_claim_allowed':False}
    controls=[x['reduced_index'] for x in sanity['matches'][:4]]
    if len(set(controls))!=4:raise NumericalFailure('four distinct lift controls required')
    lead=float(np.max(np.real(returned)))
    band=[];challengers=[]
    for r in records:
        value=mapped(r);hits=[]
        if value is not None:
            growth=factor*value.real
            hits=[v['source_CSV_line'] for v in rectangles if v['ca_bounds'][0]<=ca<=v['ca_bounds'][1] and v['growth_bounds'][0]<=growth<=v['growth_bounds'][1]]
        r['rectangle_ids']=hits;r['mapped_paper_growth']=factor*value.real if value is not None else None
        r['additional_challenger_numerical_mapping']=value is not None and value.real>lead and r['index'] not in controls
        if hits and r['index'] not in controls:band.append(r['index'])
        elif r['additional_challenger_numerical_mapping']:challengers.append(r['index'])
    lookup={r['index']:r for r in records}
    order=lambda indices:sorted(indices,key=lambda i:(-mapped(lookup[i]).real,i))
    priority=order(band)+order(challengers);selected=controls+priority[:cap-4]
    for i in selected:lookup[i]['selected']=True
    unlifted=priority[cap-4:]
    return {'indices':selected,'control_indices':controls,'rectangle_priority_indices':order(band),
        'additional_challenger_indices':order(challengers),'unlifted_candidate_indices':unlifted,
        'cap_truncated_any_candidate_set':bool(unlifted),'all_unselected_root_indices':[r['index'] for r in records if r['index'] not in selected],
        'unresolved_root_indices':[r['index'] for r in records if r['classification']=='unresolved'],
        'poorly_resolved_root_indices':[r['index'] for r in records if r['classification']=='poorly_resolved'],
        'absence_claim_allowed':False,'scope':'Four returned leading controls, then individual exact-rectangle candidates, then rightmost additional numerical challengers; every other root retained.'}

def diagnose(value,vector,j,m,layout,factor,limits):
    v=np.asarray(vector,dtype=np.complex128);finite(v,'lifted vector');finite(value,'mapped lambda')
    vn=float(np.linalg.norm(v));jn=norm_fro(j);mn=norm_fro(m);ms=float(np.max(abs(m.data),initial=0))
    jv=j@v;mv=m@v;res=float(np.linalg.norm(jv-value*mv))
    back=res/max((jn+abs(value)*mn)*vn,1e-300);mass=float(np.linalg.norm(mv))/max(ms*vn,1e-300)
    forced=float(np.linalg.norm(v[np.asarray(layout['forced'],dtype=np.int64)]))/max(vn,1e-300)
    actions={}
    for name,ids in layout['globals'].items():
        row=j.getrow(ids[0]);action=complex((row@v)[0])
        actions[name]={'J_action':c_record(action),'absolute_J_action':float(abs(action)),
            'relative_J_action':float(abs(action)/max(np.linalg.norm(row.data)*vn,1e-300)),
            'coefficient_abs':float(abs(v[ids[0]]))}
    reasons=[]
    if not vn:reasons.append('zero vector')
    if not all(math.isfinite(x) for x in (back,mass,forced)):reasons.append('invalid eigenvector/diagnostic')
    if back>limits['backward']:reasons.append('original-pencil backward residual')
    if mass<=limits['mass_action']:reasons.append('zero/near-null mass action')
    if forced>limits['forced_coefficient']:reasons.append('forced coefficients')
    if any(a['relative_J_action']>limits['backward'] for a in actions.values()):reasons.append('algebraic constraint action')
    safe=lambda x:float(x) if math.isfinite(x) else None
    return {'lambda':c_record(value),'paper_growth':safe(factor*value.real),'paper_frequency':safe(-factor*value.imag),
        'vector_norm_l2':safe(vn),'absolute_residual_l2':safe(res),'backward_residual_l2':safe(back),'mass_action_ratio':safe(mass),
        'forced_coefficient_ratio':safe(forced),'constraint_actions':actions,'rejection_reasons':reasons,
        'computed_candidate_passes_original_pencil_gates':not reasons,'rigorously_certified_eigenvalue':False}

def p1(hwm_bytes):return hwm_bytes+1.5*1024**3<=60*1024**3
def p2(elapsed,block_seconds,remaining,total_seconds=10800):return elapsed+1.5*remaining*block_seconds+900<=total_seconds

def geometry_diagnostics(vector,geometry,helper,j,volume_row):
    results={}
    for phase,g in geometry.items():
        eq=g['position_equations'];move=np.zeros_like(g['coordinates'],dtype=np.complex128);free=eq>=0;move[free]=vector[eq[free]]
        volume=helper['volume_differential'](g['coordinates'],g['connectivity'],move)
        w,n,dn=helper['quadrature']();c=g['connectivity'];points=g['coordinates'];p=points[c]
        zp=p[:,:,0]@dn.T;rp=p[:,:,1]@dn.T;r=p[:,:,1]@n.T;speed=np.sqrt(zp*zp+rp*rp)
        if np.any(speed==0):raise NumericalFailure('degenerate P2 translation diagnostic')
        nz=-rp/speed;nr=zp/speed;motion=move[c];eta=(motion[:,:,0]@n.T)*nz+(motion[:,:,1]@n.T)*nr
        weights=2*math.pi*r*speed*w[None,:];a=float(np.sum(abs(eta)**2*weights));b=float(np.sum(nz*nz*weights))
        overlap=float(abs(np.sum(np.conj(eta)*nz*weights))/math.sqrt(a*b)) if a and b else None
        results[phase]={'deltaV':c_record(volume['delta_volume']),'absolute_deltaV':float(abs(volume['delta_volume'])),
            'absolute_integrand_scale':volume['absolute_integrand_scale'],'raw_geometric_surface_l2':math.sqrt(a),
            'axis_translation_overlap':overlap,'scope':'Saved P2/equation-addressed analytic diagnostics; no native epsilon/halving/normal qualification.'}
    action=complex((j.getrow(volume_row)@vector)[0]);gas=complex(results['gas']['deltaV']['real'],results['gas']['deltaV']['imag'])
    results['complete_stored_volume_row_action']=c_record(action)
    results['gas_deltaV_plus_complete_row_abs']=float(abs(gas+action));return results
