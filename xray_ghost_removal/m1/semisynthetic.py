"""Apply fixed, previously fitted M1 parameters to measured-signal overlays.

Uses frozen block16 Y, not model-generated ghosts. All targets are designated
baseline references, not certified clean images. Reuses one stable subdirectory;
no Markdown is generated. Testing performs no parameter fit or segmentation.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from xray_ghost_removal.common.paths import PROJECT_ROOT
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo
import shutil
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'ghost-mpl'))
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from xray_ghost_removal.m1.frozen import load_m1_pair
from xray_ghost_removal.common.provenance import sha256 as sha
from xray_ghost_removal.common.spatial import block_mean as blocks
from xray_ghost_removal.common.dicom import save_dicom

ROOT=PROJECT_ROOT
RAW=ROOT/'data/raw/AI修残影例图'
OUT=ROOT/'outputs/dicom_review'
DONORS=((3,4),(5,6))
TARGETS=(4,6,27)


def lift(values,valid,shape,block=16):
    values=np.asarray(values,dtype=np.float64);valid=np.asarray(valid,dtype=bool)
    if values.shape!=valid.shape or tuple(n*block for n in values.shape)!=tuple(shape):
        raise ValueError('Block grid and full-size geometry do not match')
    if not np.isfinite(values[valid]).all():raise ValueError('Non-finite valid block')
    return np.repeat(np.repeat(np.where(valid,values,0),block,0),block,1)


def synthesize(base,signal,prediction):
    base=np.asarray(base,dtype=np.float64)
    if not (base.shape==signal.shape==prediction.shape):raise ValueError('Overlay geometry mismatch')
    if not all(np.isfinite(a).all() for a in (base,signal,prediction)):
        raise ValueError('Non-finite synthesis inputs')
    contaminated=base+signal
    return contaminated,contaminated-prediction


def error_metrics(error,mask):
    e=np.asarray(error)[mask]
    if not e.size:return dict(pixels=0,mae=None,rmse=None,p99_abs=None,bias=None)
    return dict(pixels=int(e.size),mae=float(np.mean(np.abs(e))),rmse=float(np.sqrt(np.mean(e*e))),
                p99_abs=float(np.percentile(np.abs(e),99)),bias=float(np.mean(e)))


def comparisons(base,ghosted,corrected,signal,prediction,valid,target_support,donor_support):
    regions=dict(all=np.ones(base.shape,bool),donor_support=donor_support,
                 overlap=donor_support & target_support,donor_in_air=donor_support & ~target_support,
                 current_object=target_support,remaining_air=~target_support & ~donor_support)
    before=ghosted-base;after=corrected-base
    result={}
    for name,mask in regions.items():
        mask=mask & valid
        a,b=error_metrics(before,mask),error_metrics(after,mask)
        reduction=1-b['rmse']/a['rmse'] if a['rmse'] else None
        result[name]=dict(before=a,after=b,rmse_relative_reduction=reduction)
    return result


def review_template(source,study,index,stamp):
    ds=deepcopy(source)
    ds.PatientName='RESEARCH^M1-Synthetic'
    ds.PatientID='M1-SEMISYNTHETIC'
    ds.StudyDescription=f'M1 SYNTHETIC - BASE {index:02d} LIGHT'
    ds.StudyDate=stamp.strftime('%Y%m%d');ds.StudyTime=stamp.strftime('%H%M%S')
    ds.SeriesDate=ds.StudyDate;ds.SeriesTime=ds.StudyTime
    ds.StudyID=f'M1-B{index:02d}';ds.StudyInstanceUID=study
    ds.ProtocolName='Measured ghost overlay, fixed M1; research'
    return ds


def main():
    manifest_path=ROOT/'outputs/mhinge_cross_pair_v1/analysis_results.json'
    manifest=json.loads(manifest_path.read_text())
    if not manifest['frozen_background']['verified']:raise ValueError('Unverified manifest')
    if manifest['configuration']['block_sizes']!=[16] or manifest['configuration']['background_modes']!=['masked_hybrid_spline']:
        raise ValueError('Unexpected frozen model definition')
    mask_path=ROOT/'outputs/pseudo_ghost_mechanism_v1/sam_masks_single_lag.npz'
    config_path=ROOT/'configs/background_frozen_v1.json'
    if sha(mask_path)!=manifest['frozen_background']['mask_archive_sha256'] or sha(config_path)!=manifest['frozen_background']['sha256']:
        raise ValueError('Frozen background or mask provenance mismatch')
    source_hashes={};datasets={}
    needed={(i,'light') for i in TARGETS}|{(i,'light') for i,_ in DONORS}|{(j,'dark') for _,j in DONORS}
    for index,role in sorted(needed):
        name=f'{index}-{role}.dcm';path=RAW/name
        expected=manifest['input_provenance']['files'][name]['sha256']
        if sha(path)!=expected:raise ValueError(f'Raw hash mismatch {name}')
        ds=pydicom.dcmread(path)
        if (ds.PhotometricInterpretation!='MONOCHROME2' or ds.PixelRepresentation!=0
            or float(ds.get('RescaleSlope',1))!=1 or float(ds.get('RescaleIntercept',0))!=0
            or any(k in ds for k in ('ModalityLUTSequence','VOILUTSequence','PixelPaddingValue'))):
            raise ValueError('Unsupported grayscale semantics')
        source_hashes[name]=expected;datasets[(index,role)]=ds
    models={}
    for light,dark in DONORS:
        maps,fit,provenance=load_m1_pair(manifest,light,dark)
        raw_light=blocks(datasets[(light,'light')].pixel_array.astype(np.float64))
        x=raw_light-np.percentile(raw_light,75)
        if not np.allclose(x,maps['source'],atol=.002,rtol=0):raise ValueError('Frozen source mismatch')
        if not np.allclose(blocks(datasets[(dark,'dark')].pixel_array.astype(np.float64)),maps['raw_dark'],atol=.002,rtol=0):
            raise ValueError('Frozen dark mismatch')
        # Previously fitted single alpha/intercept: not fold maps and not refitted here.
        alpha=fit['alpha_full_descriptive'];intercept=fit['intercept_full_descriptive']
        prediction=alpha*maps['source'].astype(np.float64)+intercept
        models[(light,dark)]=dict(maps=maps,prediction=prediction,
            parameters=dict(alpha=alpha,intercept=intercept,status='previous_full_fit_frozen_before_synthesis'),provenance=provenance)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai'))
    rows=[]
    with tempfile.TemporaryDirectory(prefix='m1-semisynthetic-') as directory, np.load(mask_path) as archive:
        stage=Path(directory);base_copies={};windows={};target_masks={}
        for index in TARGETS:
            raw_ds=datasets[(index,'light')];base=raw_ds.pixel_array.astype(np.float64)
            ds=review_template(raw_ds,generate_uid(),index,stamp)
            lo,hi=np.percentile(base,[.5,99.5]);wc=float((lo+hi+1)/2);ww=float(max(hi-lo+1,2))
            # Existing target masks are used for evaluation/windowing only; no segmentation or fit.
            target_mask=archive[f'light_{index}'].astype(bool)
            air=~target_mask
            air_vals=base[air]
            narrow_wc=float(np.median(air_vals));narrow_ww=100.0
            ds.WindowCenter=narrow_wc;ds.WindowWidth=narrow_ww
            ds.WindowCenterWidthExplanation='GHOST DETAIL'
            ds.VOILUTFunction='LINEAR'
            base_copies[index]=ds;windows[index]=((narrow_wc,narrow_ww),(wc,ww));target_masks[index]=target_mask
        for light,dark in DONORS:
            model=models[(light,dark)];maps=model['maps'];valid=maps['evaluated_mask'].astype(bool)
            for target in TARGETS:
                case=f'G{light:02d}{dark:02d}_B{target:02d}'
                base_ds=review_template(base_copies[target],generate_uid(),target,stamp)
                case_label=f'G{light}-{dark} + L{target}'
                base_ds.StudyDescription=f'M1 SYNTHETIC {case_label}'
                base_ds.StudyID=case
                base_ds.SeriesNumber=1;base_ds.InstanceNumber=1
                base=base_ds.pixel_array.astype(np.float64)
                baseline_name=f'{case}_original_light.dcm'
                baseline_audit=save_dicom(base,base_ds,stage/'dicom'/baseline_name,
                    series_description=f'1 ORIGINAL LIGHT | {case_label}',
                    derivation_description='Pixel-identical baseline reference; one three-series study per synthetic case')
                assert baseline_audit['clipped_low']==baseline_audit['clipped_high']==0
                base_ds=pydicom.dcmread(stage/'dicom'/baseline_name)
                assert np.array_equal(base_ds.pixel_array,datasets[(target,'light')].pixel_array)
                signal=lift(maps['observable_signal'],valid,base.shape)
                prediction=lift(model['prediction'],valid,base.shape)
                valid_pixels=lift(valid,valid,base.shape).astype(bool)
                donor_support=lift(maps['source_support'],np.ones(valid.shape,bool),base.shape).astype(bool)
                proposed,corrected=synthesize(base,signal,prediction)
                # Explicitly check that the baseline affects neither the fitted parameters nor the residual.
                assert np.allclose(corrected-base,signal-prediction,atol=1e-10,rtol=0)
                floating_metrics=comparisons(base,proposed,corrected,signal,prediction,valid_pixels,target_masks[target],donor_support)
                stored=[];exports={};names={'baseline':baseline_name}
                for kind,a,offset,label in (('ghosted',proposed,2,'GHOST ADDED'),('corrected',corrected,3,'M1 CORRECTED')):
                    ds=deepcopy(base_ds);ds.SeriesNumber=offset
                    ds.InstanceNumber=1
                    name=f'{case}_{kind}.dcm'
                    desc=f'Measured Y from {light}-light->{dark}-dark added x1 to {target}-light; block16 lift; '+('fixed full-fit M1 subtraction' if kind=='corrected' else 'no model-generated ghost')
                    audit=save_dicom(a,ds,stage/'dicom'/name,
                        series_description=f'{offset} {label} | {case_label}',
                        derivation_description=desc,source_images=[base_ds])
                    reread=pydicom.dcmread(stage/'dicom'/name);decoded=reread.pixel_array.astype(np.float64)
                    assert float(reread.WindowCenter)==float(base_ds.WindowCenter)
                    assert float(reread.WindowWidth)==float(base_ds.WindowWidth)
                    expected=np.rint(np.clip(a,0,2**int(ds.BitsStored)-1))
                    assert np.array_equal(decoded,expected)
                    stored.append(decoded);exports[kind]=audit;names[kind]=name
                stored_metrics=comparisons(base,*stored,signal,prediction,valid_pixels,target_masks[target],donor_support)
                # Block means must match the frozen signal/prediction apart from quantization.
                if not any(a['clipped_low'] or a['clipped_high'] for a in exports.values()):
                    assert np.max(np.abs(blocks(stored[0]-base)[valid]-maps['observable_signal'][valid]))<=.5+1e-8
                    assert np.max(np.abs(blocks(stored[1]-base)[valid]-(maps['observable_signal']-model['prediction'])[valid]))<=.5+1e-8
                row=dict(case=case,donor=f'{light}->{dark}',baseline_light=target,parameters=model['parameters'],
                    measured_signal='frozen dark - BG, constant per block16, x1, valid blocks only',
                    floating_metrics=floating_metrics,stored_metrics=stored_metrics,windows=windows[target],exports=exports,
                    overlap_fraction_of_donor=float(np.sum(donor_support & target_masks[target])/max(donor_support.sum(),1)),
                    artifacts={k:dict(path='dicom/'+name,sha256=sha(stage/'dicom'/name)) for k,name in names.items()},
                    study_instance_uid=str(base_ds.StudyInstanceUID),series_order=['ORIGINAL LIGHT','GHOST ADDED','M1 CORRECTED'])
                rows.append(row)
                a=stored_metrics['all'];print(f"{case}: RMSE {a['before']['rmse']:.4f} -> {a['after']['rmse']:.4f}; clipping {[(v['clipped_low'],v['clipped_high']) for v in exports.values()]}",flush=True)
        all_dicoms=list((stage/'dicom').glob('*.dcm'))
        uids=[str(pydicom.dcmread(p,stop_before_pixels=True).SOPInstanceUID) for p in all_dicoms]
        assert len(uids)==len(set(uids))==18
        assert all(sha(RAW/name)==digest for name,digest in source_hashes.items())
        report=dict(generated_at=stamp.isoformat(),experiment='seen measured ghosts on new baseline light references',
            scientific_definition='B + frozen block16 Y; subtract previously fitted full-fit alpha*X+b; no fitting on synthetic targets',
            limitations=['Baseline light is not certified clean; only recovery of added signal is evaluated.',
                'Donor dark was used in the original fit; this is not unseen-ghost generalization.',
                'One parameter set per donor; not one universal coefficient set.',
                'Assumed additive stored-pixel synthesis; not established real detector interaction.',
                'Block16 lift preserves baseline texture, not pixel-level ghost modeling.',
                'Without clipping, floating residual equals donor Y minus prediction independent of baseline.'],
            rationale_for_output='Current semisynthetic DICOMs replace obsolete review derivatives in the stable output directory; retain only DICOM and machine audit.',
            manifest_sha256=sha(manifest_path),code_sha256=sha(Path(__file__)),input_sha256=source_hashes,
            model_inputs={f'{i}->{j}':m['provenance'] for (i,j),m in models.items()},cases=rows,
            dicom_count=18,display_layout='six separate studies, three ordered single-image series per case')
        (stage/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        OUT.mkdir(parents=True,exist_ok=True)
        # Replace only files owned by this generator after the staged outputs pass verification.
        for p in OUT.iterdir():
            if p.name=='dicom' and p.is_dir():shutil.rmtree(p)
            elif p.name=='semisynthetic' and p.is_dir():shutil.rmtree(p)
            elif p.name in ('results.json','overview.png','review.md','review_results.json','selection.json','.DS_Store') or (p.is_file() and p.suffix=='.png'):p.unlink()
        for p in stage.iterdir():shutil.move(str(p),OUT/p.name)
    print('Results:',OUT)

if __name__=='__main__':main()
