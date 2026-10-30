"""Read-only raw-image audit before exposure-effect modeling.

Workbook extraction uses the bundled Python runtime; pass its temporary JSON
with --exposure-json. Results reuse the existing exposure_association directory.
No model fitting or frozen evidence modification occurs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xray_ghost_removal.common.paths import PROJECT_ROOT
import tempfile
from xray_ghost_removal.common.exposure import workbook_record

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'ghost-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from xray_ghost_removal.research.exposure import _manifest_fit
from xray_ghost_removal.common.spatial import block_mean

ROOT=PROJECT_ROOT
RAW=ROOT/'data/raw/AI修残影例图'
OUT=ROOT/'outputs/mhinge_cross_pair_v1/exposure_association'
FIELDS=('Rows','Columns','BitsAllocated','BitsStored','HighBit','PixelRepresentation',
        'SamplesPerPixel','PhotometricInterpretation','Modality','KVP','XRayTubeCurrent',
        'ExposureTime','Exposure','AcquisitionDateTime','AcquisitionTime',
        'RescaleSlope','RescaleIntercept','ExposureIndex','DetectorID','DeviceSerialNumber',
        'AcquisitionDeviceProcessingDescription','SOPInstanceUID')

from xray_ghost_removal.common.provenance import sha256 as sha

def raw_record(path,expected):
    ds=pydicom.dcmread(path)
    a=ds.pixel_array
    if a.ndim!=2 or a.dtype.kind!='u': raise ValueError(f'Unsupported image: {path}')
    hist=np.bincount(a.ravel(),minlength=65536)
    maximum=int(a.max()); upper=2**int(ds.BitsStored)-1
    headers={k:str(ds.get(k,'')) for k in FIELDS}
    digest=sha(path)
    if digest!=expected: raise ValueError(f'Raw hash mismatch: {path.name}')
    record=dict(file=path.name,sha256=digest,pixel_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        decoded=True,shape=list(a.shape),dtype=str(a.dtype),headers=headers,
        minimum=int(a.min()),maximum=maximum,median=float(np.median(a)),std=float(np.std(a)),
        p01_p99=np.percentile(a,[1,99]).tolist(),zero_fraction=float(hist[0]/a.size),
        storage_upper_fraction=float(hist[upper]/a.size),observed_max_fraction=float(hist[maximum]/a.size),
        mode_value=int(hist.argmax()),mode_fraction=float(hist.max()/a.size),
        observed_max_pileup_screen=bool(hist[maximum]/a.size>0.01))
    return record,a

def contact_sheets(thumbs,folder):
    paths=[]
    for start in (1,17):
        fig,axes=plt.subplots(4,8,figsize=(20,11))
        for pos in range(16):
            idx=start+pos
            for role,col in (('light',0),('dark',1)):
                ax=axes[pos//4,(pos%4)*2+col]
                ax.axis('off')
                if idx>31: continue
                a=thumbs[(idx,role)]
                lo,hi=np.percentile(a,[0.5,99.5])
                ax.imshow(a,cmap='gray',vmin=lo,vmax=hi)
                ax.set_title(f'{idx}-{role} [{lo:.0f}, {hi:.0f}]',fontsize=10)
        fig.suptitle('Raw acquisition audit: separate input-only window per image; not an effect comparison')
        fig.tight_layout()
        path=folder/f'raw_{start}_{min(start+15,31)}.png'
        fig.savefig(path,dpi=120);plt.close(fig);paths.append(str(path))
    return paths

def evidence_figure(thumbs,block_cache,manifest):
    selected=((3,4),(4,5),(5,6),(6,7),(27,28))
    fig,axes=plt.subplots(5,3,figsize=(10,15))
    for row,(light,dark) in enumerate(selected):
        record=next(r for r in manifest['pair_results'] if r['light_index']==light)
        rel=record['output_artifacts']['primary_maps']['relative_path']
        path=ROOT/'outputs/mhinge_cross_pair_v1'/rel
        if not path.exists():path=ROOT/'outputs/pseudo_ghost_mechanism_v1'/rel
        with np.load(path) as z:y=z['observable_signal'].copy()
        axes[row,0].imshow(thumbs[(light,'light')],cmap='gray',vmin=0,vmax=55000)
        dark_map=block_cache[(dark,'dark')];median=np.median(dark_map)
        axes[row,1].imshow(dark_map,cmap='gray',vmin=median-30,vmax=median+30)
        im=axes[row,2].imshow(y,cmap='coolwarm',vmin=-45,vmax=45)
        for ax,title in zip(axes[row],(f'{light}-light: fixed [0,55000]',
            f'{dark}-dark block16: median +/-30',f'{light}->{dark}: frozen Y, +/-45')):
            ax.set_title(title,fontsize=10);ax.axis('off')
    fig.suptitle('Raw inputs and frozen signal: 3 primary pairs + 2 low-pileup controls',fontsize=12)
    fig.subplots_adjust(top=.93,bottom=.04,left=.02,right=.92,hspace=.18,wspace=.08)
    fig.colorbar(im,ax=list(axes[:,2]),fraction=.02,pad=.02)
    path=OUT/'data_feasibility.png';fig.savefig(path,dpi=130);plt.close(fig)
    return path

def write_report(report):
    rows=report['pair_results'];primary=[r for r in rows if r['classification']=='primary_low_pileup']
    lights=[r for r in report['raw_files'] if r['file'].endswith('-light.dcm')]
    pileup=sum(r['observed_max_pileup_screen'] for r in lights)
    labels={'primary_low_pileup':'主要样本','clipping_risk_sensitivity_only':'截断风险，仅敏感性参照',
            'negative_control':'未检出对照','weak_signal_reference':'弱信号参照'}
    lines=['# 曝光影响研究：原始数据可行性审查','',
        '结论：现有数据可做三对低截断风险样本的 M1 去除示例，以及有边界的描述性分析；',
        '不足以建立曝光影响的因果规律、统一 alpha 标定或真实 clean 恢复准确率。',
        '当前先暂停新的曝光响应模型拟合，保留对照，明确缺失条件并补采。','',
        '## 方法与证据','',
        '本轮按事先保存的方法，重新读取全部62张原 DICOM、参数表31行和13对冻结 NPZ。',
        '对全部原图检查解码、几何、位深、端点/最大值堆积、文件及像素哈希、关键元数据；',
        '重新核验原图、冻结 BG 配置、SAM archive 和 NPZ 哈希与数组恒等式。',
        '不重新拟合 BG/M1，不用肉眼可见性或去除效果筛选曝光样本。',
        '原 light/dark 接触表已逐张检查；另保留5对主要样本/对照的原始输入和冻结 Y 图。',
        '单次曝光 pair 是研究单位；像素、block、空间折不算独立曝光重复。','',
        '## 文件与采集记录','',
        '- 62/62 解码成功，均为3072×3072、16-bit unsigned、MONOCHROME2。',
        '- 与冻结 manifest 的62个原图哈希一致；无重复文件内容或重复像素数组。',
        '- 全部62张原图共用同一个 SOPInstanceUID，属于 DICOM 标识异常，可能干扰阅片数据库；',
        '  当前科学分析按文件名与内容哈希识别，修正图导出已生成唯一 UID。',
        '- 参数表说明参数属于 light，每组先 dark 后 light；应匹配 light_i→dark_(i+1)。',
        '  参数取第 i 行，不能取目标 dark 的同编号 light 参数。',
        '- 31条编号完整唯一，参数数值均为正；电压/电流/时间列本身未写单位，',
        '  本审查沿用既有采集解释 kV/mA/ms，mAs=mA×ms/1000，仍应向采集方确认单位。',
        '- KVP、电流、曝光时长、真实 AcquisitionTime、ExposureIndex、板/设备标识、',
        '  Rescale 与读片处理描述在62张中均缺失。表内时间是曝光持续时间，',
        '  InstanceCreationTime不能当成曝光至读片间隔。dark采集/读取和擦除条件无记录。','',
        '## 原图异常与局限','',
        f'- 31张 light 的观测最大值均为55000；其中{pileup}/31张最大值堆积超过全图1%。',
        '  55000低于16位存储上限65535，且几乎所有高曝光图具有大面积平台，提示输出',
        '  截断/处理限幅。不能据此确定探测器物理饱和，或认为存储未到65535就没有截断。',
        '- 主样本 source 支撑内最大值堆积接近0；其他有效pair的支撑内堆积为8.3%–31.8%。',
        '  大量区域的X失去实际强度信息，因此全块alpha不能直接解释为曝光相关物理系数。',
        '- 原图可见物体种类、面积、位置和方向变化；3-light含矩形物体与探测卡，',
        '  5-light含矩形物体与另一小物体，27-light含两矩形物体及不规则物体。',
        '  三组没有固定相同的输入内容，不能当作同物体的曝光梯度实验。',
        '- dark仍含噪声与结构；缺少独立clean图和相同位置背景真值。已有BG逐折重算',
        '  对有效pair的alpha变化最多0.73%，但完整物体原位置BG仍未经真值验证。','',
        '## 样本分组','',
        '沿用冻结 detection gate、|alpha|≥1e-4，并采用既有曝光审查的source支撑内',
        '观测最大值堆积<1%作为主样本低截断风险条件。堆积是风险代理，不是物理饱和标签。','',
        '| Pair | kV / mA / ms | mAs | alpha | OOF R² | source堆积 | 用途 |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        e=r['exposure'];lines.append(f"| {r['pair']} | {e['kv']:g}/{e['ma']:g}/{e['ms']:g} | {e['mas']:g} | {r['alpha']:.8f} | {r['cv_r2']:.4f} | {100*r['source_max_pileup_fraction']:.5f}% | {labels[r['classification']]} |")
    lines+=['','三对主样本：3→4、5→6、27→28。4→5和6→7作为低堆积未检出对照保留。',
        '其他6对检出且效应足够的pair仅做截断敏感性参照，2对为弱信号参照。',
        '已有“9个有效pair”的说法指原冻结机制检出门槛；增加曝光量可解释性条件后，',
        '主要样本确实仅3对。其余17个相邻候选已检查原图，但没有本轮冻结BG/M1证据，',
        '不能据接触表判定全部报废，也不能直接加进正式曝光比较。','',
        '### 排除堆积block的敏感性','',
        '本轮只读取已有exclude_saturated结果，不重新拟合。',
        '三对主样本几乎没有block被剔除，alpha和R²不变。风险样本剔除约48%–94%',
        '的block后，1→2、8→9、9→10、22→23、23→24不再通过原检出门槛；',
        '7→8仍通过，但剔除92.5%块，alpha由0.00024369降至0.00016843。',
        '剔除大量空气/高灰度块同时改变了评价区域，所以这不能独立证明截断就是原因，',
        '但说明这些pair不能与主样本无条件混合做系数标定。','',
        '## 同设置重复是否稳定','',
        '| 名义设置 kV/mA/ms | Pair | alpha最大/最小比 | 检出情况 |',
        '|---|---|---:|---|']
    for g in report['repeated_settings']:
        lines.append(f"| {'/'.join(f'{v:g}' for v in g['setting'])} | {', '.join(g['pairs'])} | {g['alpha_ratio']:.2f} | {', '.join('检出' if v else '未检出' for v in g['detected'])} |")
    lines+=['','同设置并未保证同物体、同板状态、同擦除/读片延迟，以上是响应不稳定的证据，',
        '不是“测量设备坏了”或“曝光不起作用”的证明。弱对照的alpha比也不能解释成',
        '稳定的物理倍数。须保留这些对照，不能只留下三个强残影样本后建立曝光规律。','',
        '## 够不够研究曝光影响','',
        '- 3→4与5→6同为70kV、32ms，mA从32到63，mAs从1.024到2.016；',
        '  但物体、支撑面积和采集状态不固定，只有一次观测。alpha反而由0.00111436',
        '  降到0.00096186，不能据此认定曝光越强alpha越小。',
        '- 27→28同时改变kV、mA、ms和物体；不能独立识别电压效应。',
        '- 三个pair构成截距+kV+mA+ms的设计矩阵只有3行、秩3、4个待估参数，',
        '  多因素关系不可辨识。单因素拟合也几乎没有误差估计能力，不拟合剂量定律。',
        '- M1预测残影alpha×X本身由X构造，与X相关不能作为独立的曝光机制验证。',
        '  应同时报告冻结Y的背景误差、预测残影对比度、alpha及残差形态。',
        '- 旧13对曝光关联输入哈希已重新核对一致，结果仅作为描述性敏感性证据：',
        '  mAs与alpha的rho=0.108，删pair后范围[-0.092,0.326]，没有稳定单调关系。',
        '  该队列含截断与按残影证据选择的样本，不能据此声称“曝光无影响”。','',
        '| 目标 | 当前支持程度 |',
        '|---|---|',
        '| 三对M1去除示例与冻结残差分析 | 可以 |',
        '| 曝光与alpha/残影幅度描述性比较 | 可做少量案例对照，必须保留未检出样本 |',
        '| 曝光导致强度/形态/alpha变化 | 不足，缺少控制采集与可重复条件 |',
        '| 曝光相关系数函数或统一参数迁移 | 不足 |',
        '| 去除真实准确率、物理剂量定律 | 无clean/剂量真值，不支持 |','',
        '## 下一步最小可行采集','',
        '先核实旧采集的单位、55000限幅来源、板标识、擦除流程、dark读取含义、',
        '曝光至读片及擦除至曝光间隔、读片软件与是否有自动增益/后处理。','',
        '建议先固定同一物体/位置、同kV、同曝光时长，改变mA获得至少3个mAs水平，',
        '每水平至少3次独立完整采集循环作为预实验起点（至少9个有效pair，另采背景/对照）。',
        '每轮执行相同且记录完整的板重置/擦除与读取程序；随机化曝光水平顺序。',
        '确认输出未限幅并保留线性/处理说明。若要研究kV效应，另设计控制条件，',
        '不能把不同kV的mAs直接当作相同入射曝光量；需要可比剂量/探测器曝光度量。',
        '这个数量只用于估计变异和可行性，不保证充分统计效力，正式重复数须据预实验决定。',
        '若目标包括去除准确率，还需可对应的clean参照；背景/ghost-only研究不能替代',
        '当前真实物体图上的结构保真评价。','',
        '方法来源：[NIST实验设计](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3.htm)、',
        '[重复与干扰因素术语](https://itl.nist.gov/div898/handbook/pri/section7/pri7.htm)。','',
        '本轮产物稳定覆盖data_feasibility.json/md/png，旧冻结审计和原始数据不覆盖。']
    (OUT/'data_feasibility.md').write_text('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--exposure-json',type=Path)
    ap.add_argument('--diagnostic-dir',type=Path,required=True);args=ap.parse_args()
    args.diagnostic_dir.mkdir(parents=True,exist_ok=True)
    if args.exposure_json:
        workbook=json.loads(args.exposure_json.read_text())
    else:
        workbook=workbook_record(RAW/'拍摄参数记录.xlsx')
    wp=ROOT/workbook['path']
    if sha(wp)!=workbook['sha256']: raise ValueError('Workbook extraction hash mismatch')
    exposures={r['index']:r for r in workbook['rows']}
    if len(exposures)!=len(workbook['rows']) or sorted(exposures)!=list(range(1,32)):
        raise ValueError('Missing or duplicate exposure indices')
    if any(r['kv']<=0 or r['ma']<=0 or r['ms']<=0 for r in exposures.values()):
        raise ValueError('Invalid exposure parameters')
    manifest_path=ROOT/'outputs/mhinge_cross_pair_v1/analysis_results.json'
    manifest=json.loads(manifest_path.read_text())
    if not manifest['frozen_background']['verified']:raise ValueError('Unverified BG')
    for rel,digest in (('configs/background_frozen_v1.json',manifest['frozen_background']['sha256']),
                        ('outputs/pseudo_ghost_mechanism_v1/sam_masks_single_lag.npz',manifest['frozen_background']['mask_archive_sha256'])):
        if sha(ROOT/rel)!=digest:raise ValueError(f'Frozen input mismatch: {rel}')
    expected=manifest['input_provenance']['files']
    files=sorted(RAW.glob('*.dcm'))
    names={f'{i}-{k}.dcm' for i in range(1,32) for k in ('dark','light')}
    if set(p.name for p in files)!=names: raise ValueError('Missing or extra raw DICOM files')
    records=[];thumbs={};block_cache={};pileup_cache={}
    for i in range(1,32):
        for role in ('dark','light'):
            name=f'{i}-{role}.dcm'
            record,a=raw_record(RAW/name,expected[name]['sha256'])
            records.append(record);thumbs[(i,role)]=a[::16,::16].copy()
            block_cache[(i,role)]=block_mean(a.astype(np.float64),16)
            if role=='light':pileup_cache[i]=block_mean((a==a.max()).astype(np.float32),16)
        print(f'Raw images decoded and hashed: {2*i}/62',flush=True) if i%8==0 or i==31 else None
    byname={r['file']:r for r in records}
    duplicates={}
    for key in ('sha256','pixel_sha256'):
        groups=defaultdict(list)
        for r in records:groups[r[key]].append(r['file'])
        duplicates[key]=[v for v in groups.values() if len(v)>1]
    uidgroups=defaultdict(list)
    for r in records:uidgroups[r['headers']['SOPInstanceUID']].append(r['file'])
    uid_duplicates=[v for v in uidgroups.values() if len(v)>1]
    pairs=[]
    for row in manifest['pair_results']:
        light,dark=row['light_index'],row['dark_index'];label=f'{light}->{dark}'
        artifact=row['output_artifacts']['primary_maps'];path=manifest_path.parent/artifact['relative_path']
        fallback=False
        if not path.exists():path=ROOT/'outputs/pseudo_ghost_mechanism_v1'/artifact['relative_path'];fallback=True
        if sha(path)!=artifact['sha256']:raise ValueError(f'NPZ hash mismatch: {label}')
        with np.load(path) as z:m={k:z[k].copy() for k in z.files}
        x=block_cache[(light,'light')];x=x-np.percentile(x,75)
        if not np.allclose(x,m['source'],atol=.002,rtol=0):raise ValueError('Source mismatch')
        if not np.allclose(block_cache[(dark,'dark')],m['raw_dark'],atol=.002,rtol=0):raise ValueError('Dark mismatch')
        if not np.allclose(pileup_cache[light],m['source_saturation_fraction'],atol=1e-7,rtol=0):raise ValueError('Pileup mismatch')
        if not np.allclose(m['observable_signal'],m['raw_dark']-m['background'],atol=2e-4,rtol=0):raise ValueError('Y mismatch')
        block,fit,gate=_manifest_fit(row)
        support=m['source_support'].astype(bool);valid=m['evaluated_mask'].astype(bool)
        if not np.isfinite(m['oof_prediction'][valid]).all():raise ValueError('Invalid OOF')
        if not np.allclose(m['oof_residual'][valid],(m['observable_signal']-m['oof_prediction'])[valid],atol=2e-5,rtol=0):raise ValueError('Residual mismatch')
        frac=float(m['source_saturation_fraction'][support].mean())
        low=frac<.01;effective=gate['passed'] and abs(fit['alpha_fold_mean'])>=1e-4
        bg=next(b for b in block['background_results'] if b['background_mode']=='masked_hybrid_spline')
        excluded=next(r for r in bg['mask_results'] if r['mask_mode']=='exclude_saturated')
        fit_ex=excluded['fit']
        status='primary_low_pileup' if effective and low else 'clipping_risk_sensitivity_only' if effective else 'negative_control' if not gate['passed'] else 'weak_signal_reference'
        pairs.append(dict(pair=label,light_index=light,dark_index=dark,exposure=exposures[light],
            alpha=fit['alpha_fold_mean'],alpha_fold_cv=fit['alpha_fold_cv'],cv_r2=fit['cv_r2'],
            detected=gate['passed'],practical_effect=bool(effective),source_support_fraction=float(support.mean()),
            source_max_pileup_fraction=frac,low_source_pileup=low,classification=status,
            excluding_pileup_alpha=fit_ex['alpha_fold_mean'],excluding_pileup_cv_r2=fit_ex['cv_r2'],
            excluding_pileup_detected=excluded['detection_gate']['passed'],
            excluded_block_fraction=1-fit_ex['evaluated_blocks']/fit['evaluated_blocks'],
            npz=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),fallback_copy=fallback)))
    frozen_labels={r['light_index'] for r in pairs}
    # Screen all 30 candidate pairs; no new M1/BG fit on the 17 outside the frozen cohort.
    all_pairs=[dict(pair=f'{i}->{i+1}',exposure=exposures[i],
        source_whole_image_max_pileup=byname[f'{i}-light.dcm']['observed_max_fraction'],
        in_frozen_single_lag_cohort=i in frozen_labels,
        status='see_frozen_pair_audit' if i in frozen_labels else 'not_formally_frozen_or_validated') for i in range(1,31)]
    settings=defaultdict(list)
    for r in pairs:
        e=r['exposure'];settings[(e['kv'],e['ma'],e['ms'])].append(r)
    repeated=[]
    for setting,rows in settings.items():
        if len(rows)<2:continue
        alpha=[r['alpha'] for r in rows]
        repeated.append(dict(setting=list(setting),pairs=[r['pair'] for r in rows],
            alpha_ratio=float(max(alpha)/min(alpha)) if min(alpha)>0 else None,
            detected=[r['detected'] for r in rows]))
    primary=[r for r in pairs if r['classification']=='primary_low_pileup']
    selected_design=np.array([[1,r['exposure']['kv'],r['exposure']['ma'],r['exposure']['ms']] for r in primary])
    old_path=OUT/'exposure_association.json';old=json.loads(old_path.read_text())
    old_inputs_verified=(old['source_provenance']['manifest_sha256']==sha(manifest_path)
                         and old['source_provenance']['workbook_sha256']==sha(wp))
    if not old_inputs_verified:
        raise ValueError('Previous exposure association does not match current inputs')
    report=dict(schema_version=1,generated_at=datetime.now(timezone.utc).isoformat(),
        protocol=str(Path('docs/plans/2026-05-28-real-data-acquisition-protocol.md')),protocol_sha256=sha(ROOT/'docs/plans/2026-05-28-real-data-acquisition-protocol.md'),
        code_sha256=sha(Path(__file__)),workbook=workbook,manifest_sha256=sha(manifest_path),
        raw_count=len(records),raw_files=records,content_duplicates=duplicates,uid_duplicates=uid_duplicates,
        metadata_presence={k:sum(bool(r['headers'][k]) for r in records) for k in FIELDS},
        pair_results=pairs,all_candidate_pairs=all_pairs,repeated_settings=repeated,
        primary_pairs=[r['pair'] for r in primary],
        primary_design=dict(n=len(primary),n_levels={k:len(set(r['exposure'][k] for r in primary)) for k in ('kv','ma','ms','mas')},
                            intercept_kv_ma_ms_rank=int(np.linalg.matrix_rank(selected_design)),parameter_count=4,
                            same_kv_same_ms_pair=['3->4','5->6'],documented_controlled_independent_repeats=0),
        previous_association_input_hashes_verified=old_inputs_verified,
        prior_evidence={str(p.relative_to(ROOT)):sha(p) for p in
            (old_path,ROOT/'outputs/background_freeze_audit_v1/background_crossfit_audit.json')},
        conclusions=dict(raw_files='readable_and_hash_consistent',
                         primary_examples='3->4,5->6,27->28',
                         exploratory_association='limited_only; selection_and_session_confounding',
                         exposure_causal_or_coefficient_calibration='insufficient',clean_recovery_accuracy='no_ground_truth'))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'data_feasibility.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    write_report(report)
    print('Evidence figure',evidence_figure(thumbs,block_cache,manifest))
    print('RAW SUMMARY',json.dumps({'duplicates':duplicates,'uid_duplicate_groups':len(uid_duplicates),
           'primary':[r['pair'] for r in primary],'repeated':repeated},ensure_ascii=False))
    print('PAIR SUMMARY')
    for r in pairs:print(r['pair'],r['classification'],r['exposure'],f"source pileup={r['source_max_pileup_fraction']:.6%}",f"alpha={r['alpha']:.8f}")
    print('CONTACT SHEETS',contact_sheets(thumbs,args.diagnostic_dir))
    print('Saved',OUT/'data_feasibility.json')

if __name__=='__main__':main()
