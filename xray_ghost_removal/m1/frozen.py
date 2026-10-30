"""Read and verify existing M1 maps and fold parameters without refitting."""
import numpy as np
from xray_ghost_removal.common.paths import PROJECT_ROOT as ROOT
from xray_ghost_removal.common.provenance import sha256 as sha

def load_m1_pair(manifest, light, dark):
    row = next(r for r in manifest['pair_results'] if (r['light_index'], r['dark_index']) == (light, dark))
    artifact = row['output_artifacts']['primary_maps']
    path = ROOT/'outputs/mhinge_cross_pair_v1'/artifact['relative_path']
    if not path.exists():
        path = ROOT/'outputs/pseudo_ghost_mechanism_v1'/artifact['relative_path']
    if sha(path) != artifact['sha256']:
        raise ValueError(f'Frozen map hash mismatch: {path}')
    with np.load(path) as z:
        maps = {k: z[k].copy() for k in z.files}
    if str(maps['background_mode']) != 'masked_hybrid_spline':
        raise ValueError('Wrong frozen background mode')
    bg = next(r for r in row['block_results'] if r['block_size'] == 16)
    bg = next(r for r in bg['background_results'] if r['background_mode'] == 'masked_hybrid_spline')
    result = next(r for r in bg['mask_results'] if r['mask_mode'] == 'exclude_saturated')
    valid = maps['evaluated_mask'].astype(bool)
    pred = maps['oof_prediction']
    if not result['detection_gate']['passed'] or abs(result['fit']['alpha_fold_mean']) < 1e-4:
        raise ValueError('Pair does not pass the frozen practical-effect gate')
    if not np.allclose(maps['observable_signal'], maps['raw_dark']-maps['background'], atol=2e-4, rtol=0):
        raise ValueError('Frozen Y identity failed')
    if not np.allclose(maps['oof_residual'][valid], (maps['observable_signal']-pred)[valid], atol=2e-5, rtol=0):
        raise ValueError('Frozen residual identity failed')
    # Verify the stored prediction against the recorded fold-specific model.
    for params in result['fit']['fold_parameters']:
        mask = valid & (maps['spatial_folds'] == params['fold'])
        expected = params['alpha']*maps['source'][mask]+params['intercept']
        if not np.allclose(pred[mask], expected, atol=2e-5, rtol=1e-6):
            raise ValueError('Frozen OOF prediction does not match model parameters')
    return maps, result['fit'], dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
