"""Generate reproducible, split-safe Albumentations datasets for brick inspection."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np

PROFILES = ('geometry', 'photometric', 'combined')


def seed_for(seed, capture, variant):
    return int.from_bytes(hashlib.sha256(f'{seed}:{capture}:{variant}'.encode()).digest()[:4], 'big')


def pipelines(profile, seed, fill):
    common = []
    if profile in ('geometry', 'combined'):
        # SafeRotate retains corners. Padding protects them during the final scale/shift.
        common += [A.SafeRotate(limit=(-15, 15), border_mode=cv2.BORDER_CONSTANT, fill=fill, p=1),
                   A.PadIfNeeded(min_height=352, min_width=352, border_mode=cv2.BORDER_CONSTANT, fill=fill, p=1),
                   A.Affine(scale=(.95, 1.05), translate_percent=(-.02, .02), keep_ratio=True,
                            border_mode=cv2.BORDER_CONSTANT, fill=fill, p=1), A.Resize(320, 320)]
    if profile in ('photometric', 'combined'):
        common += [A.RandomBrightnessContrast(brightness_limit=.15, contrast_limit=.15, p=.75),
                   A.RandomGamma(gamma_limit=(85, 115), p=.35),
                   A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=6, val_shift_limit=5, p=.3)]
    shared = A.ReplayCompose(common, additional_targets={'view1': 'image', 'view2': 'image'})
    shared.set_random_seed(seed)
    return shared


def noise_pipeline(seed):
    return A.Compose([
        A.OneOf([A.GaussNoise(std_range=(1/255, 4/255), mean_range=(0, 0), p=1),
                 A.ISONoise(color_shift=(.001, .006), intensity=(.01, .04), p=1),
                 A.MultiplicativeNoise(multiplier=(.97, 1.03), per_channel=False, elementwise=True, p=1)], p=.7),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(.1, .6), p=1),
                 A.MotionBlur(blur_limit=(3, 3), allow_shifted=False, p=1)], p=.2),
        A.ImageCompression(quality_range=(85, 100), p=.4),
    ], seed=seed)


def augment_views(images, profile, seed):
    if len(images) != 3 or any(im.shape != (320, 320, 3) for im in images):
        raise ValueError('Expected exactly three RGB crops of size 320 × 320')
    edge = np.concatenate([images[0][0], images[0][-1], images[0][:, 0], images[0][:, -1]])
    fill = tuple(int(x) for x in np.median(edge, axis=0))
    shared = pipelines(profile, seed, fill)
    result = shared(image=images[0], view1=images[1], view2=images[2])
    output = [result[k] for k in ('image', 'view1', 'view2')]
    noise_seeds = [seed_for(seed, 'sensor-noise', i) for i in range(3)]
    if profile in ('photometric', 'combined'):
        output = [noise_pipeline(s)(image=im)['image'] for s, im in zip(noise_seeds, output)]
    return output, {'common_replay': result['replay'], 'noise_seeds': noise_seeds,
                    'noise_applied': profile in ('photometric', 'combined')}


def json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def generate(source_manifest, out, variants=4, seed=2026):
    source = json.loads(Path(source_manifest).read_text())
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a new output directory; existing results are preserved')
    out.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    rows = [{k: v for k, v in r.items() if k != 'studio_media_id'} for r in source['images']]
    for row in rows:
        if row['split'] == 'train':
            if row['category'] != 'good': raise ValueError('Anomaly training must contain only plain bricks')
            groups[row['capture']].append(row)
    manifest = {'source_manifest': str(Path(source_manifest).resolve()), 'seed': seed,
                'albumentations_version': A.__version__, 'variants_per_image': variants,
                'geometry_note': 'Safe rotation plus padded affine retains the complete original crop; final resize slightly reduces brick scale.',
                'profiles': {}}
    previews = []
    for profile in ('baseline',) + PROFILES:
        directory = out / profile
        directory.mkdir()
        profile_rows = [dict(r) for r in rows]
        if profile != 'baseline':
            for group_index, (capture, frames) in enumerate(sorted(groups.items())):
                frames = sorted(frames, key=lambda r: r['path'])
                if len(frames) != 3: raise ValueError(f'Incomplete capture: {capture}')
                images = [cv2.cvtColor(cv2.imread(r['path']), cv2.COLOR_BGR2RGB) for r in frames]
                for variant in range(1, variants + 1):
                    draw_seed = seed_for(seed, capture, variant)
                    augmented, params = augment_views(images, profile, draw_seed)
                    parent = directory / 'data' / 'good'
                    parent.mkdir(parents=True, exist_ok=True)
                    key = f'{capture}_aug{variant}'
                    for i, (original, image) in enumerate(zip(frames, augmented)):
                        path = parent / f'{key}_f{i}.jpg'
                        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            raise OSError(f'Could not save {path}')
                        profile_rows.append({**original, 'path': str(path), 'source_path': original['path'],
                                             'source_sha256': original['sha256'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                             'augmented_from': capture, 'augmentation_variant': variant,
                                             'augmentation_seed': draw_seed, 'profile': profile})
                    (parent / f'{key}.json').write_text(json.dumps(params, indent=2, default=json_default)+'\n')
                    if group_index < 2 and variant <= 2: previews.append((profile, images[0], augmented[0]))
                (out/'progress.json').write_text(json.dumps({'stage':'generating', 'profile':profile,
                    'captures_done':group_index+1, 'captures_total':len(groups)},indent=2)+'\n')
        dataset = {**source, 'images': profile_rows, 'augmentation_profile': profile,
                   'source_manifest': str(Path(source_manifest).resolve()), 'albumentations_version': A.__version__}
        (directory/'studio_dataset.json').write_text(json.dumps(dataset,indent=2)+'\n')
        manifest['profiles'][profile] = {'directory':str(directory), 'training_images':sum(r['split']=='train' for r in profile_rows),
                                         'heldout_images':sum(r['split']!='train' for r in profile_rows)}
        print(profile, manifest['profiles'][profile],flush=True)
    # Contact sheet: each row shows original next to augmented input.
    sheet = np.full((len(previews)*200,400,3),245,np.uint8)
    for i,(profile,original,augmented) in enumerate(previews):
        for col,image in enumerate((original,augmented)):
            sheet[i*200:i*200+180,col*200:(col+1)*200] = cv2.resize(cv2.cvtColor(image,cv2.COLOR_RGB2BGR),(200,180))
        cv2.putText(sheet,profile,(205,i*200+196),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
    cv2.imwrite(str(out/'augmentation_preview.jpg'),sheet)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'progress.json').write_text(json.dumps({'stage':'generation_complete','profiles':manifest['profiles']},indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--variants',type=int,default=4)
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args()
    if args.variants < 1: parser.error('--variants must be positive')
    generate(args.source_manifest,args.out,args.variants,args.seed)

if __name__=='__main__': main()
