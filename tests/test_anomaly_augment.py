import hashlib
import json

import cv2
import numpy as np

from dlsj.inspect.augment import augment_views, generate


def test_geometric_draw_is_shared_reproducible_and_does_not_modify_sources():
    image=np.zeros((320,320,3),np.uint8)
    image[10:45,10:45]=(255,0,0)
    image[-45:-10,-45:-10]=(0,255,0)
    original=image.copy()
    a,_=augment_views([image]*3,'geometry',2026)
    b,_=augment_views([image]*3,'geometry',2026)
    assert np.array_equal(image,original)
    assert all(np.array_equal(a[0],v) for v in a+b)
    assert np.any((a[0][:,:,0]>180)&(a[0][:,:,1]<50))
    assert np.any((a[0][:,:,1]>180)&(a[0][:,:,0]<50))


def test_generation_keeps_all_derivatives_in_train_and_heldout_untouched(tmp_path):
    rows=[]
    for split,kind in [('train','good'),('validation','good'),('test','bad')]:
        for i in range(3):
            p=tmp_path/f'{split}_f{i}.jpg'
            cv2.imwrite(str(p),np.full((320,320,3),110+i,np.uint8))
            rows.append({'path':str(p),'capture':split,'split':split,'category':kind,
                         'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    source=tmp_path/'source.json';source.write_text(json.dumps({'images':rows}))
    result=generate(source,tmp_path/'out',variants=1)
    for profile in result['profiles']:
        data=json.loads((tmp_path/'out'/profile/'studio_dataset.json').read_text())['images']
        assert [r for r in data if r['split']!='train']==[r for r in rows if r['split']!='train']
        augmented=[r for r in data if 'augmented_from' in r]
        assert len(augmented)==(0 if profile=='baseline' else 3)
        assert all(r['split']=='train' and r['category']=='good' and r['capture']=='train' for r in augmented)
