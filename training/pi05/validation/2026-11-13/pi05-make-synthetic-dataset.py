"""Create test-only SO-101-shaped data, never robot demonstrations."""
import json
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path('/tmp/pi05-synthetic-dataset')
joints = [f'{name}.pos' for name in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']]
features = {f'observation.images.{name}': {'dtype':'video','shape':(480,640,3), 'names':['height','width','channels']} for name in ['wrist','top','inspect']}
features.update({name:{'dtype':'float32','shape':(6,), 'names':joints} for name in ['action','observation.state']})
ds = LeRobotDataset.create(repo_id='local/so101_synthetic_smoke', fps=30, root=root, features=features, robot_type='so101_follower', image_writer_threads=4)
labels=[]
rng=np.random.default_rng(42)
for episode in range(2):
    task='place the brick on the blue plate'
    for frame in range(64):
        image=np.zeros((480,640,3), dtype=np.uint8)
        image[100:200, frame*4:frame*4+80, episode]=180
        values={f'observation.images.{name}':image.copy() for name in ['wrist','top','inspect']}
        values.update({'observation.state':rng.uniform(-50,50,6).astype('float32'), 'action':rng.uniform(-50,50,6).astype('float32'),'task':task})
        ds.add_frame(values)
    ds.save_episode()
    labels.append({'episode_index':episode,'success':True,'n_frames':64,'task':task,'synthetic':True})
ds.finalize()
(root/'meta/dlsj').mkdir(parents=True,exist_ok=True)
(root/'meta/dlsj/labels.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in labels))
(root/'SYNTHETIC_TEST_DATA_ONLY').write_text('Generated for plumbing tests; not robot demonstrations.\n')
print(root)
