"""Train controlled PatchCore augmentation runs in Studio; select on validation only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import httpx


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def capture_scores(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['category'],r['capture'])].append(r['score'])
    return [{'category':kind,'capture':capture,'score':sum(values)/len(values)}for(kind,capture),values in groups.items()]


def metrics(rows, threshold):
    result={}
    for kind in ('good','bad','circle'):
        scores=[r['score'] for r in rows if r['category']==kind]
        positives=sum(s>=threshold for s in scores)
        correct=len(scores)-positives if kind=='good' else positives
        result[kind]={'captures':len(scores),'flagged_defective':positives,'correct':correct,'accuracy':correct/len(scores)}
    return result


def balanced(result):
    return (result['good']['accuracy']+(result['bad']['accuracy']+result['circle']['accuracy'])/2)/2


def score_diagnostics(rows):
    return {kind: {'images': len(scores), 'at_upper_bound': sum(s >= .99999 for s in scores),
                   'minimum': min(scores), 'maximum': max(scores)}
            for kind in ('good', 'bad', 'circle')
            if (scores := [r['score'] for r in rows if r['category'] == kind])}


def calibrate(rows):
    scores=sorted({r['score'] for r in rows})
    if len(scores)<2:raise ValueError('All validation scores are identical; investigate inference before reporting accuracy')
    candidates=[scores[0]-1e-8]+[(a+b)/2 for a,b in zip(scores,scores[1:])]+[scores[-1]+1e-8]
    def objective(t):
        m=metrics(rows,t)
        return balanced(m),m['good']['accuracy']
    threshold=max(candidates,key=objective)
    return threshold,metrics(rows,threshold)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--studio-url',default='http://127.0.0.1:8005')
    args=parser.parse_args();root=args.run_dir.resolve()
    client=httpx.Client(base_url=args.studio_url,timeout=120)
    for _ in range(120):
        try:client.get('/api/projects').raise_for_status();break
        except httpx.TransportError:time.sleep(.5)
    results={}
    def progress(**data):dump(root/'training_progress.json',data)
    def predict(directory, model, source, split):
        target=directory/f'{split}_predictions.json'
        done=json.loads(target.read_text()) if target.exists() else []
        seen={r['path'] for r in done}
        selected=[r for r in source['images'] if r['split']==split]
        for row in selected:
            if row['path'] in seen:continue
            path=Path(row['path'])
            with path.open('rb') as f:
                p=client.post(f'/api/projects/{model["project_id"]}/models/{model["id"]}:predict',params={'device':'CPU'},files={'file':(path.name,f,'image/jpeg')}).raise_for_status().json()
            done.append({**row,'score':p['score'],'studio_label':p['label']})
            dump(target,done)
            if len(done)%60==0:
                progress(stage='evaluating',profile=directory.name,split=split,images_done=len(done),images_total=len(selected))
                print(directory.name,split,len(done),'/',len(selected),flush=True)
        return done
    for profile in ('baseline','geometry','photometric','combined'):
        directory=root/profile;dataset_path=directory/'studio_dataset.json';source=json.loads(dataset_path.read_text())
        progress(stage='importing',profile=profile)
        project_path=directory/'project.json'
        if project_path.exists():project=json.loads(project_path.read_text())
        else:
            project=client.post('/api/projects',json={'name':f'PatchCore augmentation — {profile} — {root.name}'}).raise_for_status().json()
            dump(project_path,project)
        source['project']=project
        training=[r for r in source['images'] if r['split']=='train']
        for i,row in enumerate(training):
            if row.get('studio_media_id'):continue
            path=Path(row['path'])
            with path.open('rb') as f:
                media=client.post(f'/api/projects/{project["id"]}/images',files={'file':(path.name,f,'image/jpeg')}).raise_for_status().json()
            row['studio_media_id']=media['id'];dump(dataset_path,source)
            if (i+1)%150==0:print(profile,'imported',i+1,'/',len(training),flush=True)
        job_path=directory/'studio_job.json'
        ratio=.1*207/len(training)
        if job_path.exists():job_id=json.loads(job_path.read_text())['job_id']
        else:
            payload={'project_id':project['id'],'model_name':'patchcore','device':{'type':'cpu'},'max_epochs':1,'training_seed':2026,'coreset_sampling_ratio':ratio}
            job=client.post('/api/jobs:train',json=payload).raise_for_status().json();dump(job_path,job);dump(directory/'training_config.json',payload);job_id=job['job_id']
        print(profile,'training',job_id,'coreset ratio',ratio,flush=True)
        for _ in range(1800):
            job=next(j for j in client.get('/api/jobs').raise_for_status().json()['jobs'] if j['id']==job_id)
            progress(stage='training',profile=profile,job=job)
            if job['status']=='completed':break
            if job['status'] in ('failed','canceled'):raise RuntimeError(job)
            time.sleep(2)
        else:raise TimeoutError(f'{profile} training still running')
        models=client.get(f'/api/projects/{project["id"]}/models').raise_for_status().json()['models']
        model=next(m for m in models if m['train_job_id']==job_id);dump(directory/'studio_model.json',model)
        validation=predict(directory,model,source,'validation')
        threshold,m=calibrate(capture_scores(validation))
        report={'model_id':model['id'],'job_id':job_id,'profile':profile,'aggregation':'mean of three frame scores per capture','validation_selected_threshold':threshold,'validation':m,'validation_balanced_accuracy':balanced(m),'test':None,'training_images':len(training),'coreset_sampling_ratio':ratio,'training_seed':2026,'limitations':['Physical brick identities unknown; split is by capture.','Studio internal synthetic metrics are not used for selection.','External threshold does not modify Studio default.','Memory bank budget is approximately constant across experiments.']}
        report['validation_score_diagnostics'] = score_diagnostics(validation)
        dump(directory/'evaluation.json',report);results[profile]=report;dump(root/'comparison.json',results)
        print(profile,'VALIDATION',json.dumps(m),flush=True)
    best=max(results,key=lambda k:(results[k]['validation_balanced_accuracy'],results[k]['validation']['good']['accuracy']))
    print('Selected on validation:',best,flush=True)
    # Only the selected model and seed-controlled baseline see the real test set.
    for profile in dict.fromkeys(('baseline',best)):
        directory=root/profile;source=json.loads((directory/'studio_dataset.json').read_text());model=json.loads((directory/'studio_model.json').read_text())
        predictions=predict(directory,model,source,'test')
        report=results[profile];report['test']=metrics(capture_scores(predictions),report['validation_selected_threshold']);report['test_balanced_accuracy']=balanced(report['test'])
        dump(directory/'evaluation.json',report)
    summary={'selected_profile':best,'selection_basis':'validation only; test evaluated after selection','profiles':results,'reference_patchcore':'../cleaned-20261114T024821/patchcore-20261114T025755','status':'complete'}
    dump(root/'comparison.json',summary);progress(stage='complete',selected_profile=best)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
