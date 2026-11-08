"""Study register controls embedded in the existing local workbench."""

STUDY_HTML = r'''
<section class="study-panel" aria-label="Reviewed dataset">
  <h2>Reviewed dataset</h2>
  <p>Finish the original-video review in Ground truth, configure its lanes, then register the current recording. Group clips from the same physical source together.</p>
  <div class="settings">
    <label for="studyCaseName">Case name</label><input id="studyCaseName" maxlength="120"/>
    <label for="studySourceGroup">Source group</label><input id="studySourceGroup" maxlength="120" placeholder="One source or related cameras"/>
    <label for="studyCamera">Physical camera ID</label><input id="studyCamera" maxlength="120"/>
    <label for="studyCaseSplit">Dataset use</label><select id="studyCaseSplit"><option value="development">Development</option><option value="validation">Validation</option><option value="test">Test</option></select>
  </div>
  <div class="row"><button id="studyRegisterBtn">Register current reviewed video</button><button id="studyRefreshBtn">Refresh dataset and results</button></div>
  <p id="studyRegisterStatus" role="status" aria-live="polite"></p>
  <div id="studyCases" class="study-table"><p>No reviewed cases registered yet.</p></div>
  <h2>Batch evaluation</h2>
  <p>Choose one detector and tracker before starting. Each batch evaluates every registered recording in the chosen split with frozen geometry and labels.</p>
  <div class="settings">
    <label for="studyRunSplit">Evaluate split</label><select id="studyRunSplit"><option value="validation">Validation</option><option value="test">Test</option><option value="development">Development</option></select>
    <label for="studyModel">Detector</label><select id="studyModel"><option value="yolo11n">YOLO11n</option><option value="yolov8n">YOLOv8n</option></select>
    <label for="studyTracker">Tracker</label><select id="studyTracker"><option value="bytetrack">ByteTrack</option><option value="botsort">BoT-SORT</option></select>
    <label for="studyDevice">Compute device</label><select id="studyDevice" class="compute-device"><option value="auto">Automatic (NVIDIA when available)</option><option value="cpu">CPU</option><option value="cuda">NVIDIA CUDA</option></select>
  </div>
  <p class="small-note compute-status">Checking compute hardware...</p>
  <div class="row"><button id="studyRunBtn" class="primary">Run reviewed batch</button><span id="studyStatus" role="status" aria-live="polite">Ready</span></div>
  <label for="studyRunHistory">Completed batches</label><select id="studyRunHistory"><option value="">Select a completed batch</option></select>
  <div id="studyReport"></div>
</section>
'''

STUDY_STYLE = r'''
.study-panel{border-top:1px solid var(--line,#ccc);margin-top:24px;padding-top:20px}
.study-panel h2{margin:24px 0 10px}.study-panel select,.study-panel input{max-width:100%}
.study-table{overflow-x:auto;margin:16px 0}.study-table table{width:100%;border-collapse:collapse}
.study-table th,.study-table td{padding:9px 10px;text-align:left;border-bottom:1px solid #ccd0c5;vertical-align:top}
.study-error{border-top:1px solid #ccd0c5;padding:16px 0}.study-error img,.study-error video{width:100%;max-width:420px;margin:8px 12px 8px 0;vertical-align:top}
.study-panel .row{margin:12px 0}.study-downloads{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}
'''

STUDY_SCRIPT = r'''
let studyRevision=0,studyLoadVersion=0,studyReportVersion=0,studyActiveJob=null,studyLastSource=null;
const studyElement=id=>document.getElementById(id);
async function studyRequest(url,payload){
  const options=payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)};
  const response=await fetch(url,options),data=await response.json();
  if(!response.ok)throw Error(data.error||'Study request failed');
  return data;
}
async function loadStudyRegister(){
  const version=++studyLoadVersion;
  try{
    const data=await studyRequest('/api/study/cases');
    if(version!==studyLoadVersion)return;
    studyRevision=data.revision;
    const host=studyElement('studyCases');
    if(!data.cases.length)host.innerHTML='<p>No reviewed cases registered yet.</p>';
    else host.innerHTML='<table><thead><tr><th>Case</th><th>Use</th><th>Source / camera</th><th>Labels</th><th>Input check</th><th></th></tr></thead><tbody>'+data.cases.map(c=>
      `<tr><td>${escapeHtml(c.name)}</td><td>${escapeHtml(c.split)}</td><td>${escapeHtml(c.source_group_id)} / ${escapeHtml(c.camera_id)}</td><td>${c.negative_video?'Reviewed negative':Number(c.label_count)}</td><td>${escapeHtml(c.issues.length?c.issues.join(' '):'Ready')}</td><td><button data-study-remove="${escapeHtml(c.id)}">Remove</button></td></tr>`).join('')+'</tbody></table>';
    host.querySelectorAll('[data-study-remove]').forEach(button=>button.onclick=async()=>{
      if(!confirm('Remove this case from the dataset register? Its recordings, labels and past results remain saved.'))return;
      button.disabled=true;
      try{await studyRequest('/api/study/remove',{case_id:button.dataset.studyRemove,revision:studyRevision});await loadStudyRegister()}
      catch(error){studyElement('studyRegisterStatus').textContent=error.message;button.disabled=false}
    });
    const history=studyElement('studyRunHistory'),selected=history.value;
    history.replaceChildren(new Option('Select a completed batch',''));
    for(const run of data.runs)history.add(new Option(`${run.run_name} · ${run.split} · ${run.case_count} recordings`,run.run_name));
    if(data.runs.some(run=>run.run_name===selected))history.value=selected;
  }catch(error){if(version===studyLoadVersion)studyElement('studyRegisterStatus').textContent=error.message}
}
async function registerStudyCase(){
  const button=studyElement('studyRegisterBtn'),status=studyElement('studyRegisterStatus');
  button.disabled=true;
  try{
    if(!serverVideoPath||!groundTruthFinalized)throw Error('Load the recording and finalize its original-video review in Ground truth first.');
    const profile=cameraProfilePayload();
    await studyRequest('/api/study/cases',{
      video_path:serverVideoPath,profile,revision:studyRevision,
      name:studyElement('studyCaseName').value,source_group_id:studyElement('studySourceGroup').value,
      camera_id:studyElement('studyCamera').value,split:studyElement('studyCaseSplit').value
    });
    status.textContent='Reviewed labels and camera geometry registered. Existing completed batches keep their original inputs.';
    await loadStudyRegister();
  }catch(error){status.textContent=error.message}
  finally{button.disabled=false}
}
function studyMetric(score,key){
  const tp=Number(score.true_positives),fp=Number(score.false_positives),fn=Number(score.false_negatives);
  const denominator=key==='precision'?tp+fp:key==='recall'?tp+fn:2*tp+fp+fn;
  return denominator?Number(score[key]).toFixed(3):'—';
}
function studyAsset(run,name){return '/api/study/asset?run='+encodeURIComponent(run)+'&name='+encodeURIComponent(name)}
function renderStudyReport(report){
  const host=studyElement('studyReport'),metrics=report.metrics,score=metrics.micro_average;
  const title=document.createElement('h2');title.textContent=report.split+' results';
  const info=document.createElement('p');
  info.textContent=`${report.cases.length} recordings · ${metrics.dataset.unique_source_groups} source groups · ${report.normal_video_count} reviewed negative recordings · ${report.model} + ${report.tracker} · ${report.compute_device}`;
  const totals=document.createElement('p');
  totals.textContent=`TP ${score.true_positives} · FP ${score.false_positives} · FN ${score.false_negatives} · Precision ${studyMetric(score,'precision')} · Recall ${studyMetric(score,'recall')} · F1 ${studyMetric(score,'f1')} · FP/hour ${metrics.false_positives_per_hour??'—'}`;
  const note=document.createElement('p');note.className='small-note';
  note.textContent='Metrics describe this split and frozen setup. A dash marks an undefined denominator. Development and validation results stay in their own batches.';
  host.replaceChildren(title,info,totals,note);
  if(metrics.confidence_intervals_95){
    const interval=document.createElement('p');interval.className='small-note';
    interval.textContent='95% source-group bootstrap intervals: '+Object.entries(metrics.confidence_intervals_95).map(([key,value])=>key+' '+value.join('–')).join(' · ');
    host.appendChild(interval);
  }
  const downloads=document.createElement('div');downloads.className='study-downloads';
  for(const [name,label] of [['report.json','Download report JSON'],['cases.csv','Download case metrics CSV'],['errors.csv','Download errors CSV']]){
    const link=document.createElement('a');link.href=studyAsset(report.run_name,name);link.download=name;link.textContent=label;downloads.appendChild(link);
  }
  host.appendChild(downloads);
  const table=document.createElement('div');table.className='study-table';
  table.innerHTML='<table><thead><tr><th>Case</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>'+report.cases.map(c=>
    `<tr><td>${escapeHtml(c.name)}</td><td>${c.true_positives}</td><td>${c.false_positives}</td><td>${c.false_negatives}</td><td>${studyMetric(c,'precision')}</td><td>${studyMetric(c,'recall')}</td><td>${studyMetric(c,'f1')}</td></tr>`).join('')+'</tbody></table>';
  host.appendChild(table);
  const rules=document.createElement('div');rules.className='study-table';
  rules.innerHTML='<table><thead><tr><th>Rule</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>'+Object.entries(metrics.by_type).map(([type,c])=>
    `<tr><td>${escapeHtml(eventNames[type]||type)}</td><td>${c.true_positives}</td><td>${c.false_positives}</td><td>${c.false_negatives}</td><td>${studyMetric(c,'precision')}</td><td>${studyMetric(c,'recall')}</td><td>${studyMetric(c,'f1')}</td></tr>`).join('')+'</tbody></table>';
  host.appendChild(rules);
  const heading=document.createElement('h3');heading.textContent='False positives and missed events';host.appendChild(heading);
  if(!report.errors.length){const empty=document.createElement('p');empty.textContent='No unmatched predictions or labels in this batch.';host.appendChild(empty)}
  for(const error of report.errors){
    const card=document.createElement('article');card.className='study-error';
    const label=document.createElement('p');label.textContent=`${error.kind==='FP'?'False positive':'Missed event'} · ${error.name} · ${Number(error.timestamp_seconds).toFixed(3)} s · ${eventNames[error.candidate_event]||error.candidate_event} · ${error.lane_id||'lane unspecified'}`;
    card.appendChild(label);
    if(error.frame){const image=document.createElement('img');image.src=studyAsset(report.run_name,error.frame.split('/').pop());image.alt='Original source frame at the unmatched event time';image.loading='lazy';card.appendChild(image)}
    if(error.crop){const image=document.createElement('img');image.src=studyAsset(report.run_name,error.crop.split('/').pop());image.alt='Selected vehicle at the unmatched event time';image.loading='lazy';card.appendChild(image)}
    if(error.clip){const video=document.createElement('video');video.controls=true;video.preload='none';video.src=studyAsset(report.run_name,error.clip.split('/').pop());card.appendChild(video)}
    const link=document.createElement('a');link.href='/api/study/source?run='+encodeURIComponent(report.run_name)+'&case='+encodeURIComponent(error.case_id)+'#t='+Number(error.timestamp_seconds);link.target='_blank';link.rel='noopener';link.textContent='Open original recording at this time';card.appendChild(link);
    host.appendChild(card);
  }
}
async function loadStudyReport(run){
  if(!run)return;
  const version=++studyReportVersion;
  try{const data=await studyRequest('/api/study/report?run='+encodeURIComponent(run));if(version===studyReportVersion)renderStudyReport(data)}
  catch(error){if(version===studyReportVersion)studyElement('studyStatus').textContent=error.message}
}
async function startStudyBatch(){
  const button=studyElement('studyRunBtn');button.disabled=true;
  try{
    const data=await studyRequest('/api/study/run',{split:studyElement('studyRunSplit').value,model:studyElement('studyModel').value,tracker:studyElement('studyTracker').value,device:studyElement('studyDevice').value});
    studyActiveJob=data.job_id;studyElement('studyStatus').textContent='Queued';pollStudyBatch(data.job_id);
  }catch(error){studyElement('studyStatus').textContent=error.message;button.disabled=false}
}
async function pollStudyBatch(id){
  if(studyActiveJob!==id)return;
  try{
    const job=await studyRequest('/api/status?id='+encodeURIComponent(id));
    if(studyActiveJob!==id)return;
    if(job.kind!=='study'||job.status==='error')throw Error(job.error||'Could not read study progress');
    studyElement('studyStatus').textContent=job.status==='queued'?'Queued behind the current analysis':`${job.completed_cases}/${job.total_cases} recordings · ${job.stage}`;
    if(job.status!=='complete'){setTimeout(()=>pollStudyBatch(id),1000);return}
    studyElement('studyRunBtn').disabled=false;studyActiveJob=null;
    await loadStudyRegister();studyElement('studyRunHistory').value=job.run_name;await loadStudyReport(job.run_name);
  }catch(error){if(studyActiveJob===id){studyElement('studyStatus').textContent=error.message;studyElement('studyRunBtn').disabled=false;studyActiveJob=null}}
}
studyElement('studyRegisterBtn').onclick=registerStudyCase;
studyElement('studyRefreshBtn').onclick=loadStudyRegister;
studyElement('studyRunBtn').onclick=startStudyBatch;
studyElement('studyRunHistory').onchange=event=>loadStudyReport(event.target.value);
document.addEventListener('workspacechange',event=>{
  if(event.detail!=='readiness-step')return;
  for(const [target,value] of [['studyCaseName',fileName.textContent],['studySourceGroup',studyElement('cameraId').value],['studyCamera',studyElement('cameraId').value]]){
    if(studyLastSource!==serverVideoPath||!studyElement(target).value)studyElement(target).value=value;
  }
  studyLastSource=serverVideoPath;
  loadStudyRegister();
});
if(location.hash==='#readiness-step')loadStudyRegister();
'''
