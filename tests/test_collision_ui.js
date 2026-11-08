const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(__dirname,'../src/traffic_edge/web_ui.py'),'utf8');
const handlers=['startCollisionReview','pollCollisionReview'].map(name=>source.split('\n').find(line=>line.startsWith(`async function ${name}(`))).join('\n');
function fixture(){const controls={collisionStart:{value:'0'},collisionEnd:{value:'4'},collisionRunBtn:{disabled:false},collisionResults:{replaceChildren(){}},collisionProgress:{style:{}}};const context={annotationSourceVersion:1,selectedFile:{name:'first.mp4'},serverVideoPath:null,currentSample:null,groundTruthFinalized:false,collisionStatus:{textContent:''},collisionJobId:'first-job',Number,JSON,encodeURIComponent,document:{getElementById:id=>controls[id]},setTimeout(){throw Error('A stale job must not schedule another poll')}};vm.createContext(context);vm.runInContext(handlers,context);return {context,controls}}
(async()=>{
  const {context}=fixture();let resolveUpload;let requests=0;
  context.uploadFile=()=>new Promise(resolve=>resolveUpload=resolve);context.fetch=()=>{requests++;throw Error('Must not start a stale review')};
  const work=context.startCollisionReview();context.annotationSourceVersion=2;context.serverVideoPath='data/new-source.mp4';resolveUpload('data/old-source.mp4');await work;
  assert.equal(context.serverVideoPath,'data/new-source.mp4');assert.equal(requests,0);
  const second=fixture();let resolveStatus;second.context.fetch=()=>new Promise(resolve=>resolveStatus=resolve);
  const poll=second.context.pollCollisionReview('first-job',1);second.context.annotationSourceVersion=2;second.context.collisionJobId='new-job';second.context.collisionStatus.textContent='New source';
  resolveStatus({ok:true,json:async()=>({kind:'collision_review',status:'complete',summary:{}})});await poll;
  assert.equal(second.context.collisionStatus.textContent,'New source');
  let navigate;const paused={collision:0,annotation:0,source:0};
  const media={document:{addEventListener(_event,callback){navigate=callback}},collisionPreview:{pause(){paused.collision++}},reviewVideo:{pause(){paused.annotation++}},preview:{pause(){paused.source++}}};
  vm.createContext(media);vm.runInContext(source.split('\n').find(line=>line.startsWith("document.addEventListener('workspacechange'")),media);
  navigate({detail:'collision-step'});assert.equal(paused.annotation,1,'Hidden annotation video must stop before it can count unseen review coverage');assert.equal(paused.collision,0);
  console.log('Collision UI checks passed: upload/source isolation, stale-job rejection and hidden-video playback.');
})().catch(error=>{console.error(error);process.exitCode=1});
