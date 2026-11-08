// Execute the actual UI handlers with a small DOM fixture. No duplicated guard logic.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname,'../src/traffic_edge/web_ui.py'),'utf8');
const wanted = ['sceneLaneKey','refreshSceneProposalButtons','applyMarkedLane','applyMarkedStopLine','currentLaneState','loadLaneState'];
const handlers = wanted.map(name=>{
  const line=source.split('\n').find(line=>line.startsWith(`function ${name}(`));
  assert.ok(line,`Missing production handler ${name}`);
  return line;
}).join('\n');
const controls = {};
const buttons=[0,1].map(index=>({dataset:{proposalLane:String(index)},disabled:false,title:''}));
const proposal=index=>({name:`Lane proposal ${index+1}`,method:'learned_marking_bounded_road_region',
  polygon:[[.1+index*.4,.2],[.3+index*.4,.2],[.3+index*.4,.8],[.1+index*.4,.8]],
  direction:[],direction_status:'unknown',boundary_polylines:[[[.1,.2],[.1,.8]]],
  stop_line_candidates:[{points:[[.1+index*.4,.3],[.3+index*.4,.3]]}]});
const context={intersectionAssistData:{proposal_set_id:'run-a',lane_proposals:[proposal(0),proposal(1)]},
  laneAssistMetadata:{method:'manual_polygon'},points:[],directionPoints:[],stopLinePoints:[],signalPoints:[],
  restrictedPoints:[],restrictedInitialPresence:false,autoLaneLines:[],activeLaneIndex:0,drawingArmed:false,
  alerts:[],confirm:()=>true,alert(message){context.alerts.push(message)},setMode(){},
  document:{querySelectorAll:()=>buttons,getElementById(id){return controls[id]??=( {checked:false,value:'Lane 1',textContent:''} )}}};
vm.createContext(context);
vm.runInContext(handlers,context);
context.redraw=()=>context.refreshSceneProposalButtons();

context.refreshSceneProposalButtons();
assert.ok(buttons.every(button=>button.disabled),'Manual lane must not accept an unrelated proposed stop line');
context.applyMarkedStopLine(1,0);
assert.equal(context.stopLinePoints.length,0);
assert.equal(context.alerts.length,1);

context.applyMarkedLane(0);
assert.equal(buttons[0].disabled,false);
assert.equal(buttons[1].disabled,true);
context.applyMarkedStopLine(0,0);
const correctStop=JSON.stringify(context.stopLinePoints);
context.applyMarkedStopLine(1,0);
assert.equal(JSON.stringify(context.stopLinePoints),correctStop,'Wrong-lane candidate must not replace the stop line');

const saved=context.currentLaneState();
context.loadLaneState({polygon:[],lane_assist:{method:'unconfigured'}});
assert.ok(buttons.every(button=>button.disabled),'New lane must not inherit proposal ownership');
context.loadLaneState(saved);
assert.equal(buttons[0].disabled,false,'Returning to the configured lane restores its ownership');
assert.equal(context.autoLaneLines.length,1,'Boundary evidence survives a lane switch');

context.intersectionAssistData={...context.intersectionAssistData,proposal_set_id:'run-b'};
context.refreshSceneProposalButtons();
assert.ok(buttons.every(button=>button.disabled),'New recognition results invalidate old proposal ownership');
context.applyMarkedStopLine(0,0);
assert.equal(JSON.stringify(context.stopLinePoints),correctStop,'Stale proposals must not modify current geometry');
console.log('Calibration UI regression checks passed: lane ownership, lane switching, stale results, boundary persistence.');

{
  const handlers=['sourceFramePresented','scheduleSourceFrameCapture'].map(name=>source.split('\n').find(line=>line.startsWith(`function ${name}(`))).join('\n');
  const callbacks=[];let captures=0,applied=0;
  const fixture={annotationSourceVersion:1,frameReady:false,pendingSample:{camera:'first'},
    preview:{requestVideoFrameCallback(callback){callbacks.push(callback)}},
    captureFrame(){captures++;fixture.frameReady=true},applySampleSetup(){applied++},loadVideoMetadata(){}};
  vm.createContext(fixture);vm.runInContext(handlers,fixture);
  fixture.scheduleSourceFrameCapture();
  assert.equal(captures,0,'Metadata alone must not freeze an unpresented black video frame');
  fixture.annotationSourceVersion=2;
  callbacks[0]();assert.equal(captures,0,'An old source frame must not freeze the new source');
  fixture.scheduleSourceFrameCapture();callbacks[1]();
  assert.equal(captures,1);assert.equal(applied,1);
  console.log('Setup capture check passed: wait for a presented video frame and reject stale source callbacks.');
}

// Related cameras share a data split, but must retain distinct calibration IDs.
(async()=>{
  const handler=source.split('\n').find(line=>line.startsWith('async function loadDatasetCandidates('));
  assert.ok(handler);
  const samples={},options=[];
  const fixture={sampleCatalog:samples,caseFacts:{},caseOrder:[],console,renderCaseGallery(){},
    fetch:async()=>({ok:true,json:async()=>({candidates:['north','south'].map(id=>({
      id,library_tier:'curated',rule:'wrong_way',role:'control_candidate',source_id:'related-cameras',
      camera_id:`camera-${id}`,reason:'Visible lane directions',path:`data/${id}.mp4`,media_url:`/media/${id}`
    }))})}),
    document:{createElement:()=>({}),getElementById:()=>({appendChild(option){options.push(option)}})}};
  vm.createContext(fixture);vm.runInContext(handler,fixture);
  await fixture.loadDatasetCandidates();
  assert.equal(samples['candidate-north'].sourceGroup,samples['candidate-south'].sourceGroup);
  assert.notEqual(samples['candidate-north'].camera,samples['candidate-south'].camera,
    'A shared train/test group must not reuse calibration across different viewpoints');
  assert.equal(samples['candidate-north'].points.length,0,'Unreviewed clips must not acquire invented geometry');
  console.log('Video library UI check passed: related-source grouping preserves distinct camera geometry.');
})().catch(error=>{console.error(error);process.exitCode=1});
