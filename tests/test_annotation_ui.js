const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../src/traffic_edge/web_ui.py'), 'utf8');
const names = ['ensureAnnotationVideo', 'saveAnnotationDraft', 'scheduleDraftSave', 'restoreAnnotationDraft', 'loadVideoMetadata'];
const handlers = names.map(name => {
  const line = source.split('\n').find(line => line.startsWith(`function ${name}(`) || line.startsWith(`async function ${name}(`));
  assert.ok(line, `Missing production handler ${name}`);
  return line;
}).join('\n');
const deferred = () => {let resolve; const promise = new Promise(done => {resolve = done}); return {promise, resolve}};
const response = data => ({ok: true, json: async () => data});
function fixture() {
  const controls = {};
  const timers = [];
  const context = {
    annotationSourceVersion: 1, annotationRevision: 0, annotationSaveQueue: Promise.resolve(), annotationUpload: null,
    annotationDraftTimer: null, annotationRestoredFor: null, serverVideoPath: 'data/a.mp4', selectedFile: {},
    labels: [{id: 'a'}], reviewedIntervals: [], groundTruthFinalized: false, sourceVideoFps: 30,
    reviewVideo: {currentTime: 0}, renderLabels() {},
    document: {getElementById(id) {return controls[id] ??= {textContent: '', value: ''}}},
    setTimeout(callback) {timers.push(callback); return timers.length},
    draftPayload() {return {video_path: context.serverVideoPath, labels: context.labels}},
  };
  vm.createContext(context);
  vm.runInContext(handlers, context);
  return {context, controls, timers};
}

(async () => {
  // A slow restore for video A cannot replace video B's labels.
  {
    const {context: c} = fixture();
    const pending = deferred(); c.fetch = () => pending.promise;
    const loading = c.restoreAnnotationDraft();
    c.annotationSourceVersion++; c.serverVideoPath = 'data/b.mp4'; c.labels = [{id: 'b'}];
    pending.resolve(response({draft: {labels: [{id: 'old-a'}]}})); await loading;
    assert.equal(c.labels[0].id, 'b');
  }
  // A restore also cannot overwrite edits made while its request was pending.
  {
    const {context: c} = fixture();
    const pending = deferred(); c.fetch = () => pending.promise;
    const loading = c.restoreAnnotationDraft();
    c.annotationRevision++; c.labels = [{id: 'edited'}];
    pending.resolve(response({draft: {labels: [{id: 'old-a'}]}})); await loading;
    assert.equal(c.labels[0].id, 'edited');
  }
  // Save snapshots remain attached to their source, and writes cannot overtake.
  {
    const {context: c, controls} = fixture();
    const first = deferred(); const calls = [];
    c.fetch = (_url, options) => {calls.push(JSON.parse(options.body)); return calls.length === 1 ? first.promise : Promise.resolve(response({path: 'b'}))};
    const a = c.saveAnnotationDraft();
    c.labels[0].id = 'mutated-after-save';
    await Promise.resolve(); await Promise.resolve();
    c.annotationSourceVersion++; c.serverVideoPath = 'data/b.mp4'; c.labels = [{id: 'b'}];
    const b = c.saveAnnotationDraft();
    await Promise.resolve(); await Promise.resolve();
    assert.equal(calls.length, 1, 'Only one save may be in flight');
    assert.equal(calls[0].labels[0].id, 'a', 'Save must hold an immutable snapshot');
    controls.draftState = {textContent: 'B is unsaved'};
    first.resolve(response({path: 'a'})); await a; await b;
    assert.equal(calls[0].video_path, 'data/a.mp4');
    assert.equal(calls[1].video_path, 'data/b.mp4');
    assert.equal(calls[1].labels[0].id, 'b');
    assert.equal(controls.labelSaveStatus.textContent, 'Draft saved: b');
  }
  // Continuous playback must not postpone saving forever.
  {
    const {context: c, timers} = fixture();
    let saves = 0; c.saveAnnotationDraft = () => {saves++};
    for (let i = 0; i < 20; i++) c.scheduleDraftSave();
    assert.equal(timers.length, 1);
    timers[0](); assert.equal(saves, 1); assert.equal(c.annotationDraftTimer, null);
    c.scheduleDraftSave(); assert.equal(timers.length, 2);
  }
  // A shared pending upload must not attach itself to a different video.
  {
    const {context: c} = fixture();
    c.serverVideoPath = null;
    const pending = deferred(); let uploads = 0;
    c.uploadFile = () => {uploads++; return pending.promise};
    const a = c.ensureAnnotationVideo(), b = c.ensureAnnotationVideo();
    const settled = Promise.allSettled([a, b]);
    c.annotationSourceVersion++; c.serverVideoPath = 'data/b.mp4';
    pending.resolve('data/a-upload.mp4');
    assert.ok((await settled).every(item => item.status === 'rejected'));
    assert.equal(uploads, 1);
    assert.equal(c.serverVideoPath, 'data/b.mp4');
  }
  // Old metadata must not replace the frame rate used for the new video.
  {
    const {context: c} = fixture(); const pending = deferred();
    c.fetch = () => pending.promise;
    const loading = c.loadVideoMetadata();
    c.annotationSourceVersion++; c.serverVideoPath = 'data/b.mp4'; c.sourceVideoFps = 60;
    pending.resolve(response({fps: 12, frames: 120})); await loading;
    assert.equal(c.sourceVideoFps, 60);
  }
  console.log('Annotation UI checks passed: source isolation, immutable ordered saves, playback autosave, upload and metadata races.');
})().catch(error => {console.error(error); process.exitCode = 1});

// A short seek gap remains unwatched in the coverage gate.
{
  const c={reviewVideo:{duration:10},reviewedIntervals:[[0,4],[4.3,10]],Number,Math};
  vm.createContext(c);
  for(const name of ['mergeReviewedIntervals','reviewCoverageFraction'])
    vm.runInContext(source.split('\n').find(line=>line.startsWith(`function ${name}(`)),c);
  assert.equal(c.reviewCoverageFraction(),.97);
  assert.equal(c.reviewedIntervals.length,2);
}
