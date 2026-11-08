const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync(path.join(__dirname, '../src/traffic_edge/study_ui.py'), 'utf8');
const embedded = source.match(/STUDY_SCRIPT = r'''([\s\S]*?)'''/)[1];
const handlers = embedded.slice(0, embedded.indexOf("studyElement('studyRegisterBtn').onclick="));
const controls = {};
const rendered = [];
const requests = [];
const pending = {};
const control = id => controls[id] ??= {value: '', disabled: false, textContent: ''};
const fixture = {
  document: {getElementById: control},
  groundTruthFinalized: false,
  serverVideoPath: 'data/source.mp4',
  cameraProfilePayload: () => ({camera_id: 'camera-a'}),
  fetch: async (url, options) => {
    requests.push({url, options});
    if (url.includes('/report?')) return new Promise(resolve => {pending[url] = resolve;});
    return {ok: true, json: async () => ({job_id: 'job-one'})};
  },
  setTimeout() {},
};
vm.createContext(fixture);
vm.runInContext(handlers, fixture);
fixture.recordRender = report => rendered.push(report.run_name);
vm.runInContext('renderStudyReport = recordRender', fixture);

async function main() {
  assert.equal(fixture.studyMetric({true_positives: 0, false_positives: 0, false_negatives: 1, precision: 0}, 'precision'), '—');
  assert.equal(fixture.studyMetric({true_positives: 0, false_positives: 0, false_negatives: 1, recall: 0}, 'recall'), '0.000');
  await fixture.registerStudyCase();
  assert.equal(requests.length, 0, 'Independent review must finish before registration');
  assert.match(control('studyRegisterStatus').textContent, /finalize/);
  assert.equal(control('studyRegisterBtn').disabled, false);

  const first = fixture.loadStudyReport('old');
  const second = fixture.loadStudyReport('new');
  pending['/api/study/report?run=new']({ok: true, json: async () => ({run_name: 'new'})});
  await second;
  pending['/api/study/report?run=old']({ok: true, json: async () => ({run_name: 'old'})});
  await first;
  assert.deepEqual(rendered, ['new'], 'A delayed report cannot overwrite the most recent selection');

  requests.length = 0;
  control('studyRunSplit').value = 'test';
  control('studyModel').value = 'yolo11n';
  control('studyTracker').value = 'bytetrack';
  control('studyDevice').value = 'cpu';
  fixture.fetch = async (url, options) => {
    requests.push({url, options});
    return {ok: false, json: async () => ({error: 'No registered test cases'})};
  };
  await fixture.startStudyBatch();
  assert.equal(JSON.parse(requests[0].options.body).split, 'test');
  assert.equal(JSON.parse(requests[0].options.body).device, 'cpu');
  assert.equal(control('studyRunBtn').disabled, false, 'A failed start permits correction and retry');
  assert.equal(control('studyStatus').textContent, 'No registered test cases');
  const panel = {isConnected: true, textContent: '', requestVersion: 0};
  const diagnosed = [];
  fixture.recordDiagnosis = (panel, run, error, data) => diagnosed.push(data.selected_track);
  vm.runInContext('renderStudyDiagnosis = recordDiagnosis', fixture);
  const diagnosticPending = {};
  fixture.fetch = async url => new Promise(resolve => {diagnosticPending[url] = resolve;});
  const oldDiagnosis = fixture.loadStudyDiagnosis(panel, 'batch', 'error-00001', 7);
  const newDiagnosis = fixture.loadStudyDiagnosis(panel, 'batch', 'error-00001', 8);
  diagnosticPending[fixture.diagnosisUrl('batch', 'error-00001', 8)]({ok: true,json: async()=>({selected_track:8})});
  await newDiagnosis;
  diagnosticPending[fixture.diagnosisUrl('batch', 'error-00001', 7)]({ok: true,json: async()=>({selected_track:7})});
  await oldDiagnosis;
  assert.deepEqual(diagnosed,[8],'A late response must not replace the newly selected vehicle');
  const detached = fixture.loadStudyDiagnosis(panel, 'batch', 'error-00002');
  panel.isConnected = false;
  diagnosticPending[fixture.diagnosisUrl('batch', 'error-00002')]({ok:true,json:async()=>({selected_track:99})});
  await detached;
  assert.deepEqual(diagnosed,[8],'Changing reports must discard pending diagnostic responses');
  console.log('Study UI checks passed: blind review, split/device forwarding, undefined metrics, stale reports and retry.');
}

main().catch(error => {console.error(error); process.exitCode = 1;});
