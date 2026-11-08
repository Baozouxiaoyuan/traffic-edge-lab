// Exercise the production workspace handlers, including blind-review visibility.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname,'../src/traffic_edge/web_ui.py'),'utf8');
const handlers = ['showWorkspace','syncAnalysisWorkspace'].map(name => {
  const line = source.split('\n').find(line => line.startsWith(`function ${name}(`));
  assert.ok(line, `Missing production handler: ${name}`);
  return line;
}).join('\n');
const ids = ['video-step','geometry-step','review-step','labels-step','readiness-step'];
const panels = ids.map(id => ({id,tagName:id==='labels-step'||id==='readiness-step'?'DETAILS':'SECTION',
  hidden:false,open:false,querySelector:()=>null}));
const links = ids.map(id => ({dataset:{workspaceLink:id},attributes:{},
  setAttribute(key,value){this.attributes[key]=value},removeAttribute(key){delete this.attributes[key]}}));
let complete = false;
const controls = {result:{classList:{contains:()=>complete}},annotationMode:{value:'ground_truth'},
  analysisEmpty:{},runBtn:{disabled:false},blindReviewNotice:{},sessionState:{}};
const fixture = {document:{querySelectorAll(selector){return selector==='[data-workspace]'?panels:links},
  getElementById(id){return controls[id]}},location:{hash:''},
  history:{pushState(_state,_title,hash){fixture.location.hash=hash}},window:{scrollTo(){}},statusEl:{textContent:'Ready'}};
vm.createContext(fixture);vm.runInContext(handlers,fixture);
assert.equal(fixture.showWorkspace('video-step'),'video-step');
assert.equal(panels.filter(panel=>!panel.hidden).length,1);
fixture.showWorkspace('labels-step');
assert.equal(panels[3].open,true,'The annotation workspace must open when selected');
fixture.showWorkspace('geometry-step');
assert.equal(panels[3].open,false,'Hidden annotation must not consume the playback shortcut');
assert.equal(links[1].attributes['aria-current'],'page');
assert.equal(links[3].attributes['aria-current'],undefined);
assert.equal(fixture.showWorkspace('analysis-step'),'review-step','Old settings links must still resolve');
assert.equal(fixture.showWorkspace('unknown'),'video-step');
fixture.syncAnalysisWorkspace();
assert.equal(controls.analysisEmpty.hidden,false);
complete=true;fixture.statusEl.textContent='Analysis complete';fixture.syncAnalysisWorkspace();
assert.equal(controls.analysisEmpty.hidden,true);
assert.equal(controls.blindReviewNotice.hidden,false,'A hidden result must have a visible explanation');
assert.equal(controls.sessionState.textContent,'Analysis complete');
controls.annotationMode.value='error_analysis';fixture.syncAnalysisWorkspace();
assert.equal(controls.blindReviewNotice.hidden,true);
console.log('Workspace UI checks passed: navigation, legacy links, keyboard scope and blind-review status.');
