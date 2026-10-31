/* Offline geometry calibration UI; Plotly is embedded in the HTML by geometry_task.py.
   This editor never contacts a server. The ZIP contains calibrated GLBs for convenience,
   while the cloud finalizer deterministically regenerates and validates approved GLBs. */
'use strict';
const context = JSON.parse(document.getElementById('geometry-editor-data').textContent);
const panel = document.getElementById('geometry-editor-controls');
const chart = document.querySelector('.plotly-graph-div');
const names = context.objects;
const state = Object.fromEntries(names.map(name => [name, {
  angles:[0,0,0], axis:'x', lengths:{x:'',y:'',z:''}, origin:'bottom_center'
}]));
const originals = {};
for (const name of names) {
  originals[name] = context.vertices[name].map((encoded,k) => {
    const bytes=asArrayBuffer(encoded);
    if(bytes.byteLength%12!==0)throw Error('无效 mesh 顶点缓存');
    const vertices=new Float32Array(bytes.buffer,bytes.byteOffset,bytes.length/4);
    const x=[],y=[],z=[];
    for(let i=0;i<vertices.length;i+=3){x.push(vertices[i]);y.push(vertices[i+1]);z.push(vertices[i+2]);}
    return {idx:context.indices[name][k],x,y,z};
  });
}
let selected = names[0], debounce = null, last = {};
const $ = id => document.getElementById(id);
function rad(deg){ return deg * Math.PI / 180; }
function matrixXYZ(angles){
  const [x,y,z]=angles.map(rad), [cx,sx,cy,sy,cz,sz]=[Math.cos(x),Math.sin(x),Math.cos(y),Math.sin(y),Math.cos(z),Math.sin(z)];
  return [
    [cy*cz, cz*sx*sy-cx*sz, sx*sz+cx*cz*sy],
    [cy*sz, cx*cz+sx*sy*sz, cx*sy*sz-cz*sx],
    [-sy, cy*sx, cx*cy]
  ];
}
function rotate(p,R){return R.map(r=>r[0]*p[0]+r[1]*p[1]+r[2]*p[2]);}
function geometryInfo(name){
  const st=state[name], R=matrixXYZ(st.angles), lo=[Infinity,Infinity,Infinity], hi=[-Infinity,-Infinity,-Infinity];
  for(const geo of originals[name]) for(let i=0;i<geo.x.length;i++){
    const p=rotate([geo.x[i],geo.y[i],geo.z[i]],R);
    for(let j=0;j<3;j++){lo[j]=Math.min(lo[j],p[j]);hi[j]=Math.max(hi[j],p[j]);}
  }
  const bounds=hi.map((v,j)=>v-lo[j]);
  const v=Number(st.lengths[st.axis]);
  const measured=String(st.lengths[st.axis]).trim()!=='' && Number.isFinite(v) && v>0;
  const scale=measured ? (v/100)/bounds['xyz'.indexOf(st.axis)] : 1;
  const min=lo.map(v=>v*scale), max=hi.map(v=>v*scale);
  let offset=[0,0,0];
  if(st.origin==='bottom_center') offset=[-(min[0]+max[0])/2,-(min[1]+max[1])/2,-min[2]];
  else if(st.origin==='bbox_center') offset=min.map((v,j)=>-(v+max[j])/2);
  const dims=bounds.map(v=>v*scale);
  return {R,scale,offset,dims,measured,lo,min,max};
}
function syncFromInputs(){
  const st=state[selected];
  st.angles=[Number($('rx').value),Number($('ry').value),Number($('rz').value)];
  st.axis=$('anchor').value; st.origin=$('origin').value;
  for(const axis of 'xyz') st.lengths[axis]=$('len'+axis).value;
}
function syncToInputs(){
  const st=state[selected];
  ['rx','ry','rz'].forEach((id,j)=>{ $(id).value=st.angles[j]; $(id+'slide').value=Math.max(-180,Math.min(180,st.angles[j]));});
  $('anchor').value=st.axis; $('origin').value=st.origin;
  for(const axis of 'xyz') $('len'+axis).value=st.lengths[axis];
  $('active-label').textContent=selected;
}
function render(){
  const info=geometryInfo(selected); last[selected]=info;
  const st=state[selected], updates=[];
  for(const name of names){
    for(const geo of originals[name]){
      if(name!==selected){updates.push({idx:geo.idx, visible:false});continue;}
      const tx=[],ty=[],tz=[];
      for(let i=0;i<geo.x.length;i++){
        const p=rotate([geo.x[i],geo.y[i],geo.z[i]],info.R).map((v,j)=>v*info.scale+info.offset[j]);
        tx.push(p[0]);ty.push(p[1]);tz.push(p[2]);
      }
      updates.push({idx:geo.idx,x:tx,y:ty,z:tz,visible:true});
    }
  }
  const ix=[], vis=[], X=[], Y=[], Z=[];
  for(const u of updates){ix.push(u.idx);vis.push(u.visible); X.push(u.x || null);Y.push(u.y || null);Z.push(u.z || null);}
  // visibility is separate, to avoid changing the cached data for hidden traces
  Plotly.restyle(chart, {visible:vis}, ix);
  for(const u of updates){ if(u.visible) Plotly.restyle(chart,{x:[u.x],y:[u.y],z:[u.z]},[u.idx]); }
  const axisLen=Math.max(...info.dims,1e-4)*0.6;
  const ai=context.axis_indices;
  for(let j=0;j<3;j++){
    const coords=[ [0,0], [0,0], [0,0] ];coords[j][1]=axisLen;
    Plotly.restyle(chart,{x:[coords[0]],y:[coords[1]],z:[coords[2]]},[ai[j]]);
  }
  let message='';
  if(info.measured){
    message=`当前尺寸（厘米）：X ${(+info.dims[0]*100).toFixed(2)} · Y ${(+info.dims[1]*100).toFixed(2)} · Z ${(+info.dims[2]*100).toFixed(2)}；统一缩放 ${info.scale.toPrecision(5)}`;
    for(const [i,axis] of [...'xyz'].entries()){
      const expected=Number(st.lengths[axis]);
      if(String(st.lengths[axis]).trim()!=='' && (!Number.isFinite(expected) || expected<=0)) message+='\n⚠ '+axis.toUpperCase()+' 实测长度无效';
      else if(expected>0 && Math.abs(info.dims[i]*100-expected)/expected>.25) message+='\n⚠ '+axis.toUpperCase()+' 与实测值偏差超过25%，请检查几何或方向';
    }
  }else message='请在选定的尺度基准轴填写实际尺寸（厘米）；当前只展示原始模型方向。';
  $('dimensions').textContent=message;
  Plotly.relayout(chart,{'title.text':`${selected}  ·  ${info.measured?'米制预览':'原始尺度，未校准'}`});
}
function schedule(){clearTimeout(debounce);debounce=setTimeout(render,100);}
function updateInput(){syncFromInputs();schedule();}
for(const axis of ['x','y','z']){
  const id='r'+axis;
  $(id).addEventListener('input',()=>{$(id+'slide').value=$(id).value;updateInput();});
  $(id+'slide').addEventListener('input',()=>{$(id).value=$(id+'slide').value;updateInput();});
  $('len'+axis).addEventListener('input',updateInput);
  $('plus'+axis).addEventListener('click',()=>{$(id).value=Number($(id).value)+90;$(id+'slide').value=$(id).value;updateInput();});
  $('minus'+axis).addEventListener('click',()=>{$(id).value=Number($(id).value)-90;$(id+'slide').value=$(id).value;updateInput();});
}
$('anchor').addEventListener('change',updateInput);$('origin').addEventListener('change',updateInput);
$('object-select').innerHTML=names.map(n=>`<option value="${n}">${n}</option>`).join('');
$('object-select').addEventListener('change',()=>{syncFromInputs();selected=$('object-select').value;syncToInputs();render();});
function calibrationRecord(){
  syncFromInputs();
  const all={};
  for(const name of names){
    const st=state[name], info=geometryInfo(name);
    if(!info.measured || !Number.isFinite(info.scale) || info.scale<=0) throw Error(`${name} 尚未在尺度基准轴填写正数长度（厘米）`);
    const lengths={};
    for(const axis of 'xyz'){
      const v=st.lengths[axis];
      lengths[axis]=v===''?null:Number(v)/100;
      if(lengths[axis]!==null && (!Number.isFinite(lengths[axis])||lengths[axis]<=0)) throw Error(`${name} 的 ${axis} 长度无效`);
    }
    for(const [i,axis] of [...'xyz'].entries()) if(lengths[axis]!==null && Math.abs(info.dims[i]-lengths[axis])/lengths[axis]>.25)
      throw Error(`${name} 的 ${axis} 轴模型与实测尺寸相差超过 25%；不能用尺度校准修复形状错误`);
    all[name]={rotation_xyz_deg:st.angles.map(Number),scale_anchor_axis:st.axis,measurements_m:lengths,origin:st.origin};
  }
  return {schema:context.schema,unit:'meter',objects:all,editor_provenance:{
    ui_version:1,raw_mesh_sha256:context.hashes,generated_utc:new Date().toISOString(),
    review:'calibrated_in_3d_editor'}
  };
}
function asArrayBuffer(base64){
  const binary=atob(base64),out=new Uint8Array(binary.length);
  for(let i=0;i<binary.length;i++)out[i]=binary.charCodeAt(i);
  return out;
}
function u32(b,off){return new DataView(b.buffer,b.byteOffset,b.byteLength).getUint32(off,true);}
function pack32(v){const b=new Uint8Array(4);new DataView(b.buffer).setUint32(0,v,true);return b;}
function join(parts){const n=parts.reduce((s,p)=>s+p.byteLength,0),out=new Uint8Array(n);let i=0;for(const p of parts){out.set(p,i);i+=p.length;}return out;}
function appendGlbRootTransform(raw, info){
  if(u32(raw,0)!==0x46546c67||u32(raw,4)!==2||u32(raw,8)!==raw.length)throw Error('无效 GLB');
  let offset=12,parts=[];
  while(offset<raw.length){
    const size=u32(raw,offset),type=u32(raw,offset+4);offset+=8;
    const chunk=raw.slice(offset,offset+size);offset+=size;
    parts.push({type,chunk});
  }
  const jsonIndex=parts.findIndex(p=>p.type===0x4e4f534a);
  if(jsonIndex<0)throw Error('GLB JSON chunk 缺失');
  const doc=JSON.parse(new TextDecoder().decode(parts[jsonIndex].chunk));
  if(!Array.isArray(doc.scenes)||doc.scenes.length===0) throw Error('GLB scenes 缺失');
  doc.nodes=doc.nodes||[];
  const R=info.R,s=info.scale,t=info.offset;
  const transform=[R[0][0]*s,R[1][0]*s,R[2][0]*s,0,
    R[0][1]*s,R[1][1]*s,R[2][1]*s,0,
    R[0][2]*s,R[1][2]*s,R[2][2]*s,0,t[0],t[1],t[2],1];
  // Apply the calibrated world transform to each active scene root node.
  // Avoid adding a new parent node: some GLB loaders (including trimesh)
  // traverse the node graph differently and may ignore synthetic parents.
  function multiply(a,b){
    const c=new Array(16).fill(0);
    for(let col=0;col<4;col++)for(let row=0;row<4;row++)
      for(let k=0;k<4;k++) c[col*4+row]+=a[k*4+row]*b[col*4+k];
    return c;
  }
  function nodeMatrix(node){
    if(node.matrix)return node.matrix;
    const [x,y,z,w]=node.rotation||[0,0,0,1];
    const [sx,sy,sz]=node.scale||[1,1,1];
    const [tx,ty,tz]=node.translation||[0,0,0];
    return [
      (1-2*y*y-2*z*z)*sx,(2*x*y+2*z*w)*sx,(2*x*z-2*y*w)*sx,0,
      (2*x*y-2*z*w)*sy,(1-2*x*x-2*z*z)*sy,(2*y*z+2*x*w)*sy,0,
      (2*x*z+2*y*w)*sz,(2*y*z-2*x*w)*sz,(1-2*x*x-2*y*y)*sz,0,
      tx,ty,tz,1
    ];
  }
  const updated=new Set();
  for(const scene of doc.scenes){
    if(!Array.isArray(scene.nodes))throw Error('GLB 场景缺少根节点');
    for(const rootIndex of scene.nodes){
      const root=doc.nodes[rootIndex];if(!root)throw Error('无效 GLB 根节点');
      // Trimesh GLBs use a sentinel node named `world`, which some loaders
      // treat as their own graph base and ignore its local matrix. Move the
      // calibration onto its children instead, preserving the world frame.
      const targets = root.name==='world' && root.mesh===undefined && Array.isArray(root.children)
        ? root.children : [rootIndex];
      for(const index of targets){
        if(updated.has(index))continue;
        const node=doc.nodes[index];if(!node)throw Error('无效 GLB 子节点');
        node.matrix=multiply(transform,nodeMatrix(node));
        delete node.translation;delete node.scale;delete node.rotation;
        updated.add(index);
      }
    }
  }
  let jsonChunk=new TextEncoder().encode(JSON.stringify(doc));
  if(jsonChunk.length%4)jsonChunk=join([jsonChunk,new Uint8Array(4-jsonChunk.length%4).fill(32)]);
  parts[jsonIndex].chunk=jsonChunk;
  const chunkParts=parts.map(p=>join([pack32(p.chunk.length),pack32(p.type),p.chunk]));
  const total=12+chunkParts.reduce((a,p)=>a+p.length,0);
  return join([pack32(0x46546c67),pack32(2),pack32(total),...chunkParts]);
}
const crcTable=new Uint32Array(256);
for(let n=0;n<256;n++){let c=n;for(let k=0;k<8;k++)c=(c&1)?0xedb88320^(c>>>1):c>>>1;crcTable[n]=c>>>0;}
function crc32(bytes){let c=0xffffffff;for(const b of bytes)c=crcTable[(c^b)&255]^(c>>>8);return (c^0xffffffff)>>>0;}
function u16(n){const b=new Uint8Array(2);new DataView(b.buffer).setUint16(0,n,true);return b;}
function zipStored(files){
  let off=0;const local=[],central=[];
  for(const {name,data} of files){
    const filename=new TextEncoder().encode(name),checksum=crc32(data);
    const header=join([pack32(0x04034b50),u16(20),u16(0),u16(0),u16(0),u16(0),
      pack32(checksum),pack32(data.length),pack32(data.length),u16(filename.length),u16(0),filename]);
    local.push(header,data);
    central.push(join([pack32(0x02014b50),u16(20),u16(20),u16(0),u16(0),u16(0),u16(0),
      pack32(checksum),pack32(data.length),pack32(data.length),u16(filename.length),u16(0),u16(0),u16(0),
      u16(0),pack32(0),pack32(off),filename]));
    off+=header.length+data.length;
  }
  const c=join(central);
  const trailer=join([pack32(0x06054b50),u16(0),u16(0),u16(files.length),u16(files.length),
    pack32(c.length),pack32(off),u16(0)]);
  return join([...local,c,trailer]);
}
function downloadBlob(bytes,filename,type){
  const blob=new Blob([bytes],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),20000);
}
async function exportZip(){
  const button=$('export');button.disabled=true;
  try{
    const record=calibrationRecord(),files=[{name:'calibration.json',data:new TextEncoder().encode(JSON.stringify(record,null,2)+'\n')}];
    for(const name of names){
      const raw=asArrayBuffer(context.raw[name]);
      const glb=appendGlbRootTransform(raw,geometryInfo(name));
      files.push({name:`meshes/${name}.glb`,data:glb});
    }
    downloadBlob(zipStored(files),'geometry_metric_export.zip','application/zip');
    $('export-status').textContent='✅ 已下载 geometry_metric_export.zip。上传到云端 geometry/ 目录，然后执行 finalize。';
  }catch(err){$('export-status').textContent='❌ '+String(err.message || err);}
  finally{button.disabled=false;}
}
$('export').addEventListener('click',exportZip);
syncToInputs();render();
