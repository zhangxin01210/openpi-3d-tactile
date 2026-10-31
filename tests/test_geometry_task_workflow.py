"""Offline synthetic end-to-end tests; no A800/SAM2/SAM3D weights needed."""
from __future__ import annotations

import importlib.util
import os
import base64
import subprocess
import shutil
import zipfile
import json
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from PIL import Image

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/preprocess/geometry_task.py'
spec=importlib.util.spec_from_file_location('geometry_task',SCRIPT)
geom=importlib.util.module_from_spec(spec)
spec.loader.exec_module(geom)


def args(out,**kwargs):
    return SimpleNamespace(out=out,**kwargs)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'output/preprocess/task_a/geometry'
        (self.root/'photos').mkdir(parents=True)
        for i,name in enumerate(('button','box')):
            img=np.full((300,400,3),230,np.uint8)
            cv2.rectangle(img,(70+i*10,60),(220+i*10,250),(20,10+80*i,180),-1)
            cv2.imwrite(str(self.root/'photos'/f'{name}.jpg'),img)
        geom.prepare(args(self.root,objects='button,box',max_side=1024))
        for name in ('button','box'):
            rd=self.root/'reference'/name
            pic=cv2.imread(str(rd/'image.jpg'))
            obj={'schema':geom.SCHEMA,'object':name,'image_width':pic.shape[1],
                 'image_height':pic.shape[0],'points':{'positive':[[120,120]],'negative':[]}}
            geom.save_json(rd/'prompts.json',obj)

    def make_segmented(self):
        fake=types.ModuleType('sam2_task')
        fake.image_predictor=lambda _:object()
        def seg(predictor,frame,pos,neg):
            mask=np.zeros(frame.shape[:2],np.uint8)
            mask[50:260,60:240]=1
            return mask,.91
        fake.sam_predict=seg
        with patch.dict(sys.modules,{'sam2_task':fake}):
            geom.segment(args(self.root,checkpoint=Path('/dev/null')))
        geom.approve_masks(args(self.root,reviewer='test'))

    def make_raws(self):
        for name in ('button','box'):
            rd=self.root/'reconstruction'/name
            rd.mkdir(parents=True)
            obj=trimesh.creation.box(extents=(2,4,6))
            obj.export(str(rd/'raw.glb'))
            geom.save_json(rd/'report.json',{'schema':geom.SCHEMA,'object':name,
                  'raw_sha256':geom.digest(rd/'raw.glb'),
                  'photo_sha256':geom.manifest(self.root)['photos'][name]['reference_sha256'],
                  'mask_sha256':geom.digest(geom.mask_path(self.root,name)),
                  'mask_review_sha256':geom.digest(self.root/'reference'/name/'mask_review.json')})

    def test_browser_zip_export_and_one_command_finalize(self):
        """Real Node execution of offline JS: ZIP contains valid metric GLBs."""
        if shutil.which('node') is None:
            self.skipTest('Node.js is not installed')
        self.make_segmented(); self.make_raws()
        geom.inspect_meshes(args(self.root))
        editor=self.root/'preview'/'metric_editor.html'
        page=editor.read_text(encoding='utf8')
        self.assertIn('旋转模型',page)
        self.assertIn('geometry-editor-data',page)
        self.assertIn('geometry_metric_export.zip',page)
        self.assertIn('button',page)
        self.assertIn('box',page)
        # Feed the real JS a synthetic DOM and real trimesh-exported GLB bytes.
        # This exercises live geometry math, browser GLB wrapping and ZIP CRCs.
        m=geom.manifest(self.root)
        payload={'schema':geom.SCHEMA,'objects':m['objects'],'indices':{},'axis_indices':[2,3,4],
                 'raw':{},'hashes':{},'vertices':{}}
        traces=[]
        for i,name in enumerate(m['objects']):
            raw=self.root/'reconstruction'/name/'raw.glb'
            scene=geom.scene_load(raw)
            v=np.asarray(scene.dump()[0].vertices)
            payload['indices'][name]=[i]
            payload['vertices'][name]=[base64.b64encode(np.asarray(v,dtype='<f4').tobytes()).decode('ascii')]
            payload['raw'][name]=base64.b64encode(raw.read_bytes()).decode('ascii')
            payload['hashes'][name]=geom.digest(raw)
            traces.append({'x':v[:,0].tolist(),'y':v[:,1].tolist(),'z':v[:,2].tolist()})
        traces.extend({'x':[0,1],'y':[0,0],'z':[0,0]} for _ in range(3))
        bundle=self.root/'geometry_metric_export.zip'
        source=SCRIPT.with_name('geometry_editor.js')
        node_script=r"""
const fs=require('fs'),vm=require('vm');
const payload=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const traces=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const fields={};
function elem(id){return fields[id] ||= {value:'',innerHTML:'',textContent:'',disabled:false,
  addEventListener(type,cb){this['on'+type]=cb;},click(){}};}
const controls={};
global.document={getElementById(id){if(id==='geometry-editor-data')return {textContent:JSON.stringify(payload)};return elem(id)},
  querySelector(){return {data:traces}},createElement(){return {click(){}}}};
global.Plotly={restyle(){},relayout(){}};
global.URL.createObjectURL=blob=>{global.savedBlob=blob;return 'blob:test'};
global.URL.revokeObjectURL=()=>{};
vm.runInThisContext(fs.readFileSync(process.argv[4],'utf8'),{filename:'geometry_editor.js'});
async function go(){
  // First object uses the input element, the other persists in state.
  fields.lenx.value='4';fields.rx.value='90';state.box.lengths.x='4';
  await exportZip();
  if(!global.savedBlob)throw Error('UI failed to export: '+fields['export-status'].textContent);
  fs.writeFileSync(process.argv[5],Buffer.from(await global.savedBlob.arrayBuffer()));
}
go().catch(e=>{console.error(e);process.exit(1)});
"""
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            (d/'script.js').write_text(node_script)
            (d/'payload.json').write_text(json.dumps(payload))
            (d/'traces.json').write_text(json.dumps(traces))
            subprocess.run(['node',str(d/'script.js'),str(d/'payload.json'),str(d/'traces.json'),str(source),str(bundle)],check=True,capture_output=True,text=True)
        with zipfile.ZipFile(bundle) as z:
            self.assertEqual(set(z.namelist()),{'calibration.json','meshes/button.glb','meshes/box.glb'})
            cfg=json.loads(z.read('calibration.json'))
            self.assertAlmostEqual(cfg['objects']['button']['measurements_m']['x'],.04)
            for name in m['objects']:
                client=trimesh.load(io.BytesIO(z.read(f'meshes/{name}.glb')),file_type='glb',force='scene')
                intended=[.04,.12,.08] if name=='button' else [.04,.08,.12]
                self.assertTrue(np.allclose(client.extents,intended,atol=1e-5))
        geom.finalize_ui(args(self.root,bundle=bundle,max_shape_error=.25,reviewer='tester'))
        final=geom.read_json(self.root/'approved'/'manifest.json')
        self.assertEqual(final['status'],'human_accepted')
        self.assertTrue((self.root/'approved'/'button.glb').is_file())
        self.assertTrue((self.root/'approved'/'box.glb').is_file())
        with self.assertRaises(FileExistsError):
            geom.finalize_ui(args(self.root,bundle=bundle,max_shape_error=.25,reviewer='tester'))

    def test_sam3d_runner_api_mock(self):
        """Verify the pinned SAM3D-style pipeline.run(rgb, mask=...) adapter."""
        self.make_segmented()
        fake_hydra=types.ModuleType('hydra')
        fake_utils=types.ModuleType('hydra.utils')
        fake_omega=types.ModuleType('omegaconf')
        fake_torch=types.ModuleType('torch')
        fake_torch.cuda=SimpleNamespace(is_available=lambda:True)
        calls=[]
        class Pipeline:
            def run(self, image, mask, seed, stage1_only, with_mesh_postprocess, with_texture_baking,
                    with_layout_postprocess, use_vertex_color):
                calls.append((image.shape,mask.shape,seed,with_layout_postprocess))
                return {'glb':trimesh.creation.box(extents=(2,3,4))}
        fake_utils.instantiate=lambda cfg:Pipeline()
        fake_hydra.utils=fake_utils
        fake_omega.OmegaConf=SimpleNamespace(load=lambda _:SimpleNamespace(
            depth_model=SimpleNamespace(model=SimpleNamespace(pretrained_model_name_or_path=''))))
        fake_source=Path(self.temp.name)/'sam-3d-objects'
        fake_package=fake_source/'sam3d_objects'
        fake_package.mkdir(parents=True)
        (fake_package/'__init__.py').write_text('')
        fake_pipeline=fake_source/'checkpoints/hf/pipeline.yaml'
        fake_pipeline.parent.mkdir(parents=True)
        fake_pipeline.write_text('hi: 1')
        fake_moge=Path(self.temp.name)/'model.pt';fake_moge.write_bytes(b'fake')
        with patch.dict(sys.modules,{'hydra':fake_hydra,'hydra.utils':fake_utils,
                                     'omegaconf':fake_omega,'torch':fake_torch}), \
             patch.object(geom,'_sam3d_local_dino',lambda:None), \
             patch.dict(os.environ, {'SAM3D_SOURCE_ROOT':str(fake_source)}), \
             patch.object(sys,'path',sys.path.copy()):
            geom.reconstruct(args(self.root,object=None,pipeline=fake_pipeline,moge=fake_moge,seed=42))
        self.assertEqual(len(calls),2)
        self.assertTrue((self.root/'reconstruction/button/raw.glb').is_file())
        self.assertTrue((self.root/'reconstruction/box/raw.glb').is_file())
        self.assertEqual(calls[0][2:],(42,False))
        # A completed run can be safely resumed without loading model again.
        geom.reconstruct(args(self.root,object=None,pipeline=fake_pipeline,moge=fake_moge,seed=42))

    def test_sam3d_runtime_recovers_previous_cloud_import_failure(self):
        """No manual PYTHONPATH/LIDRA/SDPA exports are needed for Hydra imports."""
        fake_source=Path(self.temp.name)/'sam-3d-objects'
        package=fake_source/'sam3d_objects'
        package.mkdir(parents=True)
        # Simulate the source checkout failing unless the known init bypass is set.
        (package/'__init__.py').write_text(
            "import os\n"
            "if os.environ.get('LIDRA_SKIP_INIT') != 'true':\n"
            "    import sam3d_objects.init\n"
        )
        config=fake_source/'checkpoints/hf/pipeline.yaml'
        config.parent.mkdir(parents=True)
        config.write_text('hi: 1')
        with patch.dict(os.environ, {}, clear=True), patch.object(sys,'path',sys.path.copy()):
            geom._configure_sam3d_runtime(config)
            self.assertEqual(os.environ['LIDRA_SKIP_INIT'],'true')
            self.assertEqual(os.environ['ATTN_BACKEND'],'sdpa')
            self.assertEqual(os.environ['SPARSE_ATTN_BACKEND'],'sdpa')
            self.assertEqual(sys.path[0],str(fake_source))
            import importlib
            try:
                obj=importlib.import_module('sam3d_objects')
                self.assertTrue(str(Path(obj.__file__).resolve()).startswith(str(fake_source)))
            finally:
                sys.modules.pop('sam3d_objects',None)

    def test_sam3d_runtime_preserves_explicit_overrides(self):
        fake_source=Path(self.temp.name)/'custom-source'
        package=fake_source/'sam3d_objects'
        package.mkdir(parents=True)
        (package/'__init__.py').write_text('')
        with patch.dict(os.environ, {'SAM3D_SOURCE_ROOT':str(fake_source),
                                     'LIDRA_SKIP_INIT':'true','ATTN_BACKEND':'flash',
                                     'SPARSE_ATTN_BACKEND':'sdpa','PYTHONPATH':'existing-module'}), \
             patch.object(sys,'path',sys.path.copy()):
            geom._configure_sam3d_runtime(Path(self.temp.name)/'elsewhere/pipeline.yaml')
            self.assertEqual(os.environ['ATTN_BACKEND'],'flash')
            self.assertEqual(os.environ['PYTHONPATH'],str(fake_source)+os.pathsep+'existing-module')

    def test_happy_path_and_integrity(self):
        self.make_segmented();self.make_raws()
        geom.inspect_meshes(args(self.root))
        spec=geom.read_json(self.root/'calibration.json')
        for name in spec['objects']:
            spec['objects'][name]['measurements_m']={'x':.04,'y':.08,'z':.12}
        geom.save_json(self.root/'calibration.json',spec)
        geom.calibrate(args(self.root,max_shape_error=.25,allow_shape_mismatch=False))
        geom.approve(args(self.root,reviewer='tester'))
        final=geom.read_json(self.root/'approved'/'manifest.json')
        self.assertEqual(final['status'],'human_accepted')
        for name in ('button','box'):
            mesh=geom.scene_load(self.root/'approved'/f'{name}.glb')
            self.assertTrue(np.allclose(mesh.bounds[1]-mesh.bounds[0],[.04,.08,.12],atol=1e-5))
            self.assertAlmostEqual(float(mesh.bounds[0,2]),0,places=5)
            self.assertEqual(final['assets'][name]['scale_m_per_unit'],1)
        with self.assertRaises(FileExistsError):
            geom.calibrate(args(self.root,max_shape_error=.25,allow_shape_mismatch=False))
        # Reject manual modification of an approved mesh
        with (self.root/'approved'/'button.glb').open('ab') as f:f.write(b'changed')
        geom.status(args(self.root))
        self.assertFalse(all(geom.digest(Path(final['assets'][n]['mesh']))==final['assets'][n]['sha256'] for n in final['objects']))

    def test_provenance_gate_photo(self):
        self.make_segmented()
        with (self.root/'photos'/'button.jpg').open('ab') as f:f.write(b'mutated')
        with self.assertRaises(ValueError):
            geom.verify_masks(self.root,geom.manifest(self.root),approved=True)

    def test_resegment_allowed_before_approval_and_locked_after(self):
        fake=types.ModuleType('sam2_task')
        fake.image_predictor=lambda _:object()
        fake.sam_predict=lambda predictor,frame,pos,neg:(np.ones(frame.shape[:2],dtype=np.uint8),.75)
        with patch.dict(sys.modules,{'sam2_task':fake}):
            geom.segment(args(self.root,checkpoint=Path('/dev/null')))
            geom.segment(args(self.root,checkpoint=Path('/dev/null')))
            geom.approve_masks(args(self.root,reviewer='tester'))
            with self.assertRaises(FileExistsError):
                geom.segment(args(self.root,checkpoint=Path('/dev/null')))

    def test_shape_mismatch_cannot_be_silently_fixed(self):
        self.make_segmented();self.make_raws()
        geom.inspect_meshes(args(self.root))
        spec=geom.read_json(self.root/'calibration.json')
        for name in spec['objects']:
            spec['objects'][name]['measurements_m']={'x':.04,'y':.5}
        geom.save_json(self.root/'calibration.json',spec)
        with self.assertRaisesRegex(ValueError,'shape inconsistent'):
            geom.calibrate(args(self.root,max_shape_error=.25,allow_shape_mismatch=False))

    def test_rotated_scale_and_origin(self):
        self.make_segmented();self.make_raws()
        geom.inspect_meshes(args(self.root))
        spec=geom.read_json(self.root/'calibration.json')
        for name in spec['objects']:
            spec['objects'][name]['rotation_xyz_deg']=[90,0,0]
            spec['objects'][name]['measurements_m']={'x':.04,'y':.12,'z':.08}
            spec['objects'][name]['origin']='bbox_center'
        geom.save_json(self.root/'calibration.json',spec)
        geom.calibrate(args(self.root,max_shape_error=.25,allow_shape_mismatch=False))
        mesh=geom.scene_load(self.root/'calibrated'/'box.glb')
        self.assertTrue(np.allclose(mesh.bounds[1]-mesh.bounds[0],[.04,.12,.08],atol=1e-5))
        self.assertTrue(np.allclose(mesh.bounds.mean(axis=0),[0,0,0],atol=1e-5))


if __name__=='__main__':
    unittest.main()
