"""Offline contract for live LeRobot raw packet receiver and shared depth registration."""
import io
import json
import threading
from http.server import HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from scripts.deploy.receive_live_packets import (
    PacketReceiver,construct_training_packet,make_http_handler
)


def _raw(i=0):
    return {
        "front_rgb":np.zeros((480,640,3),dtype=np.uint8),
        "left_rgb":np.zeros((480,640,3),dtype=np.uint8),
        "right_rgb":np.zeros((480,640,3),dtype=np.uint8),
        "front_depth_native":np.ones((480,640),dtype=np.uint16)*1000,
        "robot_state":np.zeros(1972,dtype=np.float32),
        "timestamp_s":np.asarray(i/15.),
        "frame_index":np.asarray(i),
    }


def _register(depth,hw):
    assert depth.dtype==np.uint16
    assert hw==(480,640)
    return depth.astype(np.float32)*0.001


def test_receiver_reuses_training_depth_and_keeps_native_exact(tmp_path):
    receiver=PacketReceiver(tmp_path/"packets",register_depth=_register,depth_scale=0.001)
    result=receiver.accept(_raw())
    assert result["no_robot_action"]
    with np.load(tmp_path/"packets"/"frame_000000.npz",allow_pickle=False) as packet:
        np.testing.assert_array_equal(packet["front_depth_native"],_raw()["front_depth_native"])
        np.testing.assert_allclose(packet["front_depth_color_aligned"],1000)
        assert packet["robot_state"].shape==(1972,)
    with pytest.raises(ValueError,match="Noncontiguous"):
        receiver.accept(_raw(2))
    assert sorted((tmp_path/"packets").glob("frame_*.npz"))[0].name=="frame_000000.npz"


def test_bad_training_depth_contract_rejected(tmp_path):
    bad=_raw()
    bad["front_depth_native"]=np.zeros((480,640),dtype=np.float32)
    with pytest.raises(ValueError,match="native depth"):
        construct_training_packet(bad,register_depth=_register,depth_scale=.001)
    bad=_raw()
    bad["robot_state"]=np.zeros(1973,dtype=np.float32)
    with pytest.raises(ValueError,match="1972"):
        construct_training_packet(bad,register_depth=_register,depth_scale=.001)


def test_http_transport_authenticated_writes_only_packets(tmp_path):
    token="testing-sensor-only-token"
    receiver=PacketReceiver(tmp_path/"receiver",register_depth=_register,depth_scale=.001)
    server=HTTPServer(("127.0.0.1",0),make_http_handler(receiver,token))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        bio=io.BytesIO()
        np.savez_compressed(bio,**_raw(0))
        addr=f"http://127.0.0.1:{server.server_port}/frame"
        wrong=Request(addr,data=bio.getvalue(),method="POST",
                      headers={"X-Live-Packet-Token":"wrong"})
        with pytest.raises(HTTPError) as err:
            urlopen(wrong,timeout=3)
        assert err.value.code==403
        assert receiver.count==0
        req=Request(addr,data=bio.getvalue(),method="POST",
                    headers={"X-Live-Packet-Token":token})
        with urlopen(req,timeout=3) as res:
            ack=json.loads(res.read().decode())
        assert ack["accepted_frame"]==0
        assert ack["no_robot_action"]
        assert (tmp_path/"receiver"/"frame_000000.npz").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
