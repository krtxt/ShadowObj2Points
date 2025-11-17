import json
import os
import random
import warnings
import typing

import mujoco
import numpy as np
import pytorch3d
import pytorch_kinematics as pk
import torch
import torch.nn
import trimesh as tm
import trimesh.sample
import transforms3d
import urdf_parser_py.urdf as URDF_PARSER
from plotly import graph_objects as go
from pytorch3d.ops import knn_points
from pytorch_kinematics.urdf_parser_py.urdf import URDF, Box, Cylinder, Mesh, Sphere
from torchsdf import compute_sdf, index_vertices_by_faces

from .rotation_spec import get_rotation_spec
from .rot6d import *


ANCHOR_ALIASES = {'base_rz180_offset': 'mjcf'}
MJCF_ANCHOR_TRANSLATION = torch.tensor([0.0, -0.01, 0.213], dtype=torch.float32)
PALM_CENTER_ANCHOR_TRANSLATION = torch.tensor([0.008, -0.013, 0.283], dtype=torch.float32)

JOINT_KEYPOINTS_STATIC_SHADOWHAND = {
    "forearm": [[0.0, -0.01, 0.213]],
    "wrist": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.034]],
    "palm": [[0.0, 0.0, 0.0], [0.033, 0.0, 0.095], [0.011, 0.0, 0.099], [-0.011, 0.0, 0.095], [-0.033, 0.0, 0.02071], [0.034, -0.0085, 0.029]],
    "ffknuckle": [[0.0, 0.0, 0.0]],
    "ffproximal": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.045]],
    "ffmiddle": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.025]],
    "ffdistal": [[0, 0, 0.024]],
    "fftip": [],
    "mfknuckle": [[0.0, 0.0, 0.0]],
    "mfproximal": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.045]],
    "mfmiddle": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.025]],
    "mfdistal": [[0, 0, 0.024]],
    "mftip": [],
    "rfknuckle": [[0.0, 0.0, 0.0]],
    "rfproximal": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.045]],
    "rfmiddle": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.025]],
    "rfdistal": [[0, 0, 0.024]],
    "rftip": [],
    "lfmetacarpal": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.06579]],
    "lfknuckle": [[0.0, 0.0, 0.0]],
    "lfproximal": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.045]],
    "lfmiddle": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.025]],
    "lfdistal": [[0, 0, 0.024]],
    "lftip": [],
    "thbase": [[0.0, 0.0, 0.0]],
    "thproximal": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.038]],
    "thhub": [[0.0, 0.0, 0.0]],
    "thmiddle": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.032]],
    "thdistal": [[0, 0, 0.026]],
    "thtip": []
}


class HandModel:
    SUPPORTED_ANCHORS = ('base', 'palm_center', 'mjcf')

    def __init__(self, robot_name, urdf_filename, mesh_path,
                 batch_size=1,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 hand_scale=1.,
                 rot_type='quat',
                 anchor='base',
                 mesh_source: str = 'urdf'
                 ):
        self.device = device
        self.robot_name = robot_name
        self.batch_size = batch_size
        self.rot_type = rot_type  # 旋转表示类型：'quat', 'r6d', 'euler', 'axis'
        # 平移锚点语义：'base'、'palm_center' 或新的 'mjcf'
        self.anchor = self._normalize_anchor(anchor)
        
        # 缓存 RotationSpec（使用 RotationSpec 设计模式）
        self.rot_spec = get_rotation_spec(rot_type)
        
        # 固定 numpy 与 Python random 的种子，确保表面采样可复现
        np.random.seed(0)
        random.seed(0)
        
        # 验证 rot_type 参数
        if self.rot_type not in ['quat', 'r6d', 'euler', 'axis']:
            raise ValueError(f"rot_type must be one of ['quat', 'r6d', 'euler', 'axis'], got '{self.rot_type}'")
        # prepare model
        self.robot = pk.build_chain_from_urdf(open(urdf_filename).read()).to(dtype=torch.float, device=self.device)
        self.robot_full = URDF_PARSER.URDF.from_xml_file(urdf_filename)
        # prepare contact point basis and surface point samples
        # self.no_contact_dict = json.load(open(os.path.join('data', 'urdf', 'intersection_%s.json'%robot_name)))
        self.dis_key_point = {
        "palm": [],
        "ffproximal": [[-0.0002376327756792307, -0.009996689856052399, 0.038666076958179474], [-0.0035445429384708405, -0.009337972849607468, 4.728326530312188e-05], [0.0042730518616735935, -0.0090293288230896, 0.018686404451727867], [-0.003900623880326748, -0.009198302403092384, 0.027359312400221825], [-0.0034948040265589952, -0.009357482194900513, 0.011004473082721233], [0.004485304467380047, -0.008933592587709427, 0.005608899053186178], [0.00421907939016819, -0.009053671732544899, 0.030992764979600906], [-0.003979427739977837, -0.009167392738163471, 0.01910199038684368], [-0.0037133553996682167, -0.009271756745874882, 0.04499374330043793], [0.0034797703847289085, -0.009367110207676888, 0.044556595385074615]],
        "ffmiddle": [[-0.0019831678364425898, -0.007794334553182125, 0.009099956601858139], [0.0017110002227127552, -0.007856125012040138, 0.024990297853946686], [0.003553177695721388, -0.007216729689389467, 0.0004225552547723055], [0.003431637305766344, -0.007271687965840101, 0.01607479713857174], [-0.0025954796001315117, -0.007619044743478298, 0.0195100586861372], [-0.0028260457329452038, -0.007528608664870262, 0.0024113801773637533], [0.003367392346262932, -0.007300738710910082, 0.0064179981127381325], [-0.0014348605182021856, -0.007911290973424911, 0.014330973848700523], [0.0024239378981292248, -0.0076669249683618546, 0.011192334815859795], [-0.003152574645355344, -0.007400532253086567, 0.02429058402776718]],
        "ffdistal": [[-0.00094795529730618, -0.006982842925935984, 0.01811189576983452], [0.002626439556479454, -0.006539319641888142, 4.996722418582067e-05], [-0.0034360431600362062, -0.00615164078772068, 0.008426538668572903], [0.002973517868667841, -0.006382592022418976, 0.011918909847736359], [0.0026527668815106153, -0.006527431774884462, 0.02366715297102928], [-0.0034155123867094517, -0.006162168458104134, 0.0019269119948148727], [-0.0033331606537103653, -0.006204396951943636, 0.023483257740736008], [0.0025241682305932045, -0.0065797604620456696, 0.005907486192882061], [-0.00349381472915411, -0.006122016813606024, 0.0137711763381958], [0.0031557006295770407, -0.006300325505435467, 0.01670246012508869]],
        "mfproximal": [[-0.0020669603254646063, -0.009777018800377846, 0.006492570973932743], [0.00465710973367095, -0.00884400587528944, 0.04495971277356148], [-0.004878615960478783, -0.008722265250980854, 0.027143988758325577], [0.005264972802251577, -0.008492529392242432, 0.017509466037154198], [0.005084634758532047, -0.008596803992986679, 0.032350800931453705], [-0.004348081536591053, -0.008994411677122116, 0.0383140966296196], [-0.004989673383533955, -0.0086652971804142, 0.01689181476831436], [0.005276953335851431, -0.008485602214932442, 0.0004979652003385127], [0.0050704991444945335, -0.008604977279901505, 0.009669311344623566], [0.0027988858055323362, -0.00959551241248846, 0.024868451058864594]],
        "mfmiddle": [[-0.00042295613093301654, -0.008031148463487625, 0.011103776283562183], [0.003634985536336899, -0.007179737091064453, 0.02486497163772583], [0.0035494803451001644, -0.007218401413410902, 0.00027021521236747503], [-0.003957465291023254, -0.007006912492215633, 0.019527770578861237], [-0.003908041398972273, -0.007031631655991077, 0.003815919626504183], [0.0035529686138033867, -0.007216824218630791, 0.017309214919805527], [0.003624177537858486, -0.007184624206274748, 0.006498508155345917], [-0.0026851133443415165, -0.007583887316286564, 0.024684462696313858], [-0.003890471300110221, -0.007041062694042921, 0.014749204739928246], [0.00044322473695501685, -0.008030962198972702, 0.02091158926486969]],
        "mfdistal": [[0.0002320836065337062, -0.007037499453872442, 0.023355133831501007], [0.0036455930676311255, -0.006024540401995182, 6.64829567540437e-05], [-0.0032930555753409863, -0.006224961951375008, 0.010883713141083717], [0.004127402324229479, -0.005703427363187075, 0.015460449270904064], [-0.003456553677096963, -0.006141123361885548, 0.002992440015077591], [0.003985205665230751, -0.005805530119687319, 0.00806250236928463], [-0.0035007710102945566, -0.0061184498481452465, 0.017718486487865448], [0.004287987481802702, -0.00558812078088522, 0.02120518684387207], [0.0011520618572831154, -0.006951729767024517, 0.004545787815004587], [0.0013482635840773582, -0.006914287339895964, 0.011958128772675991]],
        "rfproximal": [[-0.002791694598272443, -0.009589731693267822, 0.015829697251319885], [0.003399983746930957, -0.009393873624503613, 0.04477155953645706], [0.003595913527533412, -0.009328149259090424, 0.0003250864101573825], [-0.0048502604477107525, -0.008736810646951199, 0.031506575644016266], [0.004417791962623596, -0.008964043110609055, 0.024784449487924576], [-0.004430862609297037, -0.008951948024332523, 0.006260100286453962], [0.004398377146571875, -0.008972801268100739, 0.03525833785533905], [0.00454053096473217, -0.008908682502806187, 0.009322012774646282], [-0.00461477879434824, -0.008857605047523975, 0.04111073166131973], [-0.00443872157484293, -0.008947916328907013, 0.023684965446591377]],
        "rfmiddle": [[-0.0024211457930505276, -0.007671383209526539, 0.003983666189014912], [0.002596878679469228, -0.007608974818140268, 0.024917811155319214], [-0.003061287570744753, -0.007436338346451521, 0.015137670561671257], [0.0027961833402514458, -0.007542191073298454, 0.009979978203773499], [0.0029241566080600023, -0.007499308791011572, 0.01832345686852932], [0.0027976972050964832, -0.007541683502495289, 6.793846841901541e-05], [-0.0028554939199239016, -0.007517057936638594, 0.021669354289770126], [-0.00278903404250741, -0.007543126121163368, 0.009370749816298485], [0.002888968912884593, -0.007511099800467491, 0.005062070209532976], [0.0010934327729046345, -0.007965755648911, 0.013925164006650448]],
        "rfdistal": [[0.004119039047509432, -0.005709432996809483, 0.022854819893836975], [-0.004941829480230808, -0.005017726682126522, 2.3880027583800256e-05], [0.005020809359848499, -0.0049394648522138596, 0.009076559916138649], [-0.0051115998066961765, -0.0048520066775381565, 0.014685478061437607], [0.004567775409668684, -0.00535866804420948, 2.354436037421692e-05], [-0.004747811239212751, -0.005207117181271315, 0.023783991113305092], [-0.0021952472161501646, -0.006697545759379864, 0.006976036354899406], [0.0026042358949780464, -0.006549346260726452, 0.015848159790039062], [-0.0018935244297608733, -0.0067823501303792, 0.019202016294002533], [-0.00021895798272453249, -0.007044903934001923, 0.0015841316198930144]],
        "lfmetacarpal": [],
        "lfproximal": [[-0.001103847287595272, -0.009932574816048145, 0.023190606385469437], [0.004271919839084148, -0.009029388427734375, 0.00020035798661410809], [0.0033563950564712286, -0.009408160112798214, 0.04472788795828819], [-0.00359934801235795, -0.009316868148744106, 0.010634070262312889], [-0.0028604455292224884, -0.009570563212037086, 0.0347319096326828], [0.004299539607018232, -0.009016930125653744, 0.01569746620953083], [-0.0040543111972510815, -0.009138413704931736, 0.0022569862194359303], [0.004445703700184822, -0.008951003663241863, 0.03112208843231201], [0.003617372363805771, -0.009320616722106934, 0.007815317250788212], [-0.003905154298990965, -0.009196918457746506, 0.04234839603304863]],
        "lfmiddle": [[-0.0007557488279417157, -0.00800648145377636, 0.006158251781016588], [0.003424291731789708, -0.007274557836353779, 0.024703215807676315], [-0.004018992651253939, -0.006975765340030193, 0.01652413047850132], [0.003509017638862133, -0.007236245553940535, 0.013466663658618927], [-0.0038151403423398733, -0.007080549374222755, 0.023723633959889412], [0.0034917076118290424, -0.007244073320180178, 0.0006077417056076229], [-0.003889185143634677, -0.0070418380200862885, 0.00041095237247645855], [0.0013372180983424187, -0.007935309782624245, 0.019177529960870743], [-0.0038003893569111824, -0.0070875901728868484, 0.010691785253584385], [0.0034656336065381765, -0.007255863398313522, 0.008516711182892323]],
        "lfdistal": [[0.0014511797344312072, -0.006890428718179464, 0.004574548453092575], [-0.0024943388998508453, -0.006586590316146612, 0.023863397538661957], [0.0029716803692281246, -0.006382519379258156, 0.014716518111526966], [-0.002874345052987337, -0.006437260191887617, 0.010602368041872978], [-0.0028133057057857513, -0.006461246870458126, 0.00012360091204755008], [0.00285350508056581, -0.006435882765799761, 0.020840927958488464], [-0.002667121822014451, -0.006518691778182983, 0.017100241035223007], [0.0029705220367759466, -0.006383041851222515, 0.009500452317297459], [0.002596389502286911, -0.006551986560225487, 0.00017241919704247266], [-0.002902866108343005, -0.006426053121685982, 0.0050438339821994305]],
        "thbase": [],
        "thproximal": [],
        "thhub": [],
        "thmiddle": [[-0.010736164636909962, -0.0023433465976268053, 0.005364177282899618], [-0.009576565586030483, 0.005389457568526268, 0.031773995608091354], [-0.010324702598154545, -0.0037935571745038033, 0.020191052928566933], [-0.009223436936736107, 0.005977252032607794, 0.0133969122543931], [-0.008859770372509956, 0.006510394625365734, 0.0007552475435659289], [-0.010023333132266998, -0.004508777987211943, 0.03042592667043209], [-0.009238336235284805, 0.005955408792942762, 0.022636638954281807], [-0.010435031726956367, -0.0034471736289560795, 0.012765333987772465], [-0.010987512767314911, 0.0005528760375455022, 0.026306135579943657], [-0.010371722280979156, 0.0036525875329971313, 0.007199263200163841]],
        "thdistal": [[-0.00094795529730618, -0.006982842925935984, 0.01811189576983452], [0.002626439556479454, -0.006539319641888142, 4.996722418582067e-05], [-0.0034360431600362062, -0.00615164078772068, 0.008426538668572903], [0.002973517868667841, -0.006382592022418976, 0.011918909847736359], [0.0026527668815106153, -0.006527431774884462, 0.02366715297102928], [-0.0034155123867094517, -0.006162168458104134, 0.0019269119948148727], [-0.0033331606537103653, -0.006204396951943636, 0.023483257740736008], [0.0025241682305932045, -0.0065797604620456696, 0.005907486192882061], [-0.00349381472915411, -0.006122016813606024, 0.0137711763381958], [0.0031557006295770407, -0.006300325505435467, 0.01670246012508869]]
        }
        self.keypoints = {
            "forearm": [],
            "wrist": [],
            "palm": [],
            "ffknuckle": [],
            "ffproximal": [[0, 0, 0.024]],
            "ffmiddle": [[0, 0, 0], [0, 0, 0.025]],
            "ffdistal": [[0, 0, 0.024]],
            "fftip": [],
            "mfknuckle": [],
            "mfproximal": [[0, 0, 0.024]], 
            "mfmiddle": [[0, 0, 0], [0, 0, 0.025]],
            "mfdistal": [[0, 0, 0.024]],
            "mftip":[],
            "rfknuckle": [],
            "rfproximal": [[0, 0, 0.024]], 
            "rfmiddle": [[0, 0, 0], [0, 0, 0.025]],
            "rfdistal": [[0, 0, 0.024]],
            "lfmetacarpal": [],
            "lfknuckle": [],
            "lfproximal": [[0, 0, 0.024]],
            "lfmiddle": [[0, 0, 0], [0, 0, 0.025]],
            "lfdistal": [[0, 0, 0.024]],
            "lftip": [],
            "thbase": [], 
            "thproximal": [[0, 0, 0.038]], 
            "thhub": [],
            "thmiddle": [[0, 0, 0.032]], 
            "thdistal": [[0, 0, 0.026]],
            "thtip":[]
        }
        all_links = [lk.name for lk in getattr(self.robot_full, 'links', [])] if hasattr(self.robot_full, 'links') else []
        d = {k: [] for k in all_links}
        if self.robot_name == 'shadowhand':
            for k, v in JOINT_KEYPOINTS_STATIC_SHADOWHAND.items():
                if k in d and isinstance(v, list):
                    d[k] = [list(p) for p in v]
        self.joint_key_points = d
        self.link_face_verts = {}
        # prepare geometries for visualization
        self.global_translation = None
        self.global_rotation = None
        self.softmax = torch.nn.Softmax(dim=-1)
        # prepare contact point basis and surface point samples
        self.surface_points = {}
        self.surface_points_normal = {}
        visual = URDF.from_xml_string(open(urdf_filename).read())
        self.mesh_verts = {}
        self.mesh_faces = {}

        self.canon_verts = []
        self.canon_faces = []
        self.idx_vert_faces = []
        self.face_normals = []

        if robot_name == 'shadowhand':
            self.palm_toward = torch.tensor([0., -1., 0., 0.], device=self.device).reshape(1, 1, 4).repeat(self.batch_size, 1, 1)
        else:
            raise NotImplementedError

        if mesh_source == 'urdf':
            self._load_meshes_from_urdf(visual=visual, mesh_path=mesh_path, device=device, batch_size=batch_size, robot_name=robot_name)
        elif mesh_source == 'mujoco':
            # Build meshes per body by aggregating MuJoCo geoms (same blocks as hand_util)
            if robot_name != 'shadowhand':
                raise NotImplementedError("mujoco mesh_source currently supports 'shadowhand' only")

            xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets/hand/shadow/right_hand.xml')
            xml_dir = os.path.dirname(os.path.abspath(xml_path))

            spec = mujoco.MjSpec()
            cwd = os.getcwd()
            try:
                os.chdir(xml_dir)
                with open(xml_path, 'r') as f:
                    spec.from_string(f.read())
                mj_model = spec.compile()
            finally:
                os.chdir(cwd)
            mj_data = mujoco.MjData(mj_model)
            mujoco.mj_forward(mj_model, mj_data)

            body_to_vertices = {}
            body_to_faces = {}
            body_to_vert_count = {}

            for geom_id in range(mj_model.ngeom):
                geom = mj_model.geom(geom_id)
                mesh_id = geom.dataid
                if mesh_id == -1:
                    continue
                mjm = mj_model.mesh(mesh_id)
                v_start = mjm.vertadr[0]
                v_count = mjm.vertnum[0]
                f_start = mjm.faceadr[0]
                f_count = mjm.facenum[0]
                vert = mj_model.mesh_vert[v_start: v_start + v_count]
                face = mj_model.mesh_face[f_start: f_start + f_count]

                geom_rot = mj_data.geom_xmat[geom_id].reshape(3, 3)
                geom_trans = mj_data.geom_xpos[geom_id]

                body_id = geom.bodyid
                body_name = mj_model.body(body_id).name
                body_rot = mj_data.xmat[body_id].reshape(3, 3)
                body_trans = mj_data.xpos[body_id]

                v_world = vert @ geom_rot.T + geom_trans
                v_body = (v_world - body_trans) @ body_rot

                if body_name not in body_to_vertices:
                    body_to_vertices[body_name] = []
                    body_to_faces[body_name] = []
                    body_to_vert_count[body_name] = 0

                vert_offset = body_to_vert_count[body_name]
                body_to_vertices[body_name].append(v_body)
                body_to_faces[body_name].append(face + vert_offset)
                body_to_vert_count[body_name] += v_body.shape[0]

            # Finalize per body
            for body_name in body_to_vertices:
                verts = np.concatenate(body_to_vertices[body_name], axis=0) if len(body_to_vertices[body_name]) > 0 else np.zeros((0, 3))
                faces = np.concatenate(body_to_faces[body_name], axis=0) if len(body_to_faces[body_name]) > 0 else np.zeros((0, 3), dtype=np.int32)
                self.mesh_verts[body_name] = verts
                self.mesh_faces[body_name] = faces

                # Surface points sampling for this body
                if verts.shape[0] > 0 and faces.shape[0] > 0:
                    mesh = tm.Trimesh(vertices=verts, faces=faces, process=False)
                    sample_count = 64 if self.robot_name == 'shadowhand' else 128
                    try:
                        pts, pts_face_index = tm.sample.sample_surface(mesh=mesh, count=sample_count)
                        pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)
                    except Exception:
                        pts = np.zeros((0, 3), dtype=float)
                        pts_normal = np.zeros((0, 3), dtype=float)
                else:
                    pts = np.zeros((0, 3), dtype=float)
                    pts_normal = np.zeros((0, 3), dtype=float)

                pts_h = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1) if len(pts) > 0 else np.zeros((0, 4), dtype=float)
                pts_normal_h = np.concatenate([pts_normal, np.ones([len(pts_normal), 1])], axis=-1) if len(pts_normal) > 0 else np.zeros((0, 4), dtype=float)
                self.surface_points[body_name] = torch.from_numpy(pts_h).to(device).float().unsqueeze(0).repeat(batch_size, 1, 1)
                self.surface_points_normal[body_name] = torch.from_numpy(pts_normal_h).to(device).float().unsqueeze(0).repeat(batch_size, 1, 1)

                link_vertices = torch.tensor(self.mesh_verts[body_name], dtype=torch.float)
                link_faces = torch.tensor(self.mesh_faces[body_name], dtype=torch.long) if self.mesh_faces[body_name].size > 0 else torch.zeros((0, 3), dtype=torch.long)
                if link_vertices.numel() > 0 and link_faces.numel() > 0:
                    self.link_face_verts[body_name] = index_vertices_by_faces(link_vertices, link_faces).to(device).float()
                else:
                    self.link_face_verts[body_name] = torch.zeros((0, 3, 3), dtype=torch.float, device=device)
        else:
            raise ValueError(f"Unknown mesh_source: {mesh_source}")

        # new 2.1
        self.revolute_joints = []
        for i in range(len(self.robot_full.joints)):
            if self.robot_full.joints[i].joint_type == 'revolute':
                self.revolute_joints.append(self.robot_full.joints[i])
        self.revolute_joints_q_mid = []
        self.revolute_joints_q_var = []
        self.revolute_joints_q_upper = []
        self.revolute_joints_q_lower = []
        for i in range(len(self.robot.get_joint_parameter_names())):
            for j in range(len(self.revolute_joints)):
                if self.revolute_joints[j].name == self.robot.get_joint_parameter_names()[i]:
                    joint = self.revolute_joints[j]
            assert joint.name == self.robot.get_joint_parameter_names()[i]
            self.revolute_joints_q_mid.append(
                (joint.limit.lower + joint.limit.upper) / 2)
            self.revolute_joints_q_var.append(
                ((joint.limit.upper - joint.limit.lower) / 2) ** 2)
            self.revolute_joints_q_lower.append(joint.limit.lower)
            self.revolute_joints_q_upper.append(joint.limit.upper)

        self.revolute_joints_q_lower = torch.Tensor(
            self.revolute_joints_q_lower).repeat([self.batch_size, 1]).to(device)
        self.revolute_joints_q_upper = torch.Tensor(
            self.revolute_joints_q_upper).repeat([self.batch_size, 1]).to(device)

        self.current_status = None

        self.scale = hand_scale

    def _map_to_current_status_name(self, name: str) -> typing.Optional[str]:
        """Map mesh/link name to a valid key in self.current_status.
        Handles differences between MuJoCo body names (e.g., 'rh_palm') and
        URDF/PK link names (e.g., 'palm').
        """
        try:
            cs_keys = list(self.current_status.keys()) if self.current_status is not None else []
        except Exception:
            cs_keys = []
        if not cs_keys:
            return None
        if name in self.current_status:
            return name

        def normalize(s: str) -> str:
            s = s.lower()
            # remove common prefixes and separators
            for pre in ["rh_", "lh_", "right_", "left_"]:
                if s.startswith(pre):
                    s = s[len(pre):]
            s = s.replace("-", "").replace("_", "")
            return s

        # Build normalized index for current_status keys
        norm_to_key = {}
        for k in cs_keys:
            norm_to_key[normalize(k)] = k

        # Try direct normalization match
        cand = normalize(name)
        if cand in norm_to_key:
            return norm_to_key[cand]

        # Try dropping the token before first underscore, e.g., 'rh_palm' -> 'palm'
        if "_" in name:
            tail = name.split("_", 1)[1]
            cand2 = normalize(tail)
            if cand2 in norm_to_key:
                return norm_to_key[cand2]

        return None

    

    def save_joint_keypoints_to_file(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True) if os.path.dirname(file_path) else None
        out = {
            'robot_name': self.robot_name,
            'joint_key_points': self.joint_key_points,
        }
        with open(file_path, 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    def set_joint_keypoints(self, dct: dict, overwrite: bool = True) -> None:
        all_links = [lk.name for lk in getattr(self.robot_full, 'links', [])] if hasattr(self.robot_full, 'links') else []
        if overwrite or not isinstance(self.joint_key_points, dict):
            base = {k: [] for k in all_links}
        else:
            base = {k: [list(p) for p in self.joint_key_points.get(k, [])] for k in all_links}
        for k, v in (dct.items() if isinstance(dct, dict) else []):
            if k not in base or not isinstance(v, list):
                continue
            seen = set((float(p[0]), float(p[1]), float(p[2])) for p in base[k])
            for p in v:
                if not isinstance(p, (list, tuple)) or len(p) != 3:
                    continue
                t = (float(p[0]), float(p[1]), float(p[2]))
                if t in seen:
                    continue
                seen.add(t)
                base[k].append([t[0], t[1], t[2]])
        self.joint_key_points = base

    def _load_meshes_from_urdf(self, visual: URDF, mesh_path: str, device, batch_size: int, robot_name: str) -> None:
        for i_link, link in enumerate(visual.links):
            # print(f"Processing link #{i_link}: {link.name}")
            # load mesh
            if len(link.visuals) == 0:
                continue
            if type(link.visuals[0].geometry) == Mesh:
                # print(link.visuals[0])
                if robot_name == 'shadowhand' or robot_name == 'allegro' or robot_name == 'barrett':
                    filename = link.visuals[0].geometry.filename.split('/')[-1]
                elif robot_name == 'allegro':
                    filename = f"{link.visuals[0].geometry.filename.split('/')[-2]}/{link.visuals[0].geometry.filename.split('/')[-1]}"
                else:
                    filename = link.visuals[0].geometry.filename
                mesh = tm.load(os.path.join(mesh_path, filename), force='mesh', process=False)
            elif type(link.visuals[0].geometry) == Cylinder:
                mesh = tm.primitives.Cylinder(
                    radius=link.visuals[0].geometry.radius, height=link.visuals[0].geometry.length)
            elif type(link.visuals[0].geometry) == Box:
                mesh = tm.primitives.Box(extents=link.visuals[0].geometry.size)
            elif type(link.visuals[0].geometry) == Sphere:
                mesh = tm.primitives.Sphere(
                    radius=link.visuals[0].geometry.radius)
            else:
                print(type(link.visuals[0].geometry))
                raise NotImplementedError
            try:
                scale = np.array(
                    link.visuals[0].geometry.scale).reshape([1, 3])
            except:
                scale = np.array([[1, 1, 1]])
            try:
                rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
                translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])
                # print('---')
                # print(link.visuals[0].origin.rpy, rotation)
                # print('---')
            except AttributeError:
                rotation = transforms3d.euler.euler2mat(0, 0, 0)
                translation = np.array([[0, 0, 0]])

            # Surface point
            # mesh.sample(int(mesh.area * 100000)) * scale
            if self.robot_name == 'shadowhand':
                pts, pts_face_index = trimesh.sample.sample_surface(mesh=mesh, count=64)
                pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)
            else:
                pts, pts_face_index = trimesh.sample.sample_surface(mesh=mesh, count=128)
                pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)

            if self.robot_name == 'barrett':
                if link.name in ['bh_base_link']:
                    pts = trimesh.sample.volume_mesh(mesh=mesh, count=1024)
                    pts_normal = np.array([[0., 0., 1.] for x in range(pts.shape[0])], dtype=float)
            if self.robot_name == 'ezgripper':
                if link.name in ['left_ezgripper_palm_link']:
                    pts = trimesh.sample.volume_mesh(mesh=mesh, count=1024)
                    pts_normal = np.array([[1., 0., 0.] for x in range(pts.shape[0])], dtype=float)
            if self.robot_name == 'robotiq_3finger':
                if link.name in ['gripper_palm']:
                    pts = trimesh.sample.volume_mesh(mesh=mesh, count=1024)
                    pts_normal = np.array([[0., 0., 1.] for x in range(pts.shape[0])], dtype=float)

            pts *= scale
            # pts = mesh.sample(128) * scale
            # print(link.name, len(pts))
            # new
            if robot_name == 'shadowhand':
                pts = pts[:, [0, 2, 1]]
                pts_normal = pts_normal[:, [0, 2, 1]]
                pts[:, 1] *= -1
                pts_normal[:, 1] *= -1

            pts = np.matmul(rotation, pts.T).T + translation
            # pts_normal = np.matmul(rotation, pts_normal.T).T
            pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
            pts_normal = np.concatenate([pts_normal, np.ones([len(pts_normal), 1])], axis=-1)
            self.surface_points[link.name] = torch.from_numpy(pts).to(
                device).float().unsqueeze(0).repeat(batch_size, 1, 1)
            self.surface_points_normal[link.name] = torch.from_numpy(pts_normal).to(
                device).float().unsqueeze(0).repeat(batch_size, 1, 1)

            # visualization mesh
            self.mesh_verts[link.name] = np.array(mesh.vertices) * scale
            if robot_name == 'shadowhand':
                self.mesh_verts[link.name] = self.mesh_verts[link.name][:, [0, 2, 1]]
                self.mesh_verts[link.name][:, 1] *= -1
            self.mesh_verts[link.name] = np.matmul(rotation, self.mesh_verts[link.name].T).T + translation
            self.mesh_faces[link.name] = np.array(mesh.faces)
            link_vertices = torch.tensor(self.mesh_verts[link.name], dtype=torch.float)
            link_faces = torch.tensor(self.mesh_faces[link.name], dtype=torch.long)
            self.link_face_verts[link.name] = index_vertices_by_faces(link_vertices, link_faces).to(device).float()

    def _compute_palm_center_local_from_status(self):
        """
        基于当前关节状态（self.current_status）计算手掌中心在手根坐标系下的坐标（不含全局刚体与缩放）。
        返回形状：[B, 3]
        """
        # Find palm mesh name (could be 'palm' or 'rh_palm' etc.)
        palm_mesh_name = None
        for key in self.surface_points.keys():
            if 'palm' in key.lower() and 'metacarpal' not in key.lower():
                palm_mesh_name = key
                break
        
        if palm_mesh_name is None:
            raise RuntimeError("surface_points for 'palm' not initialized.")
        
        # Map mesh name to kinematics link name
        mapped_name = self._map_to_current_status_name(palm_mesh_name)
        if mapped_name is None:
            mapped_name = palm_mesh_name
        
        trans_matrix = self.current_status[mapped_name].get_matrix()  # [B, 4, 4]
        pts_h = self.surface_points[palm_mesh_name]  # [B, N, 4]（齐次坐标）
        palm_pts_local = torch.matmul(trans_matrix, pts_h.transpose(1, 2)).transpose(1, 2)[..., :3]
        palm_center_local = palm_pts_local.mean(dim=1)
        return palm_center_local

    def get_palm_center_local(self, q=None):
        """
        计算在手根坐标系下的 palm_center（不含全局刚体与缩放）。
        若提供 q（B x pose_dim），会先更新运动学再返回。
        """
        if q is not None:
            self.update_kinematics(q)
        return self._compute_palm_center_local_from_status()

    def _normalize_anchor(self, anchor: str) -> str:
        """统一 anchor 写法，并兼容旧名称。"""
        if not isinstance(anchor, str):
            raise TypeError(f"anchor must be str, got {type(anchor)}")
        normalized = anchor.strip().lower()
        alias = ANCHOR_ALIASES.get(normalized, normalized)
        if normalized in ANCHOR_ALIASES:
            warnings.warn(
                f"anchor '{normalized}' 已重命名为 '{alias}'，请尽快更新配置以避免该提示。",
                UserWarning,
            )
        if alias not in self.SUPPORTED_ANCHORS:
            raise ValueError(f"anchor must be one of {self.SUPPORTED_ANCHORS}, got '{alias}'")
        return alias

    def _apply_anchor_transform(self, t_input: torch.Tensor, R_base: torch.Tensor):
        """根据 anchor 语义设置全局刚体（translation/rotation）。"""
        anchor = self.anchor
        if anchor == 'base':
            self.global_translation = t_input
            self.global_rotation = R_base
        elif anchor == 'palm_center':
            batch = t_input.shape[0]
            t_off = PALM_CENTER_ANCHOR_TRANSLATION.to(
                device=t_input.device,
                dtype=t_input.dtype,
            ).unsqueeze(0).expand(batch, -1)
            base_offset = torch.bmm(R_base, t_off.unsqueeze(-1)).squeeze(-1)
            self.global_translation = t_input - base_offset
            self.global_rotation = R_base
        elif anchor == 'mjcf':
            batch = t_input.shape[0]
            t_off = MJCF_ANCHOR_TRANSLATION.to(
                device=t_input.device,
                dtype=t_input.dtype,
            ).unsqueeze(0).expand(batch, -1)
            base_offset = torch.bmm(R_base, t_off.unsqueeze(-1)).squeeze(-1)
            self.global_translation = t_input - base_offset
            # 目前仅做平移覆盖，旋转覆盖逻辑保留以备后用
            # R_off = torch.tensor([[-1., 0., 0.],
            #                       [ 0.,-1., 0.],
            #                       [ 0., 0., 1.]], device=t_input.device, dtype=t_input.dtype).unsqueeze(0).repeat(batch, 1, 1)
            self.global_rotation = R_base
            # self.global_rotation = torch.bmm(R_base, R_off)
        else:
            raise RuntimeError(f"Unsupported anchor: {anchor}")

    def update_kinematics(self, q):
        """
        更新手部运动学状态
        
        输入支持两种关节维度：
        A) trans(3) + qpos(24) + rot(variable)
        B) trans(3) + qpos(22) + rot(variable)  # MJCF顺序的22维，将在内部映射为URDF的24维
        - quat 模式：A=31维 / B=29维
        - r6d 模式：A=33维 / B=31维
        - euler 模式：A=30维 / B=28维
        - axis 模式：A=30维 / B=28维
        
        Args:
            q: 手部姿态张量 [B, pose_dim]
        """
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = q.to(device=self.device, dtype=torch.float32)
        try:
            self.robot = self.robot.to(dtype=q.dtype, device=q.device)
        except Exception:
            pass
        # 确定旋转表示维度
        if self.rot_type == 'quat':
            rot_dim = 4
        elif self.rot_type == 'r6d':
            rot_dim = 6
        elif self.rot_type == 'euler':
            rot_dim = 3
        elif self.rot_type == 'axis':
            rot_dim = 3
        else:
            raise ValueError(f"Unsupported rot_type: {self.rot_type}")
        
        # 验证输入维度（支持 24 或 22 维关节）
        dim_A = 3 + 24 + rot_dim  # trans + qpos(24) + rot
        dim_B = 3 + 22 + rot_dim  # trans + qpos(22) + rot
        if q.shape[1] not in (dim_A, dim_B):
            raise ValueError(
                f"Input dimension mismatch for rot_type='{self.rot_type}'. "
                f"Expected {dim_A} (3+24+{rot_dim}) or {dim_B} (3+22+{rot_dim}), "
                f"but got {q.shape[1]}."
            )
        
        # 分解输入
        t_input = q[:, :3]  # [B, 3]，其语义由 anchor 决定
        if q.shape[1] == dim_A:
            joint_angles_24 = q[:, 3:27]  # [B, 24]
            rotation_params = q[:, 27:]   # [B, rot_dim] (4/6/3/3)
        else:
            joints22 = q[:, 3:3+22]       # [B, 22] (MJCF顺序)
            rotation_params = q[:, 3+22:] # [B, rot_dim]
            # 将22维映射为URDF顺序的24维（前两维为腕关节，设为0）
            # URDF: [WRJ2, WRJ1, FFJ4, FFJ3, FFJ2, FFJ1, MFJ4, MFJ3, MFJ2, MFJ1,
            #        RFJ4, RFJ3, RFJ2, RFJ1, LFJ5, LFJ4, LFJ3, LFJ2, LFJ1,
            #        THJ5, THJ4, THJ3, THJ2, THJ1]
            B = joints22.shape[0]
            joint_angles_24 = torch.zeros((B, 24), dtype=joints22.dtype, device=joints22.device)
            joint_angles_24[:, 0] = 0.0  # WRJ2
            joint_angles_24[:, 1] = 0.0  # WRJ1
            joint_angles_24[:, 2:6] = joints22[:, 0:4]      # FFJ4..FFJ1
            joint_angles_24[:, 6:10] = joints22[:, 4:8]     # MFJ4..MFJ1
            joint_angles_24[:, 10:14] = joints22[:, 8:12]   # RFJ4..RFJ1
            joint_angles_24[:, 14:19] = joints22[:, 12:17]  # LFJ5..LFJ1
            joint_angles_24[:, 19:24] = joints22[:, 17:22]  # THJ5..THJ1
        
        # 使用 RotationSpec 统一接口转换旋转参数为旋转矩阵
        # 归一化（如果需要）
        if self.rot_spec.needs_normalization:
            rotation_params = self.rot_spec.normalize_fn(rotation_params)
        
        # 统一转换为旋转矩阵（先保存为基础旋转，便于锚点后处理）
        self.global_rotation = self.rot_spec.to_matrix_fn(rotation_params)  # [B, 3, 3]
        R_base = self.global_rotation
        
        # 执行前向运动学
        self.current_status = self.robot.forward_kinematics(joint_angles_24)
        self.current_joint_angles_24 = joint_angles_24

        # 根据锚点语义设置 global_translation / global_rotation
        self._apply_anchor_transform(t_input, R_base)

    def save_point_cloud(self, points, filename):
        point_cloud = trimesh.points.PointCloud(points)
        point_cloud.export(filename)
        print(f"Saved point cloud to {filename}")

    def save_mesh(self, face_verts, filename):
        vertices = face_verts.reshape(-1, 3)
        
        faces = np.arange(vertices.shape[0]).reshape(-1, 3)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(filename)
        print(f"Saved mesh to {filename}")

    def pen_loss_sdf(self,obj_pcd: torch.Tensor,q=None ,test = False):
        penetration = []
        if q is not None:
            self.update_kinematics(q)
        obj_pcd = obj_pcd.float()
        global_translation = self.global_translation.float()
        global_rotation = self.global_rotation.float()
        obj_pcd = (obj_pcd - global_translation.unsqueeze(1)) @ global_rotation
        # self.save_point_cloud(obj_pcd[1].detach().cpu().numpy(), f"{1}_point_cloud.ply")
        for link_name in self.link_face_verts:
            mapped_name = self._map_to_current_status_name(link_name)
            if mapped_name is None:
                continue
            trans_matrix = self.current_status[mapped_name].get_matrix()
            obj_pcd_local = (obj_pcd - trans_matrix[:, :3, 3].unsqueeze(1)) @ trans_matrix[:, :3, :3]
            obj_pcd_local = obj_pcd_local.reshape(-1, 3)
            hand_face_verts = self.link_face_verts[link_name].detach()
            # dis_local, _, dis_signs, _, _ = compute_sdf(obj_pcd_local, hand_face_verts)
            dis_local, dis_signs, _, _ = compute_sdf(obj_pcd_local, hand_face_verts)
            dis_local = torch.sqrt(dis_local + 1e-8)#eval
            penloss_sdf = dis_local * (-dis_signs)
            penetration.append(penloss_sdf.reshape(obj_pcd.shape[0], obj_pcd.shape[1]))  # (batch_size, num_samples)
            # self.save_point_cloud(obj_pcd_local.reshape(obj_pcd.shape[0], -1,3)[1].detach().cpu().numpy(), f"{link_name}_point_cloud.ply")
            # self.save_mesh(hand_face_verts.detach().cpu().numpy(), f"{link_name}_mesh.ply")
        # penetration = torch.max(torch.stack(penetration), dim=0)[0]
        # loss_pen_sdf = penetration[penetration > 0].sum() / obj_pcd.shape[0]
        if test:
            distances = torch.max(torch.stack(penetration, dim=0), dim=0)[0]
            distances[distances <= 0] = 0
            # return max(distances.max().item(), 0)
            distances = distances.max(dim=1).values

            return distances.mean()
        
        penetration = torch.stack(penetration)
        # penetration = penetration.max(dim=0)[0]
        loss = penetration[penetration > 0].sum() / (penetration.shape[0]* penetration.shape[1])# distances[distances > 0].sum() / batch_size
        # print('eval:' ,max(penetration.max().item(), 0)) ###eval
        # print('penetration_sdf: ', penetration)
        return loss
    
    def add_keypoints(self, link_name: str, points: list):
        """
        动态添加关键点到指定 link
        
        Args:
            link_name: link 名称（如 'palm', 'ffproximal' 等）
            points: 关键点列表，每个点是 [x, y, z] 格式，坐标在 link 局部坐标系中
                   例如: [[0, 0, 0.024], [0.01, 0, 0.024]]
        
        Example:
            hand_model.add_keypoints('palm', [[0, 0, 0], [0.01, 0, 0]])
            hand_model.add_keypoints('ffproximal', [[0.01, 0, 0.024]])  # 追加到现有点
        """
        if link_name not in self.keypoints:
            # 如果 link 不存在，创建新条目
            self.keypoints[link_name] = []
        # 添加新点（追加到现有点）
        if isinstance(points, list) and len(points) > 0:
            if isinstance(points[0], (list, tuple)) and len(points[0]) == 3:
                # points 是点的列表
                self.keypoints[link_name].extend([list(p) for p in points])
            elif isinstance(points, (list, tuple)) and len(points) == 3:
                # points 是单个点
                self.keypoints[link_name].append(list(points))
        else:
            raise ValueError(f"points 必须是包含 [x, y, z] 坐标的列表")
    
    def set_keypoints(self, link_name: str, points: list):
        """
        设置指定 link 的关键点（替换现有点）
        
        Args:
            link_name: link 名称
            points: 关键点列表，每个点是 [x, y, z] 格式
        
        Example:
            hand_model.set_keypoints('palm', [[0, 0, 0], [0.01, 0, 0]])
        """
        if not isinstance(points, list):
            raise ValueError(f"points 必须是列表")
        self.keypoints[link_name] = [list(p) for p in points] if len(points) > 0 and isinstance(points[0], (list, tuple)) else []
    
    def get_keypoints(self, q=None, downsample=True):
        if q is not None:
            self.update_kinematics(q)
        keypoints = []
        for link_name in self.keypoints:
            if len(self.keypoints[link_name]) == 0:
                continue
            mapped_name = self._map_to_current_status_name(link_name)
            if mapped_name is None:
                mapped_name = link_name
            if mapped_name not in self.current_status:
                continue
            pts_local = torch.tensor(self.keypoints[link_name], device=self.device, dtype=torch.float32)
            M = self.current_status[mapped_name].get_matrix()
            pts_h = torch.cat([pts_local, torch.ones((pts_local.shape[0], 1), device=self.device, dtype=pts_local.dtype)], dim=1)
            pts_h = pts_h.unsqueeze(0).expand(M.shape[0], -1, -1)
            kp = torch.matmul(M, pts_h.transpose(1, 2)).transpose(1, 2)[..., :3]
            keypoints.append(kp)
        keypoints = torch.cat(keypoints, dim=1)
        keypoints = torch.bmm(keypoints, self.global_rotation.transpose(1, 2)) + self.global_translation.unsqueeze(1)
        return keypoints* self.scale
    
    def get_joint_keypoints(self, q=None, downsample=True, deduplicate=True, eps: float = 1e-5):
        if q is not None:
            self.update_kinematics(q)
        keypoints = []
        for link_name in self.joint_key_points:
            if len(self.joint_key_points[link_name]) == 0:
                continue
            mapped_name = self._map_to_current_status_name(link_name)
            if mapped_name is None:
                mapped_name = link_name
            if mapped_name not in self.current_status:
                continue
            pts_local = torch.tensor(self.joint_key_points[link_name], device=self.device, dtype=torch.float32)
            M = self.current_status[mapped_name].get_matrix()
            pts_h = torch.cat([pts_local, torch.ones((pts_local.shape[0], 1), device=self.device, dtype=pts_local.dtype)], dim=1)
            pts_h = pts_h.unsqueeze(0).expand(M.shape[0], -1, -1)
            kp = torch.matmul(M, pts_h.transpose(1, 2)).transpose(1, 2)[..., :3]
            keypoints.append(kp)
        if len(keypoints) == 0:
            return torch.zeros((self.batch_size, 0, 3), device=self.device, dtype=torch.float32)
        keypoints = torch.cat(keypoints, dim=1)
        keypoints = torch.bmm(keypoints, self.global_rotation.transpose(1, 2)) + self.global_translation.unsqueeze(1)
        keypoints = keypoints * self.scale
        if deduplicate and self.batch_size == 1 and keypoints.shape[1] > 0:
            pts = keypoints[0]
            if eps is not None and eps > 0:
                qk = torch.round(pts / eps).to(torch.int64).detach().cpu().numpy()
            else:
                qk = pts.detach().cpu().numpy()
            _, idx = np.unique(qk, axis=0, return_index=True)
            idx = np.sort(idx)
            pts = pts[idx]
            return pts.unsqueeze(0)
        return keypoints

    def get_joint_keypoints_unique(self, q=None, eps: float = 1e-5):
        if q is not None:
            self.update_kinematics(q)
        all_pts = []
        link_offsets = {}
        cursor = 0
        for link_name in self.joint_key_points:
            pts_local = self.joint_key_points[link_name]
            if len(pts_local) == 0:
                continue
            mapped_name = self._map_to_current_status_name(link_name)
            if mapped_name is None:
                mapped_name = link_name
            if mapped_name not in self.current_status:
                continue
            kp = self.current_status[mapped_name].transform_points(
                torch.tensor(pts_local, device=self.device, dtype=torch.float32)
            ).expand(self.batch_size, -1, -1)
            all_pts.append(kp)
            link_offsets[link_name] = (cursor, len(pts_local))
            cursor += len(pts_local)
        if len(all_pts) == 0:
            return (
                torch.zeros((self.batch_size, 0, 3), device=self.device, dtype=torch.float32),
                {},
            )
        pts = torch.cat(all_pts, dim=1)
        pts = torch.bmm(pts, self.global_rotation.transpose(1, 2)) + self.global_translation.unsqueeze(1)
        pts = pts * self.scale
        if self.batch_size != 1 or pts.shape[1] == 0:
            return pts, {k: [] for k in self.joint_key_points.keys()}
        p = pts[0]
        if eps is not None and eps > 0:
            qk = torch.round(p / eps).to(torch.int64).detach().cpu().numpy()
        else:
            qk = p.detach().cpu().numpy()
        uniq_pts = []
        uniq_map = {}
        orig_to_uniq = []
        for i in range(qk.shape[0]):
            key = (int(qk[i, 0]), int(qk[i, 1]), int(qk[i, 2]))
            if key in uniq_map:
                orig_to_uniq.append(uniq_map[key])
            else:
                idx_u = len(uniq_pts)
                uniq_map[key] = idx_u
                uniq_pts.append(p[i].unsqueeze(0))
                orig_to_uniq.append(idx_u)
        if len(uniq_pts) == 0:
            xyz_u = torch.zeros((1, 0, 3), device=self.device, dtype=torch.float32)
        else:
            xyz_u = torch.cat(uniq_pts, dim=0).unsqueeze(0)
        link_to_unique = {}
        for link_name, (st, ln) in link_offsets.items():
            if ln == 0:
                link_to_unique[link_name] = []
                continue
            idxs = []
            for k in range(ln):
                idxs.append(int(orig_to_uniq[st + k]))
            link_to_unique[link_name] = idxs
        return xyz_u, link_to_unique
    
    def get_dis_keypoints(self, q=None, downsample=True):
        if q is not None:
            self.update_kinematics(q)
        dis_points = []
        for link_name in self.dis_key_point:
            if len(self.dis_key_point[link_name]) == 0:
                continue
            mapped_name = self._map_to_current_status_name(link_name)
            if mapped_name is None:
                mapped_name = link_name
            if mapped_name not in self.current_status:
                continue
            pts_local = torch.tensor(self.dis_key_point[link_name], device=self.device, dtype=torch.float32)
            M = self.current_status[mapped_name].get_matrix()
            pts_h = torch.cat([pts_local, torch.ones((pts_local.shape[0], 1), device=self.device, dtype=pts_local.dtype)], dim=1)
            pts_h = pts_h.unsqueeze(0).expand(M.shape[0], -1, -1)
            dp = torch.matmul(M, pts_h.transpose(1, 2)).transpose(1, 2)[..., :3]
            dis_points.append(dp)
        dis_points = torch.cat(dis_points, dim=1)
        dis_points = torch.bmm(dis_points, self.global_rotation.transpose(1, 2)) + self.global_translation.unsqueeze(1)
        return dis_points* self.scale

    def set_parameters(self, q: torch.Tensor) -> None:
        self.update_kinematics(q)

    def get_penetration_keypoints(self, q: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.get_dis_keypoints(q=q)

    def cal_self_penetration_energy(self) -> torch.Tensor:
        return torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)

    def cal_joint_limit_energy(self) -> torch.Tensor:
        q = getattr(self, 'current_joint_angles_24', None)
        if q is None:
            return torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
        lower = getattr(self, 'revolute_joints_q_lower', None)
        upper = getattr(self, 'revolute_joints_q_upper', None)
        if lower is None or upper is None:
            return torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
        # 对齐维度（有些 URDF 关节数可能与 24 有差异，取最小公共长度）
        L = min(q.shape[1], lower.shape[1], upper.shape[1])
        qv = q[:, :L]
        lv = lower[:, :L]
        uv = upper[:, :L]
        lower_violation = torch.relu(lv - qv)
        upper_violation = torch.relu(qv - uv)
        viol = lower_violation + upper_violation
        return viol.mean(dim=1)

    def cal_finger_finger_distance_energy(self) -> torch.Tensor:
        return torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)

    def cal_finger_palm_distance_energy(self) -> torch.Tensor:
        return torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
    

    def get_surface_points(self, q=None, downsample=True):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []

        for link_name in self.surface_points:
            # for link_name in parts:
            # get transformation
            mapped_name = self._map_to_current_status_name(link_name)
            if mapped_name is None:
                continue
            if mapped_name  in ['forearm']:
                continue
            trans_matrix = self.current_status[mapped_name].get_matrix()
            surface_points.append(
                torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(self.global_rotation.float(), surface_points.transpose(1, 2)).transpose(1,
                                                                                                      2) + self.global_translation.unsqueeze(
            1)
        # if downsample:
        #     surface_points = surface_points[:, torch.randperm(surface_points.shape[1])][:, :778]
        return surface_points * self.scale

    def get_palm_points(self, q=None):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []

        # Find palm link in surface_points (could be 'palm' or 'rh_palm' etc.)
        palm_mesh_name = None
        for key in self.surface_points.keys():
            if 'palm' in key.lower() and 'metacarpal' not in key.lower():
                palm_mesh_name = key
                break
        
        if palm_mesh_name is not None:
            # Map mesh name to kinematics link name
            mapped_name = self._map_to_current_status_name(palm_mesh_name)
            if mapped_name is None:
                mapped_name = palm_mesh_name
            trans_matrix = self.current_status[mapped_name].get_matrix()
            surface_points.append(
                torch.matmul(trans_matrix, self.surface_points[palm_mesh_name].transpose(1, 2)).transpose(1, 2)[..., :3])
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return surface_points * self.scale

    def get_palm_toward_point(self, q=None):
        if q is not None:
            self.update_kinematics(q)

        # Try to map 'palm' to actual link name
        link_name = 'palm'
        mapped_name = self._map_to_current_status_name(link_name)
        if mapped_name is None:
            mapped_name = link_name
        trans_matrix = self.current_status[mapped_name].get_matrix()
        palm_toward_point = torch.matmul(trans_matrix, self.palm_toward.transpose(1, 2)).transpose(1, 2)[..., :3]
        palm_toward_point = torch.matmul(self.global_rotation, palm_toward_point.transpose(1, 2)).transpose(1, 2)

        return palm_toward_point.squeeze(1)

    def get_palm_center_and_toward(self, q=None):
        if q is not None:
            self.update_kinematics(q)

        palm_surface_points = self.get_palm_points()
        palm_toward_point = self.get_palm_toward_point()

        palm_center_point = torch.mean(palm_surface_points, dim=1, keepdim=False)
        return palm_center_point, palm_toward_point

    def get_surface_points_and_normals(self, q=None):
        if q is not None:
            self.update_kinematics(q=q)
        surface_points = []
        surface_normals = []

        for link_name in self.surface_points:
            # for link_name in parts:
            # get transformation
            mapped_name = self._map_to_current_status_name(link_name)
            if mapped_name is None:
                continue
            trans_matrix = self.current_status[mapped_name].get_matrix()
            surface_points.append(
                torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
            surface_normals.append(
                torch.matmul(trans_matrix, self.surface_points_normal[link_name].transpose(1, 2)).transpose(1, 2)[...,
                :3])
        surface_points = torch.cat(surface_points, 1)
        surface_normals = torch.cat(surface_normals, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1,
                                                                                                      2) + self.global_translation.unsqueeze(
            1)
        surface_normals = torch.matmul(self.global_rotation, surface_normals.transpose(1, 2)).transpose(1, 2)

        return surface_points * self.scale, surface_normals

    def get_meshes_from_q(self, q=None, i=0):
        data = []
        if q is not None: self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            mapped_name = self._map_to_current_status_name(link_name)
            if mapped_name is None:
                print(f"[HandModel] 警告: 无法将链接名 '{link_name}' 映射到当前运动学链，已跳过。")
                continue
            if mapped_name in ['forearm']:
                continue
            trans_matrix = self.current_status[mapped_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(),
                                      transformed_v.T).T + np.expand_dims(
                self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(tm.Trimesh(vertices=transformed_v, faces=f))
        return data

    def get_plotly_data(self, q=None, i=0, color='lightblue', opacity=1.):
        data = []
        if q is not None: self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            mapped_name = self._map_to_current_status_name(link_name)
            if mapped_name is None:
                print(f"[HandModel] 警告: 无法将链接名 '{link_name}' 映射到当前运动学链，已跳过。")
                continue
            if mapped_name  in ['forearm']:
                continue
            trans_matrix = self.current_status[mapped_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(),
                                      transformed_v.T).T + np.expand_dims(
                self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(
                go.Mesh3d(x=transformed_v[:, 0], y=transformed_v[:, 1], z=transformed_v[:, 2], i=f[:, 0], j=f[:, 1],
                          k=f[:, 2], color=color, opacity=opacity))
        return data


def get_handmodel(batch_size, device, hand_scale=1., robot='shadowhand', rot_type='quat', anchor='base', mesh_source: str = 'urdf'):
    """
    创建 HandModel 实例
    
    Args:
        batch_size: 批次大小
        device: 设备 ('cpu' 或 'cuda')
        hand_scale: 手部缩放比例
        robot: 机器人类型（默认 'shadowhand'）
        rot_type: 旋转表示类型，'quat'/'r6d'/'euler'/'axis'（默认 'quat'）
        anchor: 平移锚点语义（默认 'base'）
            - 'base': 根坐标系即 URDF 基坐标系（t 输入即根平移）
            - 'palm_center': t 输入表示手掌中心在世界坐标的位置
            - 'mjcf'（原 'base_rz180_offset'）: 在 'base' 基础上进行 MJCF 定义的偏移（当前仅平移 [0, -0.01, 0.213]）
        mesh_source: mesh块来源，'urdf' 或 'mujoco'（默认 'urdf'）
    
    Returns:
        HandModel 实例
    """
    urdf_assets_meta = json.load(open("assets/urdf/urdf_assets_meta.json"))
    urdf_path = urdf_assets_meta['urdf_path'][robot]
    meshes_path = urdf_assets_meta['meshes_path'][robot]
    hand_model = HandModel(robot, urdf_path, meshes_path, batch_size=batch_size, device=device, hand_scale=hand_scale, rot_type=rot_type, anchor=anchor, mesh_source=mesh_source)
    return hand_model


def compute_collision(obj_pcd_nor: torch.Tensor, hand_pcd: torch.Tensor):
    """
    :param obj_pcd_nor: N_obj x 6
    :param hand_surface_points: B x N_hand x 3
    :return:
    """
    b = hand_pcd.shape[0]
    n_obj = obj_pcd_nor.shape[0]
    n_hand = hand_pcd.shape[1]

    obj_pcd = obj_pcd_nor[:, :3]
    obj_nor = obj_pcd_nor[:, 3:6]

    # batch the obj pcd
    batch_obj_pcd = obj_pcd.unsqueeze(0).repeat(b, 1, 1).view(b, 1, n_obj, 3)
    batch_obj_pcd = batch_obj_pcd.repeat(1, n_hand, 1, 1)
    # batch the hand pcd
    batch_hand_pcd = hand_pcd.view(b, n_hand, 1, 3).repeat(1, 1, n_obj, 1)
    # compute the pair wise dist
    hand_obj_dist = (batch_obj_pcd - batch_hand_pcd).norm(dim=3)
    hand_obj_dist, hand_obj_indices = hand_obj_dist.min(dim=2)
    # gather the obj points and normals w.r.t. hand points
    hand_obj_points = torch.stack([obj_pcd[x, :] for x in hand_obj_indices], dim=0)
    hand_obj_normals = torch.stack([obj_nor[x, :] for x in hand_obj_indices], dim=0)
    # compute the signs
    hand_obj_signs = ((hand_obj_points - hand_pcd) * hand_obj_normals).sum(dim=2)
    hand_obj_signs = (hand_obj_signs > 0.).float()
    # signs dot dist to compute collision value
    collision_value = (hand_obj_signs * hand_obj_dist).max(dim=1).values
    # collision_value = (hand_obj_signs * hand_obj_dist).mean(dim=1)
    return collision_value

def compute_collision_filter(obj_pcd_nor: torch.Tensor, hand_pcd: torch.Tensor):
    """
    :param obj_pcd_nor: N_obj x 6
    :param hand_surface_points: B x N_hand x 3
    :return:
    """
    b = hand_pcd.shape[0]
    n_obj = obj_pcd_nor.shape[0]
    n_hand = hand_pcd.shape[1]

    obj_pcd = obj_pcd_nor[:, :3]
    obj_nor = obj_pcd_nor[:, 3:6]

    # batch the obj pcd
    batch_obj_pcd = obj_pcd.unsqueeze(0).repeat(b, 1, 1).view(b, 1, n_obj, 3)
    batch_obj_pcd = batch_obj_pcd.repeat(1, n_hand, 1, 1)
    # batch the hand pcd
    batch_hand_pcd = hand_pcd.view(b, n_hand, 1, 3).repeat(1, 1, n_obj, 1)
    # compute the pair wise dist
    hand_obj_dist = (batch_obj_pcd - batch_hand_pcd).norm(dim=3)
    hand_obj_dist, hand_obj_indices = hand_obj_dist.min(dim=2)
    # gather the obj points and normals w.r.t. hand points
    hand_obj_points = torch.stack([obj_pcd[x, :] for x in hand_obj_indices], dim=0)
    hand_obj_normals = torch.stack([obj_nor[x, :] for x in hand_obj_indices], dim=0)
    # compute the signs
    hand_obj_signs = ((hand_obj_points - hand_pcd) * hand_obj_normals).sum(dim=2)
    hand_obj_signs = (hand_obj_signs > 0.).float()
    # signs dot dist to compute collision value
    collision_value = (hand_obj_signs * hand_obj_dist).max(dim=1).values
    # collision_value = (hand_obj_signs * hand_obj_dist).mean(dim=1)
    print(collision_value)
    return collision_value
def ERF_loss(obj_pcd_nor: torch.Tensor, hand_pcd: torch.Tensor):
    """
    Calculate the penalty loss based on point cloud and normal.

    :param obj_pcd_nor: B x N_obj x 6 (object point cloud with normals)
    :param hand_pcd: B x N_hand x 3 (hand point cloud)
    :return: ERF_loss (scalar)
    """
    b = hand_pcd.shape[0]
    n_obj = obj_pcd_nor.shape[1]
    n_hand = hand_pcd.shape[1]

    # Separate object point cloud and normals
    obj_pcd = obj_pcd_nor[:, :, :3]
    obj_nor = obj_pcd_nor[:, :, 3:6]

    # Compute K-nearest neighbors
    knn_result = knn_points(hand_pcd, obj_pcd, K=1, return_nn=True)
    distances = knn_result.dists
    indices = knn_result.idx
    knn = knn_result.knn
    distances = distances.sqrt()
    # Extract the closest object points and normals
    hand_obj_points = torch.gather(obj_pcd, 1, indices.expand(-1, -1, 3))
    hand_obj_normals = torch.gather(obj_nor, 1, indices.expand(-1, -1, 3))
    # Compute the signs
    hand_obj_signs = ((hand_obj_points - hand_pcd) * hand_obj_normals).sum(dim=2)
    hand_obj_signs = (hand_obj_signs > 0.).float()
    # Compute collision value
    # collision_value = (hand_obj_signs * hand_obj_dist).mean(dim=1)
    collision_value = (hand_obj_signs * distances.squeeze(2)).max(dim=1).values
    ERF_loss = collision_value.mean()
    return ERF_loss

def SPF_loss(dis_points, obj_pcd: torch.Tensor, thres_dis = 0.02 ):
    dis_points = dis_points.to(dtype=torch.float32)
    obj_pcd = obj_pcd.to(dtype=torch.float32)
    dis_pred = pytorch3d.ops.knn_points(dis_points, obj_pcd).dists[:, :, 0] # 64*140  # squared chamfer distance from object_pc to contact_candidates_pred
    small_dis_pred = dis_pred < thres_dis ** 2# 64*140
    SPF_loss = dis_pred[small_dis_pred].sqrt().sum() / (small_dis_pred.sum().item() + 1e-5)#1
    return SPF_loss

def SRF_loss(points):
    B, *points_shape = points.shape
    dis_spen = (points.unsqueeze(1) - points.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
    dis_spen = torch.where(dis_spen < 1e-6, 1e6 * torch.ones_like(dis_spen), dis_spen)
    dis_spen = 0.02 - dis_spen
    dis_spen[dis_spen < 0] = 0
    SPF_loss = dis_spen.sum() / B
    return SPF_loss

if __name__ == '__main__':
    from plotly_utils import plot_point_cloud
    seed = 0
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hand_model = get_handmodel(1, 'cuda')
    print(len(hand_model.robot.get_joint_parameter_names()))

    joint_lower = np.array(hand_model.revolute_joints_q_lower.cpu().reshape(-1))
    joint_upper = np.array(hand_model.revolute_joints_q_upper.cpu().reshape(-1))
    joint_mid = (joint_lower + joint_upper) / 2
    joints_q = (joint_mid + joint_lower) / 2
    q = torch.from_numpy(np.concatenate([np.array([0, 1, 0, 0, 1, 0, 1, 0, 0]), joint_lower])).unsqueeze(0).to(
        device).float()
    data = hand_model.get_plotly_data(q=q, opacity=0.5)
    palm_center_point, palm_toward_point = hand_model.get_palm_center_and_toward()
    data.append(plot_point_cloud(palm_toward_point.cpu() + palm_center_point.cpu(), color='black'))
    data.append(plot_point_cloud(palm_center_point.cpu(), color='red'))
    fig = go.Figure(data=data)
    fig.show()
