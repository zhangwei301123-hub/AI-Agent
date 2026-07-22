import copy
import time

from google.protobuf.empty_pb2 import Empty
import grpc
import numpy as np
from symbolic_reasoning.engine_pb2 import *
from symbolic_reasoning.engine_pb2_grpc import SimulationServiceStub
from symbolic_reasoning.execute_actions import execute_actions as _execute_symbolic_actions
from symbolic_reasoning.entity import EntityEncoder as SymbolicEntityEncoder
from symbolic_reasoning.live import RpcSituationSource

from changeRespondToJson import *
from maddpg_live_adapter import legacy_entities_from_symbolic_payload
import random
from threading import Lock

IS_TRAIN = True
def set_is_train(new_flag):
    global IS_TRAIN
    IS_TRAIN = new_flag

def get_is_train():
    return IS_TRAIN

# SCENARIO = "红蓝对战3.7"
SCENARIO = "新红蓝对战3.4(增加四种专业)"
# SCENARIO = "智能体测试4"

# 创建 gRPC 通道
RPC_TARGET = '10.2.0.106:50051'
channel = grpc.insecure_channel(RPC_TARGET)

# 创建客户端
stub = SimulationServiceStub(channel)

SPEED_RATE = 2

TIME_SPEED_MAP = {
    0:1,
    1:2,
    2:5,
    3:15,
    4:30,
    5:150
}

SPEED_RATE2 = 0
_SPEED_RATE_LOCK = Lock()
_LIVE_SOURCE = None
_LIVE_SOURCE_LOCK = Lock()
_LIVE_STATE_LOCK = Lock()
_LAST_LIVE_ENTITIES = None
_LAST_SYMBOLIC_SITUATION = None
_LIVE_SITUATION_STALE = False

LIVE_SITUATION_RETRY_ATTEMPTS = 2
LIVE_SITUATION_RETRY_DELAY_SECONDS = 0.5
_TRANSIENT_SITUATION_CODES = {
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
}


def _get_live_source(logger=None):
    global _LIVE_SOURCE
    with _LIVE_SOURCE_LOCK:
        if _LIVE_SOURCE is None:
            _LIVE_SOURCE = RpcSituationSource(
                rpc_target=RPC_TARGET,
                timeout=10.0,
                logger=logger,
            )
        return _LIVE_SOURCE


def is_live_situation_stale():
    """最近一次态势读取是否因瞬时 RPC 故障而使用了缓存。"""
    with _LIVE_STATE_LOCK:
        return _LIVE_SITUATION_STALE


def get_last_symbolic_situation():
    """返回最近一次成功实时读取所生成的符号态势（只读对象）。"""
    with _LIVE_STATE_LOCK:
        return _LAST_SYMBOLIC_SITUATION

def set_speed_rate(new_rate):
    global SPEED_RATE2
    with _SPEED_RATE_LOCK:
        SPEED_RATE2 = new_rate

def get_speed_rate():
    with _SPEED_RATE_LOCK:
        return SPEED_RATE2

from shapely.geometry import MultiPoint
def build_convex_hull(points):
    """
    points: List[ (lon, lat), ... ]
    返回 Shapely Polygon 对象，是这些点的凸包。
    """
    mpts = MultiPoint(points)
    hull = mpts.convex_hull
    return hull 

def get_attack_area():
    # response = stub.get_Attack_Area(Empty())
    response = stub.getCombatArea(Empty())
    area_list = get_area(response)
    area_list = build_convex_hull(area_list)
    return area_list

def get_mission_dicts():
    """   
    # 查找匹配的任务
    mission_data = missions_dict.get(entity_mission_id)
    mission_data["area_points"]
    """

#     mission_dicts = {
#     "M001": {               # 例：巡逻任务
#         "area_points": [
#             (100.0,  1.0),
#             (100.0, -1.0),
#             (102.0, -1.0),
#             (102.0,  1.0),
#         ]
#     },
#     "M002": {               # 例：打击任务
#         "area_points": [
#             (110.0, 30.0),
#             (112.0, 28.0),
#             (114.0, 30.0),
#             (112.0, 32.0),
#         ]
#     },
#     "M003": {               # 例：侦察任务
#         "area_points": [
#             (118.0,  5.0),
#             (120.0,  3.0),
#             (122.0,  5.0),
#             (120.0,  7.0),
#         ]
#     },
# }
#     return mission_dicts
   
    mission_list = stub.getMissionList(Empty())
    mission_dict = {}
    
    dict_data = MessageToDict(
        mission_list,
        including_default_value_fields=True,
        preserving_proto_field_name=True
    )
    
    for mission in dict_data.get("mission", []):
        points = []
        for point in mission.get("areaPoints", []):
            points.append((float(point["lon"]), float(point["lat"])))
            
        mission_dict[mission["missionId"]] = {
            "area_points": points,
        }
    
    return mission_dict

def _execute_actions_legacy(actions_dict, enemy_ids, probablity=0.7, logger=None):
    rewards = np.zeros(8)
    '''
        判断指令是否执行成功的规则：
        - 如果概率 < 阈值 → 直接判定为 False（不执行）
        - 如果概率 >= 阈值：
            - 执行后 response.code == 0 → True（成功）
            - 执行后 response.code != 0 → False（失败）
        - 如果第0项（起飞）为True，则第1项（降落）强制为False
        - 如果第2项（航路机动）为True，则第3项（速度高度调整）强制为False
    '''
    execute_results = {}


    for key, value in actions_dict.items():
        id = key
        result = []

        for i in range(len(value)):
            try:
                if value[i][0] >= probablity or (i == 4 and value[i][0] >= 0.4):
                    # === 执行各类动作 ===
                    if i == 0 :
                        if random.random() >0.7:
                        	response = stub.aircraftTakeOffSinglew(IdRequestw(mdlID=id))
                    elif i == 1 and value[i][0] > 0.8:
                        response = stub.aircraftReturnToBasew(IdRequestw(mdlID=id))
                    elif i == 2 :
                        longitudes = value[i][1]
                        latitudes  = value[i][2]
                        altitudes  = value[i][3]
                        velocities = value[i][4]
                        if not isinstance(value[i][3], list):
                            longitudes = [longitudes]
                            latitudes =  [latitudes]
                            altitudes =  [altitudes]
                            velocities = [velocities]
                        route_points = []
                        for lon, lat, alt, vel in zip(longitudes, latitudes, altitudes, velocities):
                            wp = WayPointw(
                                longitude=float(lon),
                                latitude=float(lat),
                                altitude=4,
                                velocity=4,
                            )
                            route_points.append(wp)

                            response = stub.setUnitRoutew(UnitRoutew(
                                mdlID=id,
                                Route=route_points
                            ))
                    elif i == 3:
                        response = stub.adjustUnitAltitudeAndSpeed(UnitAltitudeAndSpeedw(
                            mdlID=id,
                            velocity=4,
                            altitude=4
                        ))
                    elif i == 4:
                        if value[i][2] != 0:
                            response = stub.attackOrientationw(AttackOrientationw(
                                attackerId = id,
                                lon = float(value[i][2]),
                                lat = float(value[i][3])
                            ))
                        if value[i][1] is not None :
                            if len(str(value[i][1])) < 10:
                                value[i][1] = enemy_ids[0]
                            # response1 = stub.attackContact(AttackRequest(
                            #         attackerId=id,
                            #         contactId=str(value[i][1],
                            #         AttackOptions(
                            #     AttackMode(AutoTargeted=0)
                            #         ))))
                            # print("response===========111")
                            # print(response1)

                            response = stub.attackContactw(AttackRequestw(
                                attackerId=id,
                                contactId=str(value[i][1])
                        ))
                        if len(str(value[i][4])) >= 10:
                            response =stub.attackContactw(AttackRequestw(
                                attackerId=id,
                                contactId=str(value[i][4])
                            ))
                        if response.code == 1:
                            print(f'!!!!!{key}执行打击{str(value[i][1])}任务失败!!!!{response.error_message}')
                        else:
                            print(f'{key}武器发射打击{str(value[i][1])}成功~~~~~~~~~~~~~~~~~~~~~~··')
                    elif i == 5:
                        response = stub.controlUnitSensorw(SensorControlRequestw(
                            id=id,
                            radar=(value[i][1] > 0.5),
                            sonar=(value[i][2] > 0.5),
                            ecm=(value[i][3] > 0.5)
                        ))
                    elif i == 6:
                        response = stub.delpoySonobuoyw(SonobuoyDelpoyRequestw(
                            id=id,
                            passiveOrActive=(value[i][1] > 0.5),
                            shallowOrDeep=(value[i][2] > 0.5)
                        ))
                    elif i == 7 and value[i][0] > 0.8:
                        response = stub.cancelAttackw(IdRequestw(mdlID=id))
                    else:
                        response = getEndSignal()
                        response.code = 1

                    success = (response.code == 0) # 0代表执行成功。1 代表失败
                    # if not success:
                    #     print(f'{key}执行{i}失败')
                    result.append(success)

                    # 奖励惩罚规则 0是成功 1是失败
                    # if success and i in [4, 6]:
                    if success and i in [4,6]:
                        rewards[i] -= 2 
                    if response.code == 0: #执行成功就奖励
                        rewards[i] += 1 

                    # 互斥规则：航路成功后禁止速度高度调整
                    if i == 2 and success and value[3][0] >= probablity:
                        value[3][0] = 0.0
                        rewards[3] = -1

                    # 互斥规则：起飞成功后禁止返航
                    if i == 0 and success and value[1][0] >= probablity:
                        value[1][0] = 0.0
                        rewards[1] = -1

                else:
                    result.append(False)

            except Exception as e:
                result.append(False)
                print("出错了！！！！！！！！！！")

        execute_results[key] = result
        # for i, action_is_performs in enumerate(result):
        #     if action_is_performs:
        #         rewards[i] += 0.5


    return execute_results, rewards


# MADDPG 保持原 8×5 动作格式，但实际下发统一走符号推理已经验收的
# protobuf 和 AttackTarget 流水线，避免旧、新 proto 在同一进程冲突。
def execute_actions(actions_dict, enemy_ids, probablity=0.7, logger=None):
    normalized_actions = {
        str(entity_id): [
            [value.item() if isinstance(value, np.generic) else value for value in action]
            for action in actions
        ]
        for entity_id, actions in actions_dict.items()
    }
    execute_results, rewards = _execute_symbolic_actions(
        normalized_actions,
        list(map(str, enemy_ids)),
        probablity=probablity,
        logger=logger,
        rpc_target=RPC_TARGET,
    )
    return execute_results, np.asarray(rewards, dtype=float)



def reset(logger):
    # pdb.set_trace()
    # stub.endDedice(Empty())
    # stub.restoreScenario(Empty())
    stub.loadScenario(ScenarioFileRequest(fileName=SCENARIO, scenarioXml=""))
    stub.setTimeCompression(TimeCompressionRequest(timeCompression=SPEED_RATE))
    stub.startDedicew(IdRequestw(mdlID=""))
    time.sleep(1)
    response = stub.getSituation(Empty())
    # 返回的是所有的单装信息
    response = get_situaction(response) #返回场景的 json化的 所有单装信息
    # print("================RESET================")
    logger.info("================RESET================")
    return response

def reset4test(logger):
    response = get_Situaction4test(logger)
    logger.info("================RESET================")
    return response

def get_Situaction(logger):
    response=stub.getSituation(Empty())
    response=get_situaction(response) #返回场景的 json化的 所有单装信息
    if getEndSignal().code == 0:
        pause(logger)
    return response

def get_Situaction4test(logger):
    global _LAST_LIVE_ENTITIES, _LAST_SYMBOLIC_SITUATION, _LIVE_SITUATION_STALE

    with _LIVE_STATE_LOCK:
        cached_response = copy.deepcopy(_LAST_LIVE_ENTITIES)

    # 运行过程中已有缓存时，一次超时后直接进入安全等待；启动阶段没有缓存，
    # 则允许再试一次，避免服务端偶发抖动导致程序无法启动。
    max_attempts = 1 if cached_response is not None else LIVE_SITUATION_RETRY_ATTEMPTS
    last_error = None
    response = None
    for attempt in range(1, max_attempts + 1):
        try:
            payload = _get_live_source(logger).fetch_payload()
            symbolic_situation = SymbolicEntityEncoder().encode(payload)
            response = legacy_entities_from_symbolic_payload(
                payload,
                situation=symbolic_situation,
            )
            break
        except grpc.RpcError as exc:
            if exc.code() not in _TRANSIENT_SITUATION_CODES:
                raise
            last_error = exc
            if attempt < max_attempts:
                if logger is not None:
                    logger.warning(
                        '[MADDPG实时态势] RPC暂时失败 code=%s，%.1f秒后重试(%s/%s)',
                        exc.code().name,
                        LIVE_SITUATION_RETRY_DELAY_SECONDS,
                        attempt,
                        max_attempts,
                    )
                time.sleep(LIVE_SITUATION_RETRY_DELAY_SECONDS)

    if response is None:
        if cached_response is None:
            # 首帧没有可安全使用的历史态势，保留原始 gRPC 状态码给调用方。
            raise last_error
        with _LIVE_STATE_LOCK:
            _LIVE_SITUATION_STALE = True
        if logger is not None:
            logger.warning(
                '[MADDPG实时态势] RPC暂时失败 code=%s，使用最近成功态势；恢复前暂停下发动作',
                last_error.code().name,
            )
        return cached_response

    with _LIVE_STATE_LOCK:
        _LAST_LIVE_ENTITIES = copy.deepcopy(response)
        _LAST_SYMBOLIC_SITUATION = symbolic_situation
        _LIVE_SITUATION_STALE = False
    if logger is not None:
        own_count = sum(entity.get('isCanManaged', False) for entity in response)
        target_count = sum(not entity.get('isCanManaged', False) for entity in response)
        logger.debug(
            '[MADDPG实时态势] source=GetThreeSituation(red-view) entities=%s own=%s targets=%s',
            len(response), own_count, target_count,
        )
    return response

def start(logger):
    stub.setTimeCompression(TimeCompressionRequest(timeCompression=SPEED_RATE))
    response = stub.startDedicew(IdRequestw(mdlID=""))
    # print("================START===============")
    logger.info("================START===============")
    return response

def pause(logger):
    response = stub.pauseDedicew(Empty())
    # print("================PAUSE===============")
    logger.info("================PAUSE===============")
    return response

def getEndSignal():  
    response = stub.getEndSignal(Empty())
    return response


def get_UseUpReports():
    response = stub.getUseUpReportw(Empty())
    response = useUpReportsToJson(response)
    return response
def get_AttackReports():
    response = stub.getAttackReport(Empty())
    response = attackReportsToJson(response) #返回场景的 json化的 所有单装信息
    return response

def get_DetectionReports():
    response = stub.getDetectionReport(Empty())
    response = detectionReportsToJson(response) #返回场景的 json化的 所有单装信息
    return response

def get_control_signal():
    code = getEndSignal().code
    if code == 0:
        return "start"
    elif code == 1:
        return "stop"
    elif code == 2:
        return "pause"
    elif code == 3:
        return "running"


def get_speed():
    time_compression = stub.getTimeCompression(Empty()).timeCompression
    return time_compression if time_compression <= 5 else 5



if __name__ == "__main__":
    import pdb
    # a = reset()
    # d=get_Situaction()
    # a=start()
    # a=pause()
    # a=getEndSignal()
    # b=get_AttackReports()
    # a=get_UseUpReports()
    # c=get_DetectionReports()
    stub.loadScenario(ScenarioFileRequest(fileName=SCENARIO, scenarioXml=""))
    a = get_attack_area()

    response = stub.setUnitRoutew(UnitRoutew(
        mdlID='7e661e664db14ef59669f7b2fdff826b',
        Route=[WayPointw(
            longitude=120,
            latitude=0,
            altitude=4,
            velocity=4
        )]
    ))
    print(11111)






# response = stub.aircraftTakeOffSingle(request, timeout=10)



# situationTime= SimulationControlInstructionZnt_pb2.SituationTime(="10101",currentTime="sad")

# situationTime.beginTime="asdasewq"
# print(situationTime)
# SimulationControlInstructionZnt_pb2.pauseDedice()
