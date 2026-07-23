#fake_server.py
import grpc
from concurrent import futures
import time
import datetime

# 导入根据 proto 编译出来的模块
from . import SimulationControlInstruction_pb2 as pb2
from . import SimulationControlInstruction_pb2_grpc as pb2_grpc
from .SimulationControlInstruction_pb2 import AreaPointList, AreaPoint

# 模拟服务实现类
class SimulationServiceServicer(pb2_grpc.SimulationServiceServicer):
    def __init__(self):
        super().__init__()
        self.tick = 0
    def setTimeCompression(self, request, context):
        print(f"⏱️ 设置时间压缩倍率为：{request.timeCompression}")
        return pb2.ResponseDataw(code=0, error_message="Time compression set")

    def loadScenario(self, request, context):
        print(f"📂 加载场景文件: {request.fileName}")
        return pb2.ResponseDataw(code=0, error_message="Scenario loaded")

    def startDedicew(self, request, context):
        print("🚀 启动模拟")
        return pb2.ResponseDataw(code=0, error_message="OK")

    def pauseDedicew(self, request, context):
        print("⏸️ 暂停模拟")
        return pb2.ResponseDataw(code=0, error_message="Paused")

    def getEndSignal(self, request, context):
        print("⏱️ 获取模拟结束信号")
        return pb2.EndSignalw(code=0)

    def getSituation(self, request, context):
        self.tick += 1
        print("⏱️ 获取模拟场景")
        entities = []
        # for i in range(13):
        #     entities.append(pb2.SituationDataw(
        #         forceSide="蓝方",
        #         activeLvl=100,
        #         attitude=pb2.SituationDataw.Attitudew(pitch=1.0, roll=2.0, yaw=0),
        #         attrMapw=pb2.SituationDataw.AttrMapw(AirBase="true"),
        #         entitySpatialCoord=pb2.SituationDataw.EntitySpatialCoordw(altitude=1000, latitude=0+0.0001*i, longitude=110),
        #         loadMap=pb2.SituationDataw.LoadMapw(offensive=3, offenseless=2),
        #         logisticStates=pb2.SituationDataw.LogisticStatesw(oil=0.999),
        #         innerstates=pb2.SituationDataw.InnerStatesw(IsJamReaction=0),
        #         mdlID=f"ship_{i}",
        #         mdlType="SHIP",
        #         maxRange=pb2.SituationDataw.MaxRange(
        #             maxAir=180.0,
        #             maxSubsurface=150.0,
        #             maxSurface=205.0,
        #             maxLand=0.0
        #         ),
        #         reportTime=self.tick,
        #         stateMap=pb2.SituationDataw.StateMapw(
        #             FuelBurnRate=5.0, EcmStatus=1, RadarStatus=1, SonarStatus=0,
        #             AirStatus="Flying", RemainDistance=1000, UnitStatus="Normal", IdentifyStatus=1
        #         ),
        #         velocity=pb2.SituationDataw.Velocityw(vx=10.0, vy=1.0, vz=0.0),
        #         IsUnderAttack=0,

        #     ))

        # 我方实体
        for i in range(120):
            entities.append(pb2.SituationDataw(
                forceSide="蓝方",
                activeLvl=100,
                attitude=pb2.SituationDataw.Attitudew(pitch=1.0, roll=2.0, yaw=0),
                attrMapw=pb2.SituationDataw.AttrMapw(AirBase="true"),
                entitySpatialCoord=pb2.SituationDataw.EntitySpatialCoordw(altitude=1000, latitude=0+0.0001*i, longitude=110),
                loadMap=pb2.SituationDataw.LoadMapw(offensive=3, offenseless=2),
                logisticStates=pb2.SituationDataw.LogisticStatesw(oil=0.999),
                innerstates=pb2.SituationDataw.InnerStatesw(IsJamReaction=0),
                mdlID=f"Aircraft_ASW",
                mdlType="AIRCRAFT",
                maxRange=pb2.SituationDataw.MaxRange(
                    maxAir=180.0,
                    maxSubsurface=150.0,
                    maxSurface=205.0,
                    maxLand=0.0
                ),
                reportTime=self.tick,
                stateMap=pb2.SituationDataw.StateMapw(
                    FuelBurnRate=5.0, EcmStatus=1, RadarStatus=1, SonarStatus=0,
                    AirStatus="Flying", RemainDistance=1000, UnitStatus="Normal", IdentifyStatus=1
                ),
                velocity=pb2.SituationDataw.Velocityw(vx=10.0, vy=1.0, vz=0.0),
                IsUnderAttack=0,
                unitCategory = "Aircraft_ASW",
                isCanManaged = True
            ))

        # 敌方实体
        for i in range(100):
            entities.append(pb2.SituationDataw(
                forceSide="红方",
                activeLvl=80,
                attitude=pb2.SituationDataw.Attitudew(pitch=0.0, roll=0.0, yaw=180.0),
                attrMapw=pb2.SituationDataw.AttrMapw(),
                entitySpatialCoord=pb2.SituationDataw.EntitySpatialCoordw(altitude=1000, latitude=30-i, longitude=110),
                loadMap=pb2.SituationDataw.LoadMapw(offensive=1, offenseless=0),
                logisticStates=pb2.SituationDataw.LogisticStatesw(oil=500.0),
                innerstates=pb2.SituationDataw.InnerStatesw(IsJamReaction=0),
                mdlID=f"enemy_{i}",
                mdlType="SHIP",
                reportTime=self.tick,
                stateMap=pb2.SituationDataw.StateMapw(
                    FuelBurnRate=3.0, EcmStatus=1, RadarStatus=0, SonarStatus=1,
                    AirStatus="Sailing", RemainDistance=500, UnitStatus="Normal", IdentifyStatus=2
                ),
                velocity=pb2.SituationDataw.Velocityw(vx=5.0, vy=0.0, vz=0.0),
                IsUnderAttack=0
            ))

        for i in range(20):
            entities.append(pb2.SituationDataw(
                forceSide="红方",
                activeLvl=100,
                attitude=pb2.SituationDataw.Attitudew(pitch=0.0, roll=0.0, yaw=0.0),
                attrMapw=pb2.SituationDataw.AttrMapw(),
                entitySpatialCoord=pb2.SituationDataw.EntitySpatialCoordw(altitude=1000, latitude=1+i- 0.10 * self.tick, longitude=110),# 每帧靠近 0.05°


                loadMap=pb2.SituationDataw.LoadMapw(offensive=0, offenseless=0),
                logisticStates=pb2.SituationDataw.LogisticStatesw(oil=0.0),  # 导弹通常不需要油料
                innerstates=pb2.SituationDataw.InnerStatesw(IsJamReaction=0),
                mdlID=f"WEAPON_{i}",
                mdlType="WEAPON",  # 这里是导弹
                reportTime=self.tick,
                stateMap=pb2.SituationDataw.StateMapw(
                    FuelBurnRate=0.0, EcmStatus=0, RadarStatus=0, SonarStatus=0,
                    AirStatus="Flying", RemainDistance=0, UnitStatus="Normal", IdentifyStatus=1
                ),
                velocity=pb2.SituationDataw.Velocityw(vx=300.0, vy=0.0, vz=0.0),  # 高速飞行
                IsUnderAttack=0,
            ))

        return pb2.SituationDatas(situaction=entities)

    def getUseUpReportw(self, request, context):
        report = pb2.UseUpReportw(forceSide="蓝方", id="Aircraft_ASW", type="AIRCRAFT", time=1, damage_degree=0.3)
        print("获取战损报告")
        return pb2.UseUpReports(useUpReport=[report])

    def getAttackReport(self, request, context):
        report = pb2.AttackReportw(
            launch_platform_id="Aircraft_ASW",
            attack_target_type="SHIP",
            hit=True,
            fire_time=0,
            target_damage_degree=0.8
        )
        report1 = pb2.AttackReportw(
            launch_platform_id="ally_1",
            attack_target_type="SHIP",
            hit=True,
            fire_time=0,
            target_damage_degree=0.8
        )
        print("获取攻击报告")
        return pb2.AttackReports(attackReport=[report,report1])

    def getDetectionReport(self, request, context):
        report = pb2.DetectionReportw(
            targetId="enemy_1",
            targetType="SHIP",
            detectorId="ally_1",
            detectorType="AIRCRAFT",
            detectStep=1,
            isSonobuoy=False
        )
        report2 = pb2.DetectionReportw(
            targetId="enemy_2",
            targetType="SHIP",
            detectorId="ally_2",
            detectorType="AIRCRAFT",
            detectStep=2,
            isSonobuoy=False
        )
        print("获取探测报告")
        return pb2.DetectionReports(detectionReport=[report,report2])

    def aircraftTakeOffSinglew(self, request, context):
        print("执行起飞")
        return pb2.ResponseDataw(code=0, error_message="Takeoff OK")

    def aircraftReturnToBasew(self, request, context):
        print("执行返航")
        return pb2.ResponseDataw(code=0, error_message="Return OK")

    def setUnitRoutew(self, request, context):
        # print("设置航线")
        return pb2.ResponseDataw(code=0, error_message="Route OK")

    def adjustUnitAltitudeAndSpeed(self, request, context):
        print("调整高度和速度")
        return pb2.ResponseDataw(code=0, error_message="Adjusted OK")

    def attackContactw(self, request, context):
        print("执行攻击")
        return pb2.ResponseDataw(code=0, error_message="Attack OK")

    def controlUnitSensorw(self, request, context):
        print("控制传感器")
        return pb2.ResponseDataw(code=0, error_message="Sensor OK")

    def delpoySonobuoyw(self, request, context):
        print("部署声呐")
        return pb2.ResponseDataw(code=0, error_message="Sonobuoy OK")

    def cancelAttackw(self, request, context):
        print("取消攻击")
        return pb2.ResponseDataw(code=0, error_message="Cancel OK")

    def getCombatArea(self, request, context):
        # 一定要有这一段实现
        pts = [
            AreaPoint(
                id="pt1",
                lon=100.0,
                lat=1.0,
            ),
            AreaPoint(
                id="pt2",
                lon=100.0,
                lat=-1.0,
            ),
            AreaPoint(
                id="pt3",

                lon=120.0,
                lat=1.0,

            ),
            AreaPoint(
                id="pt4",

                lon=120.0,
                lat=-1.0,

            ),
        ]
        return AreaPointList(areaPointList=pts)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_SimulationServiceServicer_to_server(SimulationServiceServicer(), server)
    server.add_insecure_port('[::]:50051')  # 默认端口
    server.start()
    print("✅ 模拟 gRPC 服务已启动: localhost:50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
