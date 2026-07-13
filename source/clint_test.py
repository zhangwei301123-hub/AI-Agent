import grpc
import time
from google.protobuf import empty_pb2

import SimulationControlInstruction_pb2 as pb2
import SimulationControlInstruction_pb2_grpc as pb2_grpc


def main():
    # channel = grpc.insecure_channel('localhost:50051')
    channel = grpc.insecure_channel('192.168.1.233:9901')
    stub = pb2_grpc.SimulationServiceStub(channel)

    print("== restoreScenario ==")
    situation = stub.restoreScenario(empty_pb2.Empty())
    print("Units:", [s.mdlID for s in situation.situaction])

    print("== startDedice ==")
    resp = stub.startDedicew(pb2.IdRequestw(mdlID=""))
    print(resp.message)

    print("== pauseDedice ==")
    resp = stub.pauseDedicew(empty_pb2.Empty())
    print(resp.message)

    print("== getEndSignal ==")
    end = stub.getEndSignal(empty_pb2.Empty())
    print("Simulation Ended?" , end.code == 1)

    print("== getSituation ==")
    situation = stub.getSituation(empty_pb2.Empty())
    for s in situation.situaction:
        print(f"{s.mdlID}: {s.mdlType}, Pos=({s.entitySpatialCoord.latitude},{s.entitySpatialCoord.longitude})")

    print("== getUseUpReport ==")
    loss_report = stub.getUseUpReportw(empty_pb2.Empty())
    for r in loss_report.useUpReport:
        print(f"Lost unit: {r.id}, type: {r.type}")

    print("== getAttackReport ==")
    attack_report = stub.getAttackReport(empty_pb2.Empty())
    for a in attack_report.attackReport:
        print(f"Attack from {a.launch_platform_id} to {a.attack_target_type}, hit={a.hit}, damage={a.target_damage_degree}")

    print("== getDetectionReport ==")
    detect_report = stub.getDetectionReport(empty_pb2.Empty())
    for d in detect_report.detectionReport:
        print(f"Detected: {d.targetId} by {d.detectorId} (Sonobuoy? {d.isSonobuoy})")

    print("== aircraftTakeOffSingle ==")
    resp = stub.aircraftTakeOffSinglew(pb2.IdRequestw(mdlID="unit_001"))
    print(resp.message)

    print("== aircraftReturnToBase ==")
    resp = stub.aircraftReturnToBasew(pb2.IdRequestw(mdlID="unit_001"))
    print(resp.message)

    print("== setUnitRoute ==")
    wp1 = pb2.WayPointw(latitude=35.0, longitude=120.0, altitude=1000, velocity=250)
    wp2 = pb2.WayPointw(latitude=36.0, longitude=121.0, altitude=1200, velocity=300)
    route = pb2.UnitRoutew(mdlID="unit_001", Route=[wp1, wp2])
    resp = stub.setUnitRoutew(route)
    print(resp.message)

    print("== adjustUnitAltitudeAndSpeed ==")
    dummy_request = pb2.UnitAltitudeAndSpeedw()  # 根据你真实的字段补充
    resp = stub.adjustUnitAltitudeAndSpeed(dummy_request)
    print(resp.message)

    print("== attackContact ==")
    dummy_attack = pb2.AttackRequestw()  # 根据你真实的字段补充
    resp = stub.attackContactw(dummy_attack)
    print(resp.message)

    print("== controlUnitSensor ==")
    dummy_sensor = pb2.SensorControlRequestw()  # 根据你真实的字段补充
    resp = stub.controlUnitSensorw(dummy_sensor)
    print(resp.message)

    print("== delpoySonobuoy ==")
    dummy_buoy = pb2.SonobuoyDelpoyRequestw()  # 根据你真实的字段补充
    resp = stub.delpoySonobuoyw(dummy_buoy)
    print(resp.message)

    print("== cancelAttack ==")
    resp = stub.cancelAttackw(pb2.IdRequestw(mdlID="unit_001"))
    print(resp.message)


if __name__ == "__main__":
    main()
