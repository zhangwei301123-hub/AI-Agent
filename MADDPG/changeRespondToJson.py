from google.protobuf.json_format import MessageToDict
import json

def get_area(response):
    dict_data = MessageToDict(response,
                              including_default_value_fields=True,
                              preserving_proto_field_name=True
                              )

    only_list = dict_data.get("areaPointList", [])
    area_list = [] #(经度， 纬度)
    for l in only_list:
        lon=l.get('lon',0)
        lat=l.get('lat',0)
        area_list.append((lon, lat))
    return area_list

def get_situaction(SituationDatas):
    dict_data = MessageToDict(SituationDatas,
                              including_default_value_fields=True, # 默认值字段保留  例如显示int里的0 
                              preserving_proto_field_name=True # 保留原始字段名
                              )
    only_list = dict_data.get("situaction", [])

    return only_list


def useUpReportsToJson(UseUpReports):
    dict_data = MessageToDict(UseUpReports,including_default_value_fields=False,preserving_proto_field_name=True)
    only_list = dict_data.get("useUpReport", [])
    return only_list

def attackReportsToJson(AttackReports):
    dict_data = MessageToDict(AttackReports,including_default_value_fields=False,preserving_proto_field_name=True)
    only_list = dict_data.get("attackReport", [])
    return only_list

def detectionReportsToJson(SituationDatas):
    dict_data = MessageToDict(SituationDatas,including_default_value_fields=False,preserving_proto_field_name=True)
    only_list = dict_data.get("detectionReport", [])
    return only_list
    
def test_get_situaction():
    s1 = SituationData(
        forceSide=1,
        activeLvl=100,
        attitude=SituationData.Attitude(pitch=2.0, roll=20.0, yaw=3.0),
        attrMap=SituationData.AttrMap(AirBase="base-uuid-1"),
        entitySpatialCoord=SituationData.EntitySpatialCoord(
            altitude=8.0,
            latitude=21.45,
            longitude=-157.76
        ),
        loadMap=SituationData.LoadMap(offensive=140, offenseless=3),
        logisticStates=SituationData.LogisticStates(oil=0.95),
        innerstates=SituationData.InnerStates(IsJamReaction=11),
        mdlID="uuid-entity-1",
        mdlType="AIRCRAFT",
        reportTime=909570365000,
        stateMap=SituationData.StateMap(
            FuelBurnRate="61",
            EcmStatus="0",
            RadarStatus="0",
            SonarStatus="0",
            AirStatus="Parked",
            MountId="13518",
            RemainDistance="2861",
            UnitStatus="UNASSIGNED",
            IdentifyStatus=1
        ),
        velocity=SituationData.Velocity(vx=120.0, vy=0.0, vz=5.0)
    )

    # 第二个实体
    s2 = SituationData(
        forceSide=2000000,
        activeLvl=80,
        attitude=SituationData.Attitude(pitch=5.0, roll=10.0, yaw=15.0),
        attrMap=SituationData.AttrMap(AirBase="base-uuid-2"),
        entitySpatialCoord=SituationData.EntitySpatialCoord(
            altitude=50.0,
            latitude=22.33,
            longitude=-158.11
        ),
        loadMap=SituationData.LoadMap(offensive=20, offenseless=1),
        logisticStates=SituationData.LogisticStates(oil=0.45),
        innerstates=SituationData.InnerStates(IsJamReaction=0),
        mdlID="uuid-entity-2",
        mdlType="DRONE",
        reportTime=909570400000,
        stateMap=SituationData.StateMap(
            FuelBurnRate="20",
            EcmStatus="1",
            RadarStatus="1",
            SonarStatus="0",
            AirStatus="Flying",
            MountId="22222",
            RemainDistance="1200",
            UnitStatus="ASSIGNED",
            IdentifyStatus=2
        ),
        velocity=SituationData.Velocity(vx=120.0, vy=0.0, vz=5.0)
    )
    # 加入 SituationDatas
    all_data = SituationDatas(situaction=[s1, s2])
    a=get_situaction(all_data)

    print(a)

def test_get_UseUpReports():
    report1 = UseUpReport(forceSide=1,id="unit-123", type="SHIP",time=3)
    report2 = UseUpReport(forceSide=1,id="unit-456", type="SHIP",time=3)
    report3 = UseUpReport(forceSide=1,id="unit-789", type="SHIP",time=3)
    all_reports = UseUpReports(useUpReport=[report1, report2, report3])
    a=get_UseUpReports(all_reports)
    print(a)

def test_get_AttackReports():
    report1 = AttackReport(
    launch_platform_name="ship-001",
    attack_target_type="ship",
    hit=True,
    fire_time=120
)

    report2 = AttackReport(
        launch_platform_name="air-007",
        attack_target_type="missille",
        hit=False,
        fire_time=125
    )

    report3 = AttackReport(
        launch_platform_name="ship-002",
        attack_target_type="aircraft",
        hit=True,
        fire_time=128
    )

    # 汇总到 AttackReports 容器中
    all_reports = AttackReports(attackReport=[report1, report2, report3])
    a=get_AttackReports(all_reports)
    print(a)

def test_get_DetectionReports():
    s1 = SituationData(
        forceSide=1,
        activeLvl=100,
        attitude=SituationData.Attitude(pitch=2.0, roll=20.0, yaw=3.0),
        attrMap=SituationData.AttrMap(AirBase="base-uuid-1"),
        entitySpatialCoord=SituationData.EntitySpatialCoord(
            altitude=8.0,
            latitude=21.45,
            longitude=-157.76
        ),
        loadMap=SituationData.LoadMap(offensive=140, offenseless=3),
        logisticStates=SituationData.LogisticStates(oil=0.95),
        innerstates=SituationData.InnerStates(IsJamReaction=11),
        mdlID="uuid-entity-1",
        mdlType="AIRCRAFT",
        reportTime=909570365000,
        stateMap=SituationData.StateMap(
            FuelBurnRate="61",
            EcmStatus="0",
            RadarStatus="0",
            SonarStatus="0",
            AirStatus="Parked",
            MountId="13518",
            RemainDistance="2861",
            UnitStatus="UNASSIGNED",
            IdentifyStatus=1
        ),
        velocity=SituationData.Velocity(vx=120.0, vy=0.0, vz=5.0)
    )

    # 第二个实体
    s2 = SituationData(
        forceSide=2000000,
        activeLvl=80,
        attitude=SituationData.Attitude(pitch=5.0, roll=10.0, yaw=15.0),
        attrMap=SituationData.AttrMap(AirBase="base-uuid-2"),
        entitySpatialCoord=SituationData.EntitySpatialCoord(
            altitude=50.0,
            latitude=22.33,
            longitude=-158.11
        ),
        loadMap=SituationData.LoadMap(offensive=20, offenseless=1),
        logisticStates=SituationData.LogisticStates(oil=0.45),
        innerstates=SituationData.InnerStates(IsJamReaction=0),
        mdlID="uuid-entity-2",
        mdlType="DRONE",
        reportTime=909570400000,
        stateMap=SituationData.StateMap(
            FuelBurnRate="20",
            EcmStatus="1",
            RadarStatus="1",
            SonarStatus="0",
            AirStatus="Flying",
            MountId="22222",
            RemainDistance="1200",
            UnitStatus="ASSIGNED",
            IdentifyStatus=2
        ),
        velocity=SituationData.Velocity(vx=120.0, vy=0.0, vz=5.0)
    )
    all_reports = SituationDatas(situaction=[s1])
    a=get_DetectionReports(all_reports)
    print(a)
if __name__ == '__main__':
    test_get_DetectionReports()
