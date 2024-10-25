<?xml version="1.0" encoding="UTF-8"?>
<!-- Scenario created in date 2024-10-25 10:22:11 by AirMosaic 8.24.2-89 (Build 202403040849)-->
<TestScenario Version="7" Repetitions="1" RepetitionsDelay="5000" Looped="false" DurationEnable="true" DurationKind="Explicit" Duration="$(SCENARIO_DURATION)" SuddenShutdown="true" Annotation="" GlobalVariables="" LayoutLinkEnabled="false" CountersLayoutPath="" ChartsLayoutPath="" PesqLayoutPath="" ScenarioType="" AppVersion="AirMosaic 8.24.2-89 (Build 202403040849)" TstmName="LTE_Tm">
  <Bearer>
    <db name="Bearer" formatname="Bearer" version="1.0 1.0" />
  </Bearer>
  <Flow>
    <db name="Flow" formatname="Flow" version="1.0 1.0" />
  </Flow>
  <BearerModificationProfileList>
    <db name="QCI3" formatname="BearerModDb" version="1.0" />
    <looped name="QCI3" value="false" />
    <db name="Mack6" formatname="BearerModDb" version="1.0" />
    <looped name="Mack6" value="false" />
    <db name="QCI7" formatname="BearerModDb" version="1.0" />
    <looped name="QCI7" value="false" />
    <db name="Serg6" formatname="BearerModDb" version="1.0" />
    <looped name="Serg6" value="false" />
    <db name="QCI6-MR-KRABS" formatname="BearerModDb" version="1.0" />
    <looped name="QCI6-MR-KRABS" value="false" />
  </BearerModificationProfileList>
  <Groups>
    <Group name="08-IP_PT" extensible="true" enableGroup="true" loopenGroupRepetitions="false" groupRepetitionsNumber="1">
      "UE"
      <TemplateProfileMobility property="true" mobilityName="Advanced Mobility" radiocondition="Excellent_RadioConditions_SA" spatialDisplacement="Index" indexdisplacement="0" temporalDisplacement="Scenario Driven" Distribution="No Delay " upondetach="false" name="com.prisma.mobility.mobilityplugin.MapDrivenMobility" cellbinding="&lt;?xml version=&quot;1.0&quot; encoding=&quot;UTF-8&quot; standalone=&quot;yes&quot;?&gt;&#xA;&lt;RadioConditionCellBinding&gt;&#xA;    &lt;cellBindingMap/&gt;&#xA;    &lt;autoCellBinding&gt;false&lt;/autoCellBinding&gt;&#xA;&lt;/RadioConditionCellBinding&gt;&#xA;" nrcellbinding="&lt;?xml version=&quot;1.0&quot; encoding=&quot;UTF-8&quot; standalone=&quot;yes&quot;?&gt;&#xA;&lt;RadioConditionNrCellBinding&gt;&#xA;    &lt;nrCellBindingMap&gt;&#xA;        &lt;entry&gt;&#xA;            &lt;key&gt;New Custom 5GNr Cell 0&lt;/key&gt;&#xA;            &lt;value&gt;1&lt;/value&gt;&#xA;        &lt;/entry&gt;&#xA;    &lt;/nrCellBindingMap&gt;&#xA;&lt;/RadioConditionNrCellBinding&gt;&#xA;" overrideoptions="&lt;?xml version=&quot;1.0&quot; encoding=&quot;UTF-8&quot; standalone=&quot;yes&quot;?&gt;&#xA;&lt;OverrideOptions&gt;&#xA;    &lt;DEFAULT_OVERRIDE_OPTION&gt;None&lt;/DEFAULT_OVERRIDE_OPTION&gt;&#xA;    &lt;initialCellAttached&gt;default&lt;/initialCellAttached&gt;&#xA;    &lt;earfcnAttached&gt;100&lt;/earfcnAttached&gt;&#xA;    &lt;overrideOptionSelected&gt;NONE&lt;/overrideOptionSelected&gt;&#xA;&lt;/OverrideOptions&gt;&#xA;" />
    </Group>
  </Groups>
  <SessionList>
    <Session name="Registration" ACCESS_NETWORK="NR" InterarrivalTime="1000" RandomRate="false" Repetitions="1" SessionRepetitionsMode="Repetitions" groupRate="1000" groupRep="1" groupBlockLoop="false" groupMode="Variable Rate" SessionMode="Normal" SchedulingKind="Timer Triggered" StartSession="000808-IP_PT0012Registration00000000000100000" loop="false" Group="08-IP_PT">
      <TestTraceManager masterTraceLength="7" traces="1">
        <TestTrace>
          <TemplateProfile name="Delay" instance="" Mandatory="false" SuccessPercentage="100.0" DELAY="1000" />
          <TemplateProfile name="Registration" instance="" Mandatory="true" SuccessPercentage="100.0" />
          <TemplateProfile name="PDU Session Establish" instance="" Mandatory="false" SuccessPercentage="100.0" NR_PDU_SESSION_ID="1" MAX_DATA_RATE_UP_INT_PROT_UL="64 kbps" MAX_DATA_RATE_UP_INT_PROT_DL="64 kbps" IPCAN_TYPE="IPv4" NR_PDU_SESSION_SSC3_ID="" ALWAYS_ON_PDU_SESSION="(Default) Not Present" PDU_TYPE="IPv4" NRSM_INFO_TRANSFER_FLAG="true" NR_S_NSSAI="" IPCAN_NR_S_NSSAI="" NR_SSC_MODE="SSC Mode 1" APN="$(APN)" PCO="" SM_INFO_TRANSFER_FLAG="false" REQUEST_TYPE="InitialRequest" EAP_IDENTITY="" EAP_ALGORITHM="MD5" EAP_USERNAME="" EAP_PASSWORD="" DN_SPEC_ID_IN_CONTAINER="false" PDN_TYPE="IPv4" ESM_INFO_TRANSFER_FLAG="false" />
          <TemplateProfile name="Ping" instance="" Mandatory="false" SuccessPercentage="100.0" BearerType="Default bearer" BEARER_TAG="NONE" T_ACT_OFFSET="0" DEACT_FLAG_ENUM="Bearer not deactivated" TIME_TO_WAIT="100" BearerType_VIDEO="Default bearer" BEARER_TAG_VIDEO="NONE" T_ACT_OFFSET_VIDEO="0" DEACT_FLAG_ENUM_VIDEO="Bearer not deactivated" TIME_TO_WAIT_VIDEO="100" FlowType="Default flow" FLOW_TAG="NONE" T_ACT_OFFSET_5GC="0" DEACT_FLAG_ENUM_5GC="Flow not deactivated" TIME_TO_WAIT_FLOW="100" FlowType_VIDEO="Default flow" FLOW_TAG_VIDEO="NONE" T_ACT_OFFSET_5GC_VIDEO="0" DEACT_FLAG_ENUM_5GC_VIDEO="Flow not deactivated" TIME_TO_WAIT_VIDEO_FLOW="100" APN="default" NR_PDU_SESSION_ID="1" PING_IP="10.5.6.14" PING_TOUT="20" PING_NUM="10" PING_INT="1" PING_PATTERN="" PING_SIZE="560" PING_TTL="20" PING_THR="100" />
          <TemplateProfile name="UDG Monodirectional transmission Downlink" instance="" Mandatory="false" SuccessPercentage="100.0" BearerType="Default bearer" BEARER_TAG="NONE" T_ACT_OFFSET="0" DEACT_FLAG_ENUM="Bearer not deactivated" TIME_TO_WAIT="100" BearerType_VIDEO="Default bearer" BEARER_TAG_VIDEO="NONE" T_ACT_OFFSET_VIDEO="0" DEACT_FLAG_ENUM_VIDEO="Bearer not deactivated" TIME_TO_WAIT_VIDEO="100" FlowType="Default flow" FLOW_TAG="NONE" T_ACT_OFFSET_5GC="0" DEACT_FLAG_ENUM_5GC="Flow not deactivated" TIME_TO_WAIT_FLOW="100" FlowType_VIDEO="Default flow" FLOW_TAG_VIDEO="NONE" T_ACT_OFFSET_5GC_VIDEO="0" DEACT_FLAG_ENUM_5GC_VIDEO="Flow not deactivated" TIME_TO_WAIT_VIDEO_FLOW="100" APN="Default APN" traffic_type="Individual UE throughput configuration" EXP_THROUGHPUT_UL_Simpl="0.0" EXP_THROUGHPUT_DL_Simpl="0.0" EXP_THROUGHPUT_UL="0.0" EXP_THROUGHPUT_DL="0.77" USR_PKT_LEN_TYPE="Fixed" USR_PKT_LEN="100" USR_PKT_LEN_Simpl="100" USR_PKT_LEN_MIN="60" USR_PKT_LEN_MAX="140" NET_PKT_LEN_TYPE="Fixed" NET_PKT_LEN="48" NET_PKT_LEN_Simpl="100" NET_PKT_LEN_MIN="60" NET_PKT_LEN_MAX="140" ProtocolLayer="IP" REMOTE_PORT="50100" UUDG_NET_IP="10.5.6.14" UUDG_NET_IPV6="2222::22" DELAY="1000" NUM_TRAN="1" TRAN_TIME="100" TRAN_T_INT="100" IDLE_OUT_TIME="0" IDLE_OUT_SIDE="UE Not Synchronous" RESTART_ON_ABORT="false" ENABLE_ETH_PDU_SESSION="false" ADDR_TYPE="IPv4" UUDG_IPV4_ETH_ADDR="22.22.22.22" UUDG_IPV6_ETH_ADDR="2222::22" UE_MAC_ADDRESS="000000000000" NET_MAC_ADDRESS="000000000000" NR_PDU_SESSION_ID="1" NET_N_PKT="1" NET_PKT_T_INT="500" NET_PKT_T_INT_UNIT="Millisecond" />
          <TemplateProfile name="Delay" instance="" Mandatory="false" SuccessPercentage="100.0" DELAY="2000" />
          <TemplateProfile name="Deregistration" instance="" Mandatory="false" SuccessPercentage="100.0" DETACH_SWOFF_IRAT="false" DETACH_SWOFF="false" DEREGISTRATION_SWOFF="true" />
        </TestTrace>
      </TestTraceManager>
    </Session>
  </SessionList>
  <globals />
  <SmartphoneSimulator userLibraryFilename="" />
  <cell_list />
  <func_low_list />
  <MdmCellBinding>&lt;?xml version="1.0" encoding="UTF-8" standalone="yes"?&gt;
&lt;RadioConditionCellBinding&gt;
    &lt;cellBindingMap&gt;
        &lt;entry&gt;
            &lt;key&gt;New Cell 0&lt;/key&gt;
            &lt;value&gt;11&lt;/value&gt;
        &lt;/entry&gt;
        &lt;entry&gt;
            &lt;key&gt;New Cell 1&lt;/key&gt;
            &lt;value&gt;Simulated&lt;/value&gt;
        &lt;/entry&gt;
    &lt;/cellBindingMap&gt;
    &lt;autoCellBinding&gt;false&lt;/autoCellBinding&gt;
&lt;/RadioConditionCellBinding&gt;</MdmCellBinding>
  <MdmCellBindingNr>&lt;?xml version="1.0" encoding="UTF-8" standalone="yes"?&gt;
&lt;RadioConditionNrCellBinding&gt;
    &lt;nrCellBindingMap&gt;
        &lt;entry&gt;
            &lt;key&gt;New 5GNr Cell 2&lt;/key&gt;
            &lt;value&gt;1&lt;/value&gt;
        &lt;/entry&gt;
        &lt;entry&gt;
            &lt;key&gt;New 5GNr Cell 3&lt;/key&gt;
            &lt;value&gt;2&lt;/value&gt;
        &lt;/entry&gt;
    &lt;/nrCellBindingMap&gt;
&lt;/RadioConditionNrCellBinding&gt;</MdmCellBindingNr>
  <AmmExternalLinkedFiles rclibexternallinkedfilename="1N_1L_RadioConditions.rclib" />
</TestScenario>

