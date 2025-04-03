import sys
import time
import select

import numpy as np

import xapp_sdk as ric
import time
from clickhouse_driver import Client
from datetime import datetime, timezone

####################
#### Configuration Constants
####################
DATALAKE_URL = 'localhost'
DATALAKE_PORT = 9000
DATALAKE_USER = 'default'
DATALAKE_PASSWORD = ''
DATALAKE_DB_NAME = 'default'
DATALAKE_TABLE_MAC_NAME = 'MAC_KPIs'
DATALAKE_TABLE_RLC_NAME = 'RLC_KPIs'
DATALAKE_TABLE_PDCP_NAME = 'PDCP_KPIs'
DATALAKE_TABLE_GTP_NAME = 'GTP_KPIs'
DATALAKE_PUSH_INTERVAL = 0.05

####################
#### Base Callback Class for ClickHouse
####################

class BaseCallback:
    def __init__(self, database, table_name):
        self.client = Client(
            host=DATALAKE_URL,
            port=DATALAKE_PORT,
            user=DATALAKE_USER,
            password=DATALAKE_PASSWORD,
        )
        self.database = database
        self.table_name = table_name
        self.ensure_table_exists()
        self.push_time = time.time()
        self.log_time = time.time()

    def ensure_table_exists(self):
        try:
            # Check if the database exists
            self.client.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            
            # Check if the table exists
            table_exists = self.client.execute(
                f"EXISTS TABLE {self.database}.{self.table_name}"
            )[0][0]
            
            if not table_exists:
                # If table does not exist, create it
                self.create_table()
            else:
                print(f"Table {self.table_name} already exists.")
        except Exception as e:
            print(f"Error ensuring table exists: {e}")
            sys.exit(1)

    def create_table(self):
        """Override in subclasses to define specific table schemas."""
        pass

    def push_to_datalake(self, data):
        if time.time() - self.push_time > DATALAKE_PUSH_INTERVAL:
            self.push_time = time.time()

            if time.time() - self.log_time > 1.0:
                self.log_time = time.time()
                print(f"Insert to database {self.database}: {data}")
            try:
                self.client.execute(
                    f"INSERT INTO {self.database}.{self.table_name} VALUES",
                    data
                )
            except Exception as e:
                print(f"Error while inserting data into ClickHouse: {e}")

    def get_current_timestamps(self):
        # Get system timestamps in datetime64(9) format
        now = datetime.now(timezone.utc)
        ts_tai_ns = now
        ts_sw_ns = now
        return ts_tai_ns, ts_sw_ns

####################
#### MAC INDICATION CALLBACK
####################

class MACCallback(BaseCallback, ric.mac_cb):
    def __init__(self, database, table_name):
        BaseCallback.__init__(self, database, table_name)
        ric.mac_cb.__init__(self)

    def create_table(self):
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.table_name} (
            TsTaiNs DateTime64,

            dl_aggr_tbs UInt64,
            ul_aggr_tbs UInt64,
            dl_aggr_bytes_sdus UInt64,
            ul_aggr_bytes_sdus UInt64,
            dl_curr_tbs UInt64,
            ul_curr_tbs UInt64,
            dl_sched_rb UInt32,
            ul_sched_rb UInt32,

            pusch_snr Float32,
            pucch_snr Float32,

            dl_bler Float32,
            ul_bler Float32,

            # dl_harq_0 UInt32,
            # dl_harq_1 UInt32,
            # dl_harq_2 UInt32,
            # dl_harq_3 UInt32,
            # dl_harq_4 UInt32,
            # ul_harq_0 UInt32,
            # ul_harq_1 UInt32,
            # ul_harq_2 UInt32,
            # ul_harq_3 UInt32,
            # ul_harq_4 UInt32,

            rnti UInt32,
            dl_aggr_prb UInt32,
            ul_aggr_prb UInt32,
            dl_aggr_sdus UInt32,
            ul_aggr_sdus UInt32,
            dl_aggr_retx_prb UInt32,
            ul_aggr_retx_prb UInt32,

            bsr UInt32,
            frame UInt32,
            slot UInt32,

            wb_cqi UInt8,
            dl_mcs1 UInt8,
            ul_mcs1 UInt8,
            dl_mcs2 UInt8,
            ul_mcs2 UInt8,
            phr Int8,

            rsrp Int32,

            latency Float64,
        ) ENGINE = MergeTree()
        ORDER BY (TsTaiNs)
        SETTINGS index_granularity = 8192;
        """
        self.client.execute(create_table_query)

    def handle(self, ind):
        if len(ind.ue_stats) > 0:
            t_now = time.time_ns() / 1_000_000
            t_mac = ind.tstamp / 1_000
            t_diff = t_now - t_mac
            ts_tai_ns, ts_sw_ns = self.get_current_timestamps()
            # print(ul_harq)
            data_to_insert = [
                {
                    "TsTaiNs": ts_tai_ns,

                    "dl_aggr_tbs": ue_stat.dl_aggr_tbs,
                    "ul_aggr_tbs": ue_stat.ul_aggr_tbs,
                    "dl_aggr_bytes_sdus": ue_stat.dl_aggr_bytes_sdus,
                    "ul_aggr_bytes_sdus": ue_stat.ul_aggr_bytes_sdus,
                    "dl_curr_tbs": ue_stat.dl_curr_tbs,
                    "ul_curr_tbs": ue_stat.ul_curr_tbs,
                    "dl_sched_rb": ue_stat.dl_sched_rb,
                    "ul_sched_rb": ue_stat.ul_sched_rb,

                    "pusch_snr": ue_stat.pusch_snr,
                    "pucch_snr": ue_stat.pucch_snr,

                    "dl_bler": ue_stat.dl_bler,
                    "ul_bler": ue_stat.ul_bler,

                    # "dl_harq_0": dl_harq[0],
                    # "dl_harq_1": dl_harq[1],
                    # "dl_harq_2": dl_harq[2],
                    # "dl_harq_3": dl_harq[3],
                    # "dl_harq_4": dl_harq[4],

                    # "ul_harq_0": ul_harq[0],
                    # "ul_harq_1": ul_harq[1],
                    # "ul_harq_2": ul_harq[2],
                    # "ul_harq_3": ul_harq[3],
                    # "ul_harq_4": ul_harq[4],

                    "rnti": ue_stat.rnti,
                    "dl_aggr_prb": ue_stat.dl_aggr_prb,
                    "ul_aggr_prb": ue_stat.ul_aggr_prb,
                    "dl_aggr_sdus": ue_stat.dl_aggr_sdus,
                    "ul_aggr_sdus": ue_stat.ul_aggr_sdus,
                    "dl_aggr_retx_prb": ue_stat.dl_aggr_retx_prb,
                    "ul_aggr_retx_prb": ue_stat.ul_aggr_retx_prb,

                    "bsr": ue_stat.bsr,
                    "frame": ue_stat.frame,
                    "slot": ue_stat.slot,

                    "wb_cqi": ue_stat.wb_cqi,
                    "dl_mcs1": ue_stat.dl_mcs1,
                    "ul_mcs1": ue_stat.ul_mcs1,
                    "dl_mcs2": ue_stat.dl_mcs2,
                    "ul_mcs2": ue_stat.ul_mcs2,
                    "phr": ue_stat.phr,
                    
                    "rsrp": ue_stat.rsrp,

                    "latency": t_diff,
                }
                for ue_stat in ind.ue_stats
            ]

            self.push_to_datalake(data_to_insert)

####################
#### RLC INDICATION CALLBACK
####################

class RLCCallback(BaseCallback, ric.rlc_cb):
    def __init__(self, database, table_name):
        BaseCallback.__init__(self, database, table_name)
        ric.rlc_cb.__init__(self)

    def create_table(self):
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.table_name} (
            rnti UInt32,
            mode UInt8,
            rbid UInt8,
            txpdu_pkts UInt32,
            txpdu_bytes UInt64,
            rxpdu_pkts UInt32,
            rxpdu_bytes UInt64,
            txsdu_pkts UInt32,
            txsdu_bytes UInt64,
            rxsdu_pkts UInt32,
            rxsdu_bytes UInt64,
            tstamp Float64,
            latency Float64,
            TsTaiNs DateTime64(9),
            TsSwNs DateTime64(9)
        ) ENGINE = MergeTree()
        ORDER BY (TsSwNs, rnti, TsTaiNs)
        SETTINGS index_granularity = 8192;
        """
        self.client.execute(create_table_query)

    def handle(self, ind):
        if len(ind.rb_stats) > 0:
            t_now = time.time_ns() / 1_000_000
            t_rlc = ind.tstamp / 1_000
            t_diff = t_now - t_rlc
            ts_tai_ns, ts_sw_ns = self.get_current_timestamps()
            data_to_insert = [
                {
                    "rnti": rb_stat.rnti,
                    "mode": rb_stat.mode,
                    "rbid": rb_stat.rbid,
                    "txpdu_pkts": rb_stat.txpdu_pkts,
                    "txpdu_bytes": rb_stat.txpdu_bytes,
                    "rxpdu_pkts": rb_stat.rxpdu_pkts,
                    "rxpdu_bytes": rb_stat.rxpdu_bytes,
                    "txsdu_pkts": rb_stat.txsdu_pkts,
                    "txsdu_bytes": rb_stat.txsdu_bytes,
                    "rxsdu_pkts": rb_stat.rxsdu_pkts,
                    "rxsdu_bytes": rb_stat.rxsdu_bytes,
                    "tstamp": t_rlc,
                    "latency": t_diff,
                    "TsTaiNs": ts_tai_ns,
                    "TsSwNs": ts_sw_ns,
                }
                for rb_stat in ind.rb_stats
            ]

            self.push_to_datalake(data_to_insert)

####################
#### PDCP INDICATION CALLBACK
####################

class PDCPCallback(BaseCallback, ric.pdcp_cb):
    def __init__(self, database, table_name):
        BaseCallback.__init__(self, database, table_name)
        ric.pdcp_cb.__init__(self)
    def create_table(self):
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.table_name} (
            txpdu_pkts UInt32,
            txpdu_bytes UInt64,
            rxpdu_pkts UInt32,
            rxpdu_bytes UInt64,
            rnti UInt32,
            mode UInt8,
            rbid UInt8,
            tstamp Float64,
            latency Float64,
            TsTaiNs DateTime64(9),
            TsSwNs DateTime64(9)
        ) ENGINE = MergeTree()
        ORDER BY (TsSwNs, rnti, TsTaiNs)
        SETTINGS index_granularity = 8192;
        """
        self.client.execute(create_table_query)

    def handle(self, ind):
        if len(ind.rb_stats) > 0:
            t_now = time.time_ns() / 1_000_000
            t_pdcp = ind.tstamp / 1_000
            t_diff = t_now - t_pdcp
            ts_tai_ns, ts_sw_ns = self.get_current_timestamps()
            data_to_insert = [
                {
                    "txpdu_pkts": rb_stat.txpdu_pkts,
                    "txpdu_bytes": rb_stat.txpdu_bytes,
                    "rxpdu_pkts": rb_stat.rxpdu_pkts,
                    "rxpdu_bytes": rb_stat.rxpdu_bytes,
                    "rnti": rb_stat.rnti,
                    "mode": rb_stat.mode,
                    "rbid": rb_stat.rbid,
                    "tstamp": t_pdcp,
                    "latency": t_diff,
                    "TsTaiNs": ts_tai_ns,
                    "TsSwNs": ts_sw_ns,
                }
                for rb_stat in ind.rb_stats
            ]
            self.push_to_datalake(data_to_insert)

####################
#### GTP INDICATION CALLBACK
####################

class GTPCallback(BaseCallback, ric.gtp_cb):
    def __init__(self, database, table_name):
        BaseCallback.__init__(self, database, table_name)
        ric.gtp_cb.__init__(self)

    def create_table(self):
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.table_name} (
            rnti UInt32,
            teid_gnb UInt32,
            teid_upf UInt32,
            qfi UInt8,
            amf_ue_ngap_id UInt64,
            tstamp Float64,
            latency Float64,
            TsTaiNs DateTime64(9),
            TsSwNs DateTime64(9)
        ) ENGINE = MergeTree()
        ORDER BY (TsSwNs, rnti, TsTaiNs)
        SETTINGS index_granularity = 8192;
        """
        self.client.execute(create_table_query)

    def handle(self, ind):
        if len(ind.gtp_stats) > 0:
            t_now = time.time_ns() / 1_000_000
            t_gtp = ind.tstamp / 1_000
            t_diff = t_now - t_gtp
            ts_tai_ns, ts_sw_ns = self.get_current_timestamps()
            data_to_insert = [
                {
                    "rnti": gtp_stat.rnti,
                    "teid_gnb": gtp_stat.teidgnb,
                    "teid_upf": gtp_stat.teidupf,
                    "qfi": gtp_stat.qfi,
                    "amf_ue_ngap_id": gtp_stat.amf_ue_ngap_id,
                    "tstamp": t_gtp,
                    "latency": t_diff,
                    "TsTaiNs": ts_tai_ns,
                    "TsSwNs": ts_sw_ns,
                }
                for gtp_stat in ind.gtp_stats
            ]
            self.push_to_datalake(data_to_insert)            

####################
#### Main Logic
####################
def main():
    # Initialize RIC
    ric.init()
    conn = ric.conn_e2_nodes()

    if not conn:
        print("No connected E2 nodes found.")
        return

    # Initialize handlers
    mac_handlers, rlc_handlers, pdcp_handlers, gtp_handlers = [], [], [], []

    try:
        for node in conn:
            # Create and start MAC handler
            mac_cb = MACCallback(DATALAKE_DB_NAME, DATALAKE_TABLE_MAC_NAME)
            mac_handler = ric.report_mac_sm(node.id, ric.Interval_ms_10, mac_cb)
            mac_handlers.append(mac_handler)

            # Create and start RLC handler
            rlc_cb = RLCCallback(DATALAKE_DB_NAME, DATALAKE_TABLE_RLC_NAME)
            rlc_handler = ric.report_rlc_sm(node.id, ric.Interval_ms_10, rlc_cb)
            rlc_handlers.append(rlc_handler)

            # Create and start PDCP handler
            pdcp_cb = PDCPCallback(DATALAKE_DB_NAME, DATALAKE_TABLE_PDCP_NAME)
            pdcp_handler = ric.report_pdcp_sm(node.id, ric.Interval_ms_10, pdcp_cb)
            pdcp_handlers.append(pdcp_handler)

            # Create and start GTP handler
            # gtp_cb = GTPCallback(DATALAKE_DB_NAME, DATALAKE_TABLE_GTP_NAME)
            # gtp_handler = ric.report_gtp_sm(node.id, ric.Interval_ms_10, gtp_cb)
            # gtp_handlers.append(gtp_handler)

        # Run loop
        while True:
            time.sleep(1)

            # Check if user pressed 'q'
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                user_input = sys.stdin.read(1).strip()
                if user_input.lower() == 'q':
                    print("\n'q' detected. Quitting...")
                    break

        # Stop MAC handlers
        for handler in mac_handlers:
            ric.rm_report_mac_sm(handler)

        # Stop RLC handlers
        for handler in rlc_handlers:
            ric.rm_report_rlc_sm(handler)

        # Stop PDCP handlers
        for handler in pdcp_handlers:
            ric.rm_report_pdcp_sm(handler)

        # Stop GTP handlers
        for handler in gtp_handlers:
            ric.rm_report_gtp_sm(handler)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nStopping all handlers...")

        # Stop MAC handlers
        for handler in mac_handlers:
            ric.rm_report_mac_sm(handler)

        # Stop RLC handlers
        for handler in rlc_handlers:
            ric.rm_report_rlc_sm(handler)

        # Stop PDCP handlers
        for handler in pdcp_handlers:
            ric.rm_report_pdcp_sm(handler)

        # Stop GTP handlers
        for handler in gtp_handlers:
            ric.rm_report_gtp_sm(handler)

        print("All handlers stopped. Exiting.")

if __name__ == "__main__":
    main()